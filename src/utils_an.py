import numpy as np
import sigpy as sp    
import matplotlib.pyplot as plt
import itertools
import pydicom as pydic
import os
import cupy as cp
from utils import convert_to_int16_range, Int16Range

# Function to perform direct GPU-to-GPU copy using cudaMemcpyPeer
# had issues with higher level transfer functions
def cuda_memcpy_peer(src, dst, src_device, dst_device):
    src_ptr = src.data.ptr  # Pointer to source memory
    dst_ptr = dst.data.ptr  # Pointer to destination memory
    size = src.nbytes       # Size of the data in bytes
    
    # Perform the peer-to-peer memory copy
    cp.cuda.runtime.memcpyPeer(dst_ptr, dst_device, src_ptr, src_device, size)

def copy_array_peer(arr, src_device, dst_device):
    with cp.cuda.Device(dst_device):
        arr_dst = cp.empty_like(arr)
        cuda_memcpy_peer(arr, arr_dst, src_device, dst_device)
        del arr
        arr = arr_dst
    return arr

def whiten(data_c_last_axis, mat, mode=''):
    """
    mode in ['cov', 'wht'] defines if cholesky is performed or not 
    """
    device = sp.get_device(data_c_last_axis)
    xp = device.xp
    if mode == 'cov':
        Psi = sp.to_device(mat, device)
        wht = xp.linalg.cholesky(xp.linalg.inv(Psi)).conj().T
    else:
        wht = sp.to_device(mat, device)
    return xp.matmul(data_c_last_axis, wht)

divs = lambda n: [x for x in range(2, min(24, n) + 1) if n/x == int(n/x)]

def clean_gpu(device, do_print=False):
    xp = device.xp
    with device:
        mempool = xp.get_default_memory_pool()
        mempool.free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()
        if do_print:
            print(f'Used on GPU: {mempool.used_bytes() / 1e6} MiB')
            print(f'Total on GPU pool {mempool.total_bytes() / 1e6} MiB')

def get_all_index_combinations(p):
    combis_inds = []
    for l in range(1, len(p) + 1):
        for comb in itertools.combinations(p, l):
            combis_inds.append(comb)
    return np.array(combis_inds, dtype=tuple)

def read_dicom_multislice(dcm_series_dir):
    img_xyz = []
    for file in sorted(os.listdir(dcm_series_dir)):
        filename = os.path.join(dcm_series_dir, file)
        img_xyz.append(pydic.dcmread(filename).pixel_array)
    img_xyz = np.array(img_xyz).transpose([1, 2, 0])
    return img_xyz


def pad_int16_img_xyzs(img_pad_xyzs):
    nx, ny, nz, ns = img_pad_xyzs.shape
    img_wfs_xyz_stackz = np.zeros((nx, ny, 3*nz), dtype=np.float32)
    for img_ii in range(3):
        img_wfs_xyz_stackz[:, :, img_ii*nz:(img_ii+1)*nz] = np.abs(img_pad_xyzs[..., img_ii]).astype(np.float32)

    img_wfs_xyz_stackz = convert_to_int16_range(img_wfs_xyz_stackz, Int16Range.HALF)[0]
    img_wfs_int16_xysz = np.reshape(img_wfs_xyz_stackz, (nx, ny, 3, nz))
    img_wfs_int16_xyzs  = img_wfs_int16_xysz.transpose((0, 1, 3, 2))
    return img_wfs_int16_xyzs

def blackman_window(N):
    x = np.arange(N)
    a = [0.42659, 0.49656, 0.076849]
    return (a[0] - a[1] * np.cos(2 * np.pi * x / N) + a[2] * np.cos(4 * np.pi * x / N))

def tukey_window(N, alpha=0.4):
    x = np.arange(N)
    w = np.zeros_like(x)
    n = int(alpha * N / 2)
    y = 0.5 * (1. - np.cos(2 * np.pi * x[:n] / (2 * n)))
    return np.concatenate((y, np.ones(N - 2 * n), y[::-1]))


def pdf_estimation_poisson(
    img_shape,
    accel,
    calib=(0, 0),
    tol=0.1,
):

    ny, nx = img_shape
    y, x = np.mgrid[:ny, :nx]
    x = np.maximum(abs(x - img_shape[-1] / 2) - calib[-1] / 2, 0)
    x /= x.max()
    y = np.maximum(abs(y - img_shape[-2] / 2) - calib[-2] / 2, 0)
    y /= y.max()
    r = np.sqrt(x**2 + y**2)

    slope_max = max(nx, ny)
    slope_min = 0
    while slope_min < slope_max:
        slope = (slope_max + slope_min) / 2
        pdf = 1/(1 + slope * np.abs(r))

        actual_accel = 1/np.mean(pdf, axis=(0,1))

        if abs(actual_accel - accel) < tol:
            break
        if actual_accel < accel:
            slope_min = slope
        else:
            slope_max = slope

    return pdf

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import torch

def nrmse(y_true, y_pred, normalization="range"):
    """
    Computes the Normalized Root Mean Square Error (NRMSE).
    
    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predicted values.
        normalization (str): Normalization method - "range" or "mean".
    
    Returns:
        torch.Tensor: The NRMSE value.
    """
    mse = torch.mean((y_true - y_pred) ** 2, dim=[-1, -2])
    rmse = torch.sqrt(mse)

    if normalization == "range":
        norm_factor = torch.amax(y_true, dim=[-1, -2]) - torch.amin(y_true, dim=[-1, -2])
    elif normalization == "mean":
        norm_factor = torch.mean(y_true, dim=[-1, -2])
    else:
        raise ValueError("Invalid normalization method. Use 'range' or 'mean'.")
    return rmse / norm_factor


def img_psnr(original, reconstructed):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.

    Parameters:
        original (ndarray): Original image as a 2D or 3D array.
        reconstructed (ndarray): Reconstructed image as a 2D or 3D array.
        max_pixel_value (int): Maximum possible pixel value (default: 255).

    Returns:
        float: PSNR value in decibels (dB).
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")  
    psnr = 10 * np.log10(original.max()**2 / mse)
    return psnr

def img_ssim(original, reconstructed):
    """
    Calculate SSIM (Structural Similarity Index) between two images.

    Parameters:
        original (ndarray): Original image as a 2D or 3D array.
        reconstructed (ndarray): Reconstructed image as a 2D or 3D array.

    Returns:
        float: SSIM value (ranges from -1 to 1).
    """
    ssim_value, _ = ssim(original, reconstructed, full=True, data_range=original.max() - original.min())
    return ssim_value

def get_poisson_sampling(ny, ne, N_acs, R_eff):
    D_yt = sp.mri.poisson((ny, 40), accel=R_eff, calib=(N_acs, 2*ne)).astype(np.complex64)
    D_yt = D_yt[:, 13:(13+ne)] 

    pdf = pdf_estimation_poisson((ny, 40), R_eff, calib=(N_acs, 2*ne))
    DCF = pdf[:, 13:(13+ne)]
    return D_yt, DCF

def get_uniform_sampling(ny, ne, N_acs, R_eff):
    D_yt = np.zeros((ny, ne), dtype=np.complex64)
    y_end_acs = (ny + N_acs) // 2 
    y_start_acs = (ny - N_acs) // 2
    D_yt[y_start_acs:0:-R_eff, :] = 1.
    D_yt[y_end_acs::R_eff, :] = 1.
    D_yt[y_start_acs:y_end_acs, :] = 1.

    DCF = np.ones((ny, ne)) * 1. / R_eff
    DCF[(ny-N_acs)//2:(ny+N_acs)//2+1, :] = 1.   
    return D_yt, DCF


def get_diagonal_sampling(ny, ne, N_acs, R_eff):
    D_yt = np.zeros((ny, ne), dtype=np.complex64)
    y_end_acs = (ny + N_acs) // 2 
    y_start_acs = (ny - N_acs) // 2
    D_yt[y_start_acs:y_end_acs, :] = 1.
    
    for i in range(ne):
        D_yt[y_start_acs-(i%R_eff):0:-R_eff, i] = 1.
        D_yt[y_end_acs+R_eff-(i%R_eff)::R_eff, i] = 1.
    DCF = np.ones((ny, ne)) * 1. / R_eff
    DCF[(ny-N_acs)//2:(ny+N_acs)//2+1, :] = 1.  
    return D_yt, DCF

from sigpy import backend
from sigpy.linop import _hstack_params, _vstack_params
class Diag_custom(sp.linop.Linop):
    """ Define Diagonally stack linear operators to allow for iaxis and oaxis=None.

    Create a Linop that splits input, applies linops independently,
    and concatenates outputs.
    In matrix form, given matrices {A1, ..., An}, returns diag([A1, ..., An]).

    Args:
        linops (list of Linops): list of linops with the same input and
            output shape.
        iaxis (int or None): If None, inputs are vectorized
            and concatenated.
        oaxis (int or None): If None, outputs are vectorized
            and concatenated.

    """

    def __init__(self, linops, oaxis=None, iaxis=None):
        self.nops = len(linops)

        self.linops = linops
        self.oaxis = oaxis
        self.iaxis = iaxis
        ishape, self.iindices = _hstack_params(
            [linop.ishape for linop in self.linops], iaxis
        )
        oshape, self.oindices = _vstack_params(
            [linop.oshape for linop in self.linops], oaxis
        )

        super().__init__(oshape, ishape)


    def _apply(self, input):
        device = backend.get_device(input)
        xp = device.xp
        with device:
            output = xp.empty(self.oshape, dtype=input.dtype)
            for n, linop in enumerate(self.linops):
                if n == 0:
                    istart = 0
                    ostart = 0
                else:
                    istart = self.iindices[n - 1]
                    ostart = self.oindices[n - 1]

                if n == self.nops - 1:
                    iend = None
                    oend = None
                else:
                    iend = self.iindices[n]
                    oend = self.oindices[n]

                if self.iaxis is None:
                    output_n = linop(
                        input[istart:iend].reshape(linop.ishape)
                    )
                else:
                    ndim = len(linop.ishape)
                    axis = self.iaxis % ndim
                    islc = tuple(
                        [slice(None)] * axis
                        + [slice(istart, iend)]
                        + [slice(None)] * (ndim - axis - 1)
                    )

                    output_n = linop(input[islc])

                if self.oaxis is None:
                    output[ostart:oend] = output_n.ravel()
                else:
                    ndim = len(linop.oshape)
                    axis = self.oaxis % ndim
                    oslc = tuple(
                        [slice(None)] * axis
                        + [slice(ostart, oend)]
                        + [slice(None)] * (ndim - axis - 1)
                    )

                    output[oslc] = output_n

            return output
        
    def _adjoint_linop(self):
        return Diag_custom(
            [op.H for op in self.linops], oaxis=self.iaxis, iaxis=self.oaxis
        )
    

def pad_images_xy(img_xyzdt, dims_pad_xyz):
    device = sp.get_device(img_xyzdt)
    xp = device.xp

    dims_xyz = img_xyzdt.shape[:3]

    ksp_pad_shape = tuple(list(dims_pad_xyz) + list(img_xyzdt.shape[3:]))
    ksp_pad_xyzdt = xp.zeros(ksp_pad_shape, dtype=np.complex64)

    ind_begin = []
    ind_end = []
    for dim_ii in range(2):
        ind_begin.append(dims_pad_xyz[dim_ii] // 2 - dims_xyz[dim_ii] // 2)
        ind_end.append(dims_pad_xyz[dim_ii] // 2 + dims_xyz[dim_ii] // 2)

    ksp_pad_xyzdt[ind_begin[0]: ind_end[0], ind_begin[1]:ind_end[1], :, :, :] = sp.fft(img_xyzdt, axes=[0, 1]) 
    img_pad_xyzdt = sp.ifft(ksp_pad_xyzdt, axes=[0, 1])
    return img_pad_xyzdt