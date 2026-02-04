import numpy as np
import sigpy as sp    

from sigpy import backend
from sigpy.linop import _hstack_params, _vstack_params

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


def tukey_window(N, alpha=0.4):
    x = np.arange(N)
    n = int(alpha * N / 2)
    y = 0.5 * (1. - np.cos(2 * np.pi * x[:n] / (2 * n)))
    return np.concatenate((y, np.ones(N - 2 * n), y[::-1]))


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


def get_cc_matrix(data_c_last_axis, n_coils):
    """
    Get the coil compression matrix by performing SVD along the last axis.
    
    Arguments
        data_c_last_axis : ndarray
            Dimensions (..., n_coils_initial)
        n_coils : int
            Number of compressed coil channels.
        
    Returns 
        cc_matrix : ndarray
            The compression matric of dimensions (n_coils_initial, n_coils)

    """
    xp = sp.get_array_module(data_c_last_axis)
    nc = data_c_last_axis.shape[-1]
    
    _, _, vh = xp.linalg.svd(xp.reshape(data_c_last_axis, (-1, nc)), full_matrices=False)
    v = xp.conj(vh.T)
    cc_matrix = v[:, 0:n_coils]

    return cc_matrix    


def apply_cc_matrix(data_c_last_axis, cc_matrix):
    CC = sp.linop.RightMatMul(data_c_last_axis.shape, cc_matrix)
    return CC * data_c_last_axis


def estimate_weights(ksp_data):
    """
    Estimate sampling mask from k-space data.

    Arguments
    ---------
    ksp_data : array
        K-space data with coil dimension first. Typical shape (nc, ...).

    Returns
    -------
    weights : array or None
        Estimated binary weights (rss > 0).
    """
    with sp.get_device(ksp_data):
        weights = (sp.rss(ksp_data, axes=(0,)) > 0).astype(ksp_data.dtype)
    return weights