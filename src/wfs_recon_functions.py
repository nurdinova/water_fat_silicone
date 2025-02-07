import numpy as np
import os
from sigpy import backend
import sigpy as sp
import matplotlib.pyplot as plt
import sigpy.mri as mr
import sigpy.plot as pl
from hmrGC.dixon_imaging import MultiEcho
from utils_an import Diag_custom

class TotalVariationRecon_WFS(sp.app.LinearLeastSquares):
    def __init__(self, ksp_cxyt, sens_cxyt, weights, fieldmap_xyt_rads=None, cse_matrix_ts=None, coord=None,
                 wave_name='db4', lamda_tv=0., lamda_ls=0., show_pbar=True, **kwargs):
        """ Subspace Reconstruction with TV regularization on water-fat-silicone proton densities. """
        device, xp = sp.get_device(ksp_cxyt), sp.get_device(ksp_cxyt).xp
        assert device == sp.get_device(sens_cxyt), "Please move all the data to one device"

        with device:
            weights = estimate_weights(ksp_cxyt, weights, coord)
            ksp_cxyt = sp.to_device(ksp_cxyt * weights ** 0.5, device) if weights is not None else ksp_cxyt

            if ksp_cxyt.ndim != 4:
                raise ValueError(f'Input to Subspace recon has shape {ksp_cxyt.shape}')
            sens_cxyt = sens_cxyt[..., None] if sens_cxyt.ndim == 3 else sens_cxyt

            nc, nx, ny, nt, ns = *ksp_cxyt.shape, cse_matrix_ts.shape[-1]
            x_shape = (ns, nx, ny)
            fieldmap_xyt_rads = xp.zeros((nx, ny, nt)) if fieldmap_xyt_rads is None else sp.to_device(fieldmap_xyt_rads, device)

            self.Transpose_x_xys = sp.linop.Transpose(x_shape, (1, 2, 0))
            self.Reshape_x_xysr = sp.linop.Reshape((nx, ny, ns, 1), self.Transpose_x_xys.oshape)
            self.E = sp.linop.MatMul(self.Reshape_x_xysr.oshape, cse_matrix_ts[None, None])
            P = sp.linop.Multiply(self.E.oshape, xp.exp(2j * xp.pi * fieldmap_xyt_rads[..., None]))
            Transpose_PEx_rxyt = sp.linop.Transpose(P.oshape, (3, 0, 1, 2))
            S = sp.linop.Multiply(Transpose_PEx_rxyt.oshape, sens_cxyt)
            F = sp.linop.FFT(S.oshape, axes=[1, 2]) if coord is None else sp.linop.NUFFT(S.oshape, coord)

            A = F * S * Transpose_PEx_rxyt * P * self.E * self.Reshape_x_xysr * self.Transpose_x_xys
            if weights is not None:
                A = sp.linop.Multiply(F.oshape, weights ** 0.5) * A

            # TV regularization setup
            if all(lamda_ == 0 for lamda_ in lamda_tv):
                G, proxg = None, None
            else:
                G = sp.linop.FiniteDifference(x_shape, axes=[1, 2])
                create_thresh_array = lambda inshape, lamda_: lamda_ * xp.ones(inshape)
                lambda_vector = xp.concatenate([create_thresh_array((2, 1, nx, ny), l) for l in lamda_tv], axis=1)
                proxg = sp.prox.L1Reg(G.oshape, lambda_vector)

            super().__init__(A, ksp_cxyt, proxg=proxg, G=G, solver='ADMM', lamda=lamda_ls, show_pbar=show_pbar, **kwargs)


def sense_multislice(ksp_cxyzt, coils_cxyz, slices_to_recon=None, max_iter=10, lamda=0., lamda_tv=0., 
                     weights=None, subspace_recon=False, fieldmap_xyzt_rads=None, cse_matrix_ts=None, 
                     grase_te_ms=None, img_init_xyzt=None):
    """ Calls sigpy SENSE slice-by-slice and echo-by-echo on GPU. Returns img_xyzt (ndarray on host). """
    xp, device = sp.get_array_module(ksp_cxyzt), backend.get_device(ksp_cxyzt)
    slices_to_recon = slices_to_recon or range(ksp_cxyzt.shape[3])

    with device:
        nc, nx, ny, nz, nt = *ksp_cxyzt.shape[:4], 1 if subspace_recon else ksp_cxyzt.shape[-1]
        if subspace_recon:
            ksp_cxyzt = ksp_cxyzt[..., None]
            weights = weights[..., None] if weights is not None else None
            img_sense_xyzt = xp.zeros((cse_matrix_ts.shape[-1], nx, ny, nz, 1), ksp_cxyzt.dtype)
        else:
            img_sense_xyzt = xp.zeros(ksp_cxyzt.shape[1:], ksp_cxyzt.dtype)

        for z_ii in slices_to_recon:
            for t_ii in range(nt):
                P = weights[:, :, :, z_ii, ..., t_ii] if weights is not None else None
                if subspace_recon:
                    sense = TotalVariationRecon_WFS(ksp_cxyzt[:, :, :, z_ii, :, t_ii], coils_cxyz[:, :, :, z_ii], 
                        weights=P, fieldmap_xyt_rads=fieldmap_xyzt_rads[:, :, z_ii, :], cse_matrix_ts=cse_matrix_ts, 
                        max_iter=max_iter, show_pbar=False, lamda_tv=lamda_tv, lamda_ls=lamda)
                else:
                    sense = mr.app.TotalVariationRecon(ksp_cxyzt[:, :, :, z_ii, t_ii], coils_cxyz[:, :, :, z_ii], 
                        weights=P, max_iter=max_iter, show_pbar=False, lamda=lamda, device=device)
                img_sense_xyzt[..., z_ii, t_ii] = xp.squeeze(sense.run())

    return img_sense_xyzt


from enum import Enum
from scipy.ndimage import gaussian_filter, gaussian_filter1d 

class SensitivityEstimationType(Enum):
    ESPIRIT = 0
    LOWRES = 1
        
def estimate_coils(ksp_cxyz, n_acs=32, sens_estim_method=SensitivityEstimationType.ESPIRIT,\
        volume_espirit="2d", slices_to_recon=None, max_iter=20, espirit_thresh=2e-2, crop_thresh=0.95,\
        lowres_gaus_sigma=5, lowres_gaus_axes=None, lowres_gaus_radius=None):
    """ 
    Input:
        ksp_calib_cxyz : ndarray, complex
            calibration k-space data should be given at the spin echo
        ksp_cxyz_shape : tuple
            tuple containing shapes of the fully-sampled data
    
    Returns:
        coils_cxyz : ndarray, complex
            coil sensitivity maps estimated from ESPIRIT
        
    """
    device = backend.get_device(ksp_cxyz) 
    xp = sp.get_array_module(ksp_cxyz)

    nc_, nx_full, ny_full, nz_full = ksp_cxyz.shape
    coils_cxyz_ = xp.zeros((nc_, nx_full, ny_full, nz_full), complex)
         
    if (sens_estim_method == SensitivityEstimationType.ESPIRIT):

        if (volume_espirit == "2d"):
            
            if not slices_to_recon:
                slices_to_recon = range(nz_full)
            
            # run ESPIRIT on the padded calibration k-space slice-by-slice
            
            for slice_recon in slices_to_recon:
                ksp_calib_cxy_device = ksp_cxyz[..., slice_recon]
                with device:
                    coils_cxyz_[..., slice_recon] = mr.app.EspiritCalib( \
                        ksp_calib_cxy_device, calib_width=n_acs, device=device, show_pbar=False, \
                        max_iter=max_iter, thresh=espirit_thresh, crop=crop_thresh, kernel_width=5).run()
                    
        elif (volume_espirit == "3d"):
            
            with device:
                ksp_calib_cxyz_device = sp.fft(ksp_calib_cxyz, device, axes=[-1])
                coils_cxyz_ = mr.app.EspiritCalib(ksp_calib_cxyz_device, \
                    device=device, show_pbar=False, max_iter=max_iter, thresh=espirit_thresh).run()
        else:
            raise NotImplementedError
        
    elif (sens_estim_method == SensitivityEstimationType.LOWRES):       
        # crop the calibration region and pad with zeros to match the full matrix
        ksp_calib_cxyz = ksp_cxyz[:, nx_full//2-n_acs//2:nx_full//2+n_acs//2, ny_full//2-n_acs//2:ny_full//2+n_acs//2, :]
        ksp_calib_cxyz_pad = np.pad( ksp_calib_cxyz, ((0, 0), \
                ((nx_full - n_acs + 1) // 2, (nx_full - n_acs) // 2), ((ny_full - n_acs + 1) // 2, (ny_full - n_acs) // 2), \
                ((nz_full - n_acs + 1) // 2, (nz_full - n_acs) // 2)) )

        # gaussian filter in image space
        coils_cxyz_ = sp.ifft(ksp_calib_cxyz_pad, axes=[1,2])
        for ax_ in lowres_gaus_axes:
            coils_cxyz_ = gaussian_filter1d(coils_cxyz_, axis=ax_, \
                sigma=lowres_gaus_sigma, radius=lowres_gaus_radius)
        
    return coils_cxyz_ 

def sort_bipolar_interleaves(data_dt):
    device = sp.get_device(data_dt)
    xp = device.xp
    n_sets_two = data_dt.shape[-2] // 2
    with device:
        for i in range(n_sets_two):
            data_copy = xp.copy(data_dt[..., 2*i, 1::2])
            data_dt[..., 2*i, 1::2] = xp.copy(data_dt[..., 2*i+1, 1::2])
            data_dt[..., 2*i+1, 1::2] = xp.copy(data_copy)
            del data_copy
    return data_dt

def conj_coil_combination(img_cxyzt, coils_cxyz):
    # conjugate-phase coil combination
    return np.sum(np.conj(coils_cxyz)[...,np.newaxis] * (img_cxyzt), 0)
   
        
def Herm(arr):
    # Hermitian operator on numpy arrays
    return arr.swapaxes(-2, -1).conj()
     
def senseweights(coil_sens_cr, Psi=None):
    """
	Function calculates the SENSE weights at a pixel given
	input sensitivities and covariance matrix.  A reduction
	factor R can be assumed from the shape of the sens matrix.
	The number of coils Nc is determined by the size of coil_sens.  
    If Psi is omitted it is assumed to be the identity matrix.

	Arguments:
		coil_sens_cr = Nc x R matrix of coil sensitivites 
		Psi = Nc x Nc noise covariance matrix.

	Returns:
		weights_rc = coil combination weights - R x Nc
		gfact_r = gfactor (1xR vector including each aliased pixel)
		imnoise_cov = image noise covariance matrix (RxR)
  
    Revised code of bhargreaves from RAD-229
    """
    Nc, R = coil_sens_cr.shape
    if Psi is None:
        Psi = np.eye(Nc)
        
    SH_invPsi = Herm(coil_sens_cr) @ np.linalg.inv(Psi)
    SH_invPsi_S = SH_invPsi @ coil_sens_cr
    inv_SH_invPsi_S = np.linalg.inv(SH_invPsi_S)
    
    weights_rc = inv_SH_invPsi_S @ SH_invPsi
    imnoise_cov = np.sqrt(inv_SH_invPsi_S)
    gfact_r = np.sqrt(np.diagonal(inv_SH_invPsi_S) * np.diagonal(SH_invPsi_S))      
        
    return weights_rc, gfact_r, imnoise_cov

def calculate_T2_map(im_vzexy, TEs, max_T2=200):
    assert im_vzexy.shape[0] == 2
    mask_zexy = np.abs(im_vzexy[0]) > 2.2e-2
    
    denominator = np.copy(np.abs(im_vzexy[1, ...]))
    denominator = denominator * mask_zexy
    denominator[denominator == 0] = 1
    
    numerator = np.copy(np.abs(im_vzexy[0, ...]))
    numerator[numerator == 0] = 1
    log_ratio_image = np.log(numerator / denominator)
    log_ratio_image[log_ratio_image == 0] = 1

    T2_map_zt = mask_zexy * ((TEs[1] - TEs[0]) / log_ratio_image)
    T2_map_zt[np.abs(T2_map_zt) > max_T2] = max_T2
    
    return T2_map_zt


def run_wfs(data_xyzt, grase_echo_times, method="WFS_DIET", range_fm=[-2, 2], lamda=1e-2, \
    te_ii=None, get_pdff=False, perform_method="single-res", voxelsize_mm=None, \
    N_peaks=1, plot=False, save_plots=False, fig_name='', plot_mask=False, verbose=True):
    
    device = sp.get_device(data_xyzt)
    xp = device.xp

    params = {}
    # new model params
    params['signal_model'] = method

    if te_ii is None:
        te_ii = 0 # to normalise the data by
        if verbose:
            print('Spin echo is not specified. Normalizing images to the first image')
    magn_threshold = 5e-2 # percent, 0.02

    number_fat_peaks = N_peaks
    if params['signal_model'] == 'WFS':
        signal_xy_et = data_xyzt
        img_keys = ["water", "fat", "silicone", "fieldmap", "r2primemap"]
        img_titles = ['Water', 'Fat', 'Silicone', 'Field map, Hz', 'R2prime map, Hz']
        vmax_scales = [4, 1, 2]
    elif params['signal_model'] == 'WF':
        signal_xy_et = data_xyzt
        img_keys = ["water", "fat", "fieldmap", "r2primemap"] 
        img_titles = ['Water', 'Fat', 'Field map, Hz', 'R2prime map, Hz']
        vmax_scales = [4, 1]
    else:
        model_str = params['signal_model']
        raise ValueError(f'Model {model_str} is not known.')
    n_imgs = len(img_keys)
        
    nx, ny, nz = signal_xy_et.shape[:3]
    signal_xzy_et = signal_xy_et.reshape((nx, ny, nz, -1))
    if verbose:
        print("Input Signal shape ", signal_xzy_et.shape)

    norm_factor = xp.max(xp.abs(signal_xzy_et[..., te_ii]))
    signal_xzy_et_norm = signal_xzy_et / norm_factor
    mip = xp.sqrt(xp.sum(xp.abs(signal_xzy_et_norm) ** 2, axis=-1))
    mask = mip > magn_threshold * xp.max(mip)

    if plot_mask:
        mask = sp.to_device(mask, sp.Device(-1))
        signal_to_plot = sp.to_device(signal_xzy_et[:,:, 0, 0], sp.Device(-1))
        fig, axs = plt.subplots(1, 2, figsize=(6,3))
        axs[0].imshow(mask[:, :, 0])
        axs[1].imshow(np.abs(signal_to_plot))
    
    # params['period'] = sequence_opts[1]['dTE_ms'] * 1e-3
    params['TE_s'] = grase_echo_times * 1e-3   # float array with dim (n_se, n_gre)
    params['centerFreq_Hz'] = 42577478 * 3.   # float
    params['fieldStrength_T'] = 3.   # float
    params['voxelSize_mm'] = [1, 1, 1] if voxelsize_mm is None else list(voxelsize_mm)   # recon voxel size with dim (3)

    params['FatModel'] = {}
    if number_fat_peaks == 9:
        params['FatModel']['freqs_ppm']  = np.array([-3.8 , -3.4 , -3.1 , -2.68, -2.46, -1.95, -0.5 ,  0.49,  0.59])
        params['FatModel']['relAmps'] = np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006,\
            0.01498501, 0.03996004, 0.00999001, 0.05694306])
    elif number_fat_peaks == 7:
        # measured breast fat model
        params['FatModel']['freqs_ppm'] = np.array([-484, -435, -331, -300, -230, -60, 90]) * 1e6 / params['centerFreq_Hz']
        params['FatModel']['relAmps'] = np.array([0.14583333, 0.39583333, 0.10416667, 0.10416667, 0.02083333, \
       0.14583333, 0.08333333])
    elif number_fat_peaks == 6:
        params['FatModel']['freqs_ppm']  = np.array([-3.8 , -3.4 , -3.1 , -2.68, -2.46, -1.95])
        params['FatModel']['relAmps'] = np.array([0.08991009+0.03996004, 0.58341658+0.05694306, 0.05994006, 0.08491508, 0.05994006,\
            0.01498501+0.00999001])
    elif number_fat_peaks == 3:
        params['FatModel']['freqs_ppm']  = np.array([-3.8 , -3.4 , -3.1])
        params['FatModel']['relAmps'] = np.array([0.08991009+0.03996004, 0.58341658+0.05694306, 0.05994006+0.08491508+0.05994006+0.01498501+0.00999001])
    elif number_fat_peaks == 1:
        params['FatModel']['freqs_ppm']  = [-3.4] 
        params['FatModel']['relAmps'] = [1.] 
    else:
        raise KeyError(f'Fat model with {number_fat_peaks} peaks is not defined')
    
    if params['signal_model'] == 'WFS':
        params['siliconePeak_ppm'] = [-4.4]
    
    params['range_fm_ppm'] = range_fm
    g = MultiEcho(signal_xzy_et, mask, params)
    g.options["reg_param"] = lamda

    # seems to not clean up after launch on gpu
    g.use_gpu = False
    
    g.r2star_correction = True   # modify runtime options, e.g. R2star correction for images
    g.verbose = verbose
    g.perform(perform_method) #  single-res, breast, multi-res

    if perform_method == "fieldmap":
        images = None
    else:
        images = []
        for img_str in img_keys:
            if img_str == "fieldmap":
                images.append(g.fieldmap)
            elif img_str == "r2primemap":
                images.append(g.r2starmap)
            else:
                images.append((g.images[img_str]))
        if get_pdff:
            pdff = g.images['fatFraction_percent']
            mask = (pdff < 0) + (pdff > 125)
            pdff[mask] = 0
            images.append(pdff)

        if plot:
            n_rows = 2
            n_cols = (n_imgs + n_rows - 1) // n_rows
            for z_ii in range(nz):
                fig, axs = plt.subplots(n_cols, n_rows, figsize=(6,6))
                max_val = 0.8 * np.max(np.abs(np.array(images[:len(vmax_scales)])))
                for img_ii, img_str in enumerate(img_keys):
                    if (img_str in ["fieldmap", "r2primemap"]):
                        cmp = axs[img_ii//n_rows, img_ii%n_rows].imshow(images[img_ii][:, :, z_ii], cmap="viridis") 
                        plt.colorbar(cmp, ax=axs[img_ii//n_rows, img_ii%n_rows])
                    else:
                        img_ = np.abs(images[img_ii][:, :, z_ii])
                        cmp = axs[img_ii//n_rows, img_ii%n_rows].imshow(img_, vmin=0., vmax=max_val/vmax_scales[img_ii]) 
                        axs[img_ii//n_rows, img_ii%n_rows].text(img_.shape[1]-40, img_.shape[0]-10, \
                                                            'x'+str(int(vmax_scales[img_ii])), color='white', fontsize=12)
                    axs[img_ii//n_rows, img_ii%n_rows].set_title(img_titles[img_ii])
                    # plt.colorbar(cmp, ax=axs[img_ii//2,img_ii%2])
                    axs[img_ii//n_rows, img_ii%n_rows].set_axis_off()
                plt.tight_layout()
                if save_plots:
                    plt.savefig(fig_name+'_WFSS_sl'+str(z_ii)+'.png', format='png', dpi=300)

    return images, signal_xy_et, g


def estimate_weights(y, weights, coord):
    if weights is None and coord is None:
        with sp.get_device(y):
            weights = (sp.rss(y, axes=(0,)) > 0).astype(y.dtype)

    return weights

