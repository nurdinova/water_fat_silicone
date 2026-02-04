import numpy as np
import sigpy as sp
import matplotlib.pyplot as plt
import sigpy.mri as mr
from cupyx.scipy.ndimage import shift as ndimage_shift
from hmrGC.dixon_imaging import MultiEcho
from sigpy import backend
from utils_an import Diag_custom, estimate_weights


def _build_sparsity_prox(x_shape, lamda_sparsity, sparsity_domain, device, diag_operator=None):
    """
    Build sparsity proximal operator (proxg) and optional operator (G).

    Arguments
    ---------
    x_shape : tuple
        Image shape expected by the optimization: (ns, nx, ny), 
        where ns is the number of species.
    lamda_sparsity : sequence
        Regularization strengths. Accepts length 1 or 3.
    sparsity_domain : str
        One of: 'mixed', 'wavelet', 'tv', 'identity'.
        'mixed' = identity for water, Wavelet for fat and silicone.
    device : sigpy.Device
        SigPy device to allocate arrays on.
    diag_operator : object or None
        Custom diagonal operator class (e.g., Diag_custom) for mixed regularization.

    Returns
    -------
    proxg : sigpy.prox.Prox or None
        Proximal operator for g(x) or g(Wx).
    G : sigpy.linop.Linop or None
        Analysis operator for TV, otherwise None.
    """
    lamda_sparsity = tuple(lamda_sparsity) if np.ndim(lamda_sparsity) > 0 else (float(lamda_sparsity),)
    if len(lamda_sparsity) == 1:
        lamda_sparsity = (lamda_sparsity[0],) * 3
    if len(lamda_sparsity) != 3:
        raise ValueError(f"lamda_sparsity must have length 1 or 3, got {len(lamda_sparsity)}.")

    if all(lamda_value == 0 for lamda_value in lamda_sparsity):
        return None, None

    ns, nx, ny = x_shape
    xp = device.xp

    if sparsity_domain == 'mixed':
        if diag_operator is None:
            raise ValueError("diag_operator must be provided for sparsity_domain='mixed'.")

        identity_water = sp.linop.Identity((1, nx, ny))
        wavelet_fat = sp.linop.Wavelet((1, nx, ny), axes=[1, 2])
        wavelet_silicone = sp.linop.Wavelet((1, nx, ny), axes=[1, 2])
        wavelet_mixed = diag_operator([identity_water, wavelet_fat, wavelet_silicone], iaxis=0, oaxis=None)

        n_identity = int(np.prod(identity_water.oshape))
        n_wavelet = int(np.prod(wavelet_fat.oshape))
        lamda_vector = xp.concatenate([
            lamda_sparsity[0] * xp.ones((n_identity,)),
            lamda_sparsity[1] * xp.ones((n_wavelet,)),
            lamda_sparsity[2] * xp.ones((n_wavelet,)),
        ])
        proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(wavelet_mixed.oshape, lamda_vector), wavelet_mixed)
        return proxg, None

    if sparsity_domain == 'wavelet':
        wavelet_operator = sp.linop.Wavelet(x_shape, axes=[1, 2])
        proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(wavelet_operator.oshape, lamda_sparsity[0]), wavelet_operator)
        return proxg, None

    if sparsity_domain == 'tv':
        finite_difference = sp.linop.FiniteDifference(x_shape, axes=[1, 2])
        # Thresholds are broadcasted across W/F/S along the first dimension of x
        lamda_vector = xp.stack([
            lamda_sparsity[0] * xp.ones(finite_difference.oshape[1:], dtype=xp.float32),
            lamda_sparsity[1] * xp.ones(finite_difference.oshape[1:], dtype=xp.float32),
            lamda_sparsity[2] * xp.ones(finite_difference.oshape[1:], dtype=xp.float32),
        ], axis=0)
        proxg = sp.prox.L1Reg(finite_difference.oshape, lamda_vector)
        return proxg, finite_difference

    if sparsity_domain == 'identity':
        lamda_vector = xp.stack([
            lamda_sparsity[0] * xp.ones((nx, ny), dtype=xp.float32),
            lamda_sparsity[1] * xp.ones((nx, ny), dtype=xp.float32),
            lamda_sparsity[2] * xp.ones((nx, ny), dtype=xp.float32),
        ], axis=0)
        proxg = sp.prox.L1Reg(x_shape, lamda_vector)
        return proxg, None

    raise ValueError(f"Unknown sparsity_domain: {sparsity_domain}")


class L1WaveletRecon_WFS(sp.app.LinearLeastSquares):
    def __init__(self, ksp_cxyt, sens_cxyt, weights=None, fieldmap_xy_Hz=None, te_ms=None, cse_matrix_ts=None,
                 lamda_sparsity=(0., 0., 0.), show_pbar=True, sparsity_domain='wavelet',
                 chemical_shift_operator=None, diag_operator=Diag_custom, **kwargs):
        """
        Joint reconstruction of water-fat-silicone with Compressed Sensing regularization on proton densities.

        Arguments
        ---------
        ksp_cxyt : array
            K-space data with shape (nc, nx, ny, nt).
        sens_cxyt : array
            Coil sensitivity maps with shape (nc, nx, ny) or (nc, nx, ny, nt).
        weights : array or None
            Optional sampling weights with shape broadcastable to ksp_cxyt.
            If None, will be computed from the k-space data.
        fieldmap_xy_Hz : array or None
            Optional field map (Hz) with shape (nx, ny). If provided, echo times "te_ms" must be provided.
        te_ms : array or None
            Echo times in milliseconds with shape (nt,).
        cse_matrix_ts : array
            Chemical shift encoding matrix with shape broadcastable to (1, 1, nt, ns).
        lamda_sparsity : sequence
            Sparsity penalties. Length 1 or 3. If length 3, interpreted as (water, fat, silicone).
        show_pbar : bool
            Whether to show progress bar in solver.
        sparsity_domain : str
            One of 'mixed', 'wavelet', 'tv', 'identity'.
        chemical_shift_operator : sigpy.linop.Linop or None
            Optional linear operator applied before chemical shift encoding (e.g. additional reshapes).
        diag_operator : callable
            Operator constructor used for mixed regularization. Defaults to Diag_custom.
        **kwargs : dict
            Passed to SigPy LinearLeastSquares.

        Returns
        -------
        None
            This initializes a SigPy app; run() returns reconstructed image coefficients.
        """
        device = sp.get_device(ksp_cxyt)
        xp = device.xp

        if sp.get_device(sens_cxyt) != device:
            raise ValueError("Please move all the data to one device.")

        with device:
            if weights is None:
                weights = estimate_weights(ksp_cxyt)

            ksp_cxyt = sp.to_device(ksp_cxyt * weights**0.5, device=device)
            if ksp_cxyt.ndim != 4:
                raise ValueError(f"Expected ksp_cxyt with 4 dims (nc,nx,ny,nt), got {ksp_cxyt.shape}.")

            if sens_cxyt.ndim == 3:
                sens_cxyt = sens_cxyt[..., None]
            if sens_cxyt.ndim != 4:
                raise ValueError(f"Expected sens_cxyt with 3 or 4 dims, got {sens_cxyt.shape}.")

            nc, nx, ny, nt = ksp_cxyt.shape
            if cse_matrix_ts is None:
                raise ValueError("cse_matrix_ts must be provided.")
            ns = cse_matrix_ts.shape[-1]
            x_shape = (ns, nx, ny)

            # Phasor term from fieldmap (Hz) and te_ms (ms)
            if fieldmap_xy_Hz is None:
                phase_xyt = xp.ones((nx, ny, nt), dtype=ksp_cxyt.dtype)
            else:
                if te_ms is None:
                    raise ValueError("te_ms must be provided when fieldmap_xy_Hz is not None.")
                if sp.get_device(fieldmap_xy_Hz) != device:
                    fieldmap_xy_Hz = sp.to_device(fieldmap_xy_Hz, device=device)
                te_ms = xp.asarray(te_ms)
                cycles_xyt = fieldmap_xy_Hz[:, :, None] * (te_ms[None, None, :] * 1e-3)
                phase_xyt = xp.exp(2j * xp.pi * cycles_xyt)

            # x: (ns,nx,ny) -> (nx,ny,ns) -> (nx,ny,ns,1)
            transpose_x_to_xys = sp.linop.Transpose(x_shape, (1, 2, 0))
            reshape_x_to_xys1 = sp.linop.Reshape((nx, ny, ns, 1), transpose_x_to_xys.oshape)
            if chemical_shift_operator is not None:
                reshape_x_to_xys1 = chemical_shift_operator * reshape_x_to_xys1

            # Chemical shift encoding: (nx,ny,ns,1) -> (nx,ny,nt,1)
            encode_xys1_to_xyt1 = sp.linop.MatMul(reshape_x_to_xys1.oshape, cse_matrix_ts[None, None])

            # Apply off-resonance phase
            apply_phase_xyt1 = sp.linop.Multiply(encode_xys1_to_xyt1.oshape, phase_xyt[..., None])

            # Arrange to match sens multiplication (polarity/coils handling below)
            transpose_to_r_xyt = sp.linop.Transpose(apply_phase_xyt1.oshape, (3, 0, 1, 2))  # (r,nx,ny,nt)

            # Polarity dimension support (r=2) if present
            if transpose_to_r_xyt.oshape[0] == 2:
                sens_cxyt = sens_cxyt.reshape(2, -1, *sens_cxyt.shape[1:])
                ksp_cxyt = ksp_cxyt.reshape(2, -1, *ksp_cxyt.shape[1:])
                if weights is not None:
                    weights = weights.reshape(2, -1, *weights.shape[1:])

                reshape_to_r1xyt = sp.linop.Reshape((2, 1, nx, ny, nt), transpose_to_r_xyt.oshape)
                transpose_to_r_xyt = reshape_to_r1xyt * transpose_to_r_xyt

            multiply_sens_r_xyt = sp.linop.Multiply(transpose_to_r_xyt.oshape, sens_cxyt)

            fft_axes = [2, 3] if chemical_shift_operator is not None else [1, 2]
            fft_operator = sp.linop.FFT(multiply_sens_r_xyt.oshape, axes=fft_axes)

            forward_operator = (
                fft_operator *
                multiply_sens_r_xyt *
                transpose_to_r_xyt *
                apply_phase_xyt1 *
                encode_xys1_to_xyt1 *
                reshape_x_to_xys1 *
                transpose_x_to_xys
            )

            if weights is not None:
                apply_weights = sp.linop.Multiply(fft_operator.oshape, weights**0.5)
                forward_operator = apply_weights * forward_operator

            proxg, G = _build_sparsity_prox(
                x_shape=x_shape,
                lamda_sparsity=lamda_sparsity,
                sparsity_domain=sparsity_domain,
                device=device,
                diag_operator=diag_operator,
            )

            super().__init__(
                forward_operator,
                ksp_cxyt,
                proxg=proxg,
                G=G,
                solver='ADMM',
                show_pbar=show_pbar,
                **kwargs,
            )


def sense_multislice(ksp_cxyzt, coils_cxyz, slices_to_recon=None, max_iter=10, 
                     lamda_sparsity=(0.,), weights=None, subspace_recon=False, 
                     fieldmap_xyz_Hz=None, te_ms=None, cse_matrix_ts=None, 
                     sparsity_domain='wavelet', 
                     chemical_shift_operator=None):
    """
    Run slice-by-slice (and echo-by-echo or jointly) SENSE reconstruction.

    Arguments
    ---------
    ksp_cxyzt : array
        K-space data with shape (nc, nx, ny, nz, nt).
    coils_cxyz : array
        Coil sensitivity maps with shape (nc, nx, ny, nz).
    slices_to_recon : sequence or None
        Slice indices to reconstruct. If None, reconstruct all slices.
    max_iter : int
        Maximum iterations for the recon solver.
    lamda_sparsity : sequence
        Sparsity penalty/penalties. For SenseRecon, only the first entry is used.
    weights : array or None
        Optional sampling weights with shape broadcastable to ksp_cxyzt.
    subspace_recon : bool
        If True, use L1WaveletRecon_WFS; otherwise use SigPy mr.app.SenseRecon.
    fieldmap_xyz_Hz : array or None
        Field map in Hz with shape (nx, ny, nz), used only for subspace_recon.
    te_ms : array or None
        Echo times in ms (nt,) used only for subspace_recon.
    cse_matrix_ts : array or None
        Chemical shift encoding matrix used only for subspace_recon.
    sparsity_domain : str
        Sparsity domain used only for subspace_recon.
    chemical_shift_operator : sigpy.linop.Linop or None
        Chemical shift operator for correcting the bipolar data displacements.

    Returns
    -------
    img_sense_xyzt : array
        Reconstructed images on the same device/array module as ksp_cxyzt.
        If subspace_recon: shape (ns, nx, ny, nz, 1).
        Else: shape (nx, ny, nz, nt).
    """
    device = backend.get_device(ksp_cxyzt)
    xp = device.xp

    nc, nx, ny, nz = ksp_cxyzt.shape[:4]
    if slices_to_recon is None:
        slices_to_recon = range(nz)

    with device:
        if subspace_recon:
            ksp_cxyzt = ksp_cxyzt[..., None]
            if weights is not None:
                weights = weights[..., None]
            ns = cse_matrix_ts.shape[-1]
            img_sense_xyzt = xp.zeros((ns, nx, ny, nz, 1), dtype=ksp_cxyzt.dtype)
            nt = 1
        else:
            nt = ksp_cxyzt.shape[-1]
            img_sense_xyzt = xp.zeros(ksp_cxyzt.shape[1:], dtype=ksp_cxyzt.dtype)

        for slice_index in slices_to_recon:
            for time_index in range(nt):
                weights_slice = weights[:, :, :, slice_index, ..., time_index] if weights is not None else None

                if subspace_recon:
                    fieldmap_xy_Hz = None if fieldmap_xyz_Hz is None else fieldmap_xyz_Hz[:, :, slice_index]
                    recon_app = L1WaveletRecon_WFS(
                        ksp_cxyzt[:, :, :, slice_index, :, time_index],
                        coils_cxyz[:, :, :, slice_index],
                        weights=weights_slice,
                        fieldmap_xy_Hz=fieldmap_xy_Hz,
                        te_ms=te_ms,
                        cse_matrix_ts=cse_matrix_ts,
                        max_iter=max_iter,
                        show_pbar=False,
                        lamda_sparsity=lamda_sparsity,
                        sparsity_domain=sparsity_domain,
                        chemical_shift_operator=chemical_shift_operator,
                    )
                else:
                    recon_app = mr.app.SenseRecon(
                        ksp_cxyzt[:, :, :, slice_index, time_index],
                        coils_cxyz[:, :, :, slice_index],
                        weights=weights_slice,
                        max_iter=max_iter,
                        show_pbar=False,
                        lamda=lamda_sparsity[0],
                        device=device,
                    )

                img_sense_xyzt[..., slice_index, time_index] = xp.squeeze(recon_app.run())

    return img_sense_xyzt


def sort_bipolar_interleaves(data_dt):
    """
    Swap odd readouts between bipolar sets.

    Arguments
    ---------
    data_dt : array
        Data with at least two trailing dimensions, where the second-to-last dimension contains
        interleave sets arranged in pairs.

    Returns
    -------
    data_dt : array
        Modified input array with swapped odd readouts for each interleave pair.
    """
    device = sp.get_device(data_dt)
    xp = device.xp
    n_pair_sets = data_dt.shape[-2] // 2

    with device:
        for pair_index in range(n_pair_sets):
            first_set_index = 2 * pair_index
            second_set_index = 2 * pair_index + 1

            tmp = data_dt[..., first_set_index, 1::2].copy()
            data_dt[..., first_set_index, 1::2] = data_dt[..., second_set_index, 1::2]
            data_dt[..., second_set_index, 1::2] = tmp

    return data_dt


def conj_coil_combination(img_cxyzt, coils_cxyz):
    """
    Conjugate-phase coil combination.

    Arguments
    ---------
    img_cxyzt : array
        Multi-coil image data with shape (nc, nx, ny, nz, nt) or compatible.
    coils_cxyz : array
        Coil sensitivity maps with shape (nc, nx, ny, nz) or compatible.

    Returns
    -------
    img_xyzt : array
        Coil-combined image with coil dimension summed out.
    """
    device = sp.get_device(img_cxyzt)
    xp = device.xp
    with device:
        return xp.sum(xp.conj(coils_cxyz)[..., None] * img_cxyzt, axis=0)


def _build_wfs_params(signal_model, grase_echo_times, voxelsize_mm, number_fat_peaks, range_fm):
    """
    Build parameter dictionary for fieldmapping using hmrGC.

    Arguments
    ---------
    signal_model : str
        'WFS' or 'WF'.
    grase_echo_times : array-like
        Echo times in ms.
    voxelsize_mm : sequence or None
        Voxel size in mm (3,).
    number_fat_peaks : int
        Fat model peaks.
    range_fm : sequence
        Fieldmap ppm range.

    Returns
    -------
    params : dict
        Parameter dictionary for MultiEcho.
    img_keys : list[str]
        Output image keys in the expected order.
    img_titles : list[str]
        Titles for plotting.
    vmax_scales : list[float]
        Per-component scaling for plotting magnitude images.
    """
    params = {'signal_model': signal_model}

    if signal_model == 'WFS':
        img_keys = ["water", "fat", "silicone", "fieldmap"]
        img_titles = ['Water', 'Fat', 'Silicone', 'Field map, Hz', 'R2prime map, Hz']
        vmax_scales = [4, 1, 2]
    elif signal_model == 'WF':
        img_keys = ["water", "fat", "fieldmap", "r2primemap"]
        img_titles = ['Water', 'Fat', 'Field map, Hz', 'R2prime map, Hz']
        vmax_scales = [4, 1]
    else:
        raise ValueError(f"Model {signal_model} is not known.")

    params['TE_s'] = np.asarray(grase_echo_times) * 1e-3
    params['centerFreq_Hz'] = 42577478 * 3.
    params['fieldStrength_T'] = 3.
    params['voxelSize_mm'] = [1, 1, 1] if voxelsize_mm is None else list(voxelsize_mm)

    params['FatModel'] = {}
    if number_fat_peaks == 9:
        params['FatModel']['freqs_ppm'] = np.array([-3.8, -3.4, -3.1, -2.68, -2.46, -1.95, -0.5, 0.49, 0.59])
        params['FatModel']['relAmps'] = np.array([0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006,
                                                  0.01498501, 0.03996004, 0.00999001, 0.05694306])
    elif number_fat_peaks == 1:
        params['FatModel']['freqs_ppm'] = [-3.4]
        params['FatModel']['relAmps'] = [1.]
    else:
        raise KeyError(f"Fat model with {number_fat_peaks} peaks is not defined")

    if signal_model == 'WFS':
        params['siliconePeak_ppm'] = [-4.4]

    params['range_fm_ppm'] = list(range_fm)
    return params, img_keys, img_titles, vmax_scales


def _normalize_and_mask_signal(signal_xyze, te_index, magn_threshold, device, plot_mask):
    """
    Normalize complex multi-echo signal and compute magnitude mask.

    Arguments
    ---------
    signal_xyze : array
        Signal with shape (nx, ny, nz, n_echoes).
    te_index : int
        Echo index used for normalization.
    magn_threshold : float
        Threshold as fraction of max MIP magnitude.
    device : sigpy.Device
        SigPy device.
    plot_mask : bool
        Whether to plot the mask and a representative magnitude image (CPU plot).

    Returns
    -------
    signal_xyze_norm : array
        Normalized signal.
    mask : array
        Boolean mask with shape (nx, ny, nz).
    norm_factor : scalar
        Normalization factor.
    """
    xp = device.xp
    with device:
        norm_factor = xp.max(xp.abs(signal_xyze[..., te_index]))
        signal_xyze_norm = signal_xyze / norm_factor
        mip = xp.sqrt(xp.sum(xp.abs(signal_xyze_norm) ** 2, axis=-1))
        mask = mip > magn_threshold * xp.max(mip)

    if plot_mask:
        mask_cpu = sp.to_device(mask, sp.cpu_device)
        signal_cpu = sp.to_device(signal_xyze[:, :, 0, 0], sp.cpu_device)
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(mask_cpu[:, :, 0])
        axs[1].imshow(np.abs(signal_cpu))

    return signal_xyze_norm, mask, norm_factor


def run_wfs(data_xyzt, grase_echo_times, method="WFS", range_fm=(-2, 2), lamda=1e-2,
            te_ii=None, get_pdff=False, perform_method="single-res", voxelsize_mm=None,
            N_peaks=1, plot=False, save_plots=False, fig_name=''):
    """
    Run fieldmapping and species separation algorithms using the hmrGC library.

    Arguments
    ---------
    data_xyzt : array
        Input complex images with shape (nx, ny, nz, n_echoes) or compatible.
    grase_echo_times : array-like
        Echo times in milliseconds.
    method : str
        Signal model identifier. Must set params['signal_model'] to 'WFS' or 'WF' for this implementation.
    range_fm : sequence
        Fieldmap ppm search range.
    lamda : float
        Regularization parameter passed into MultiEcho options.
    te_ii : int or None
        Echo index used for normalization. If None, uses 0.
    get_pdff : bool
        If True, also return fat fraction image.
    perform_method : str
        Method passed to MultiEcho.perform, e.g. 'single-res', 'multi-res', 'fieldmap'.
    voxelsize_mm : sequence or None
        Voxel size in mm.
    N_peaks : int
        Number of fat peaks in the fat model.
    plot : bool
        If True, plot outputs.
    save_plots : bool
        If True, save plots to disk.
    fig_name : str
        Prefix used for saved plot filenames.

    Returns
    -------
    images : list or None
        List of reconstructed parameter maps in the order defined by img_keys, plus optional PDFF.
        If perform_method == 'fieldmap', returns None.
    signal_xyzt : array
        Original input data (unchanged).
    multi_echo_object : object
        MultiEcho object after fitting.
    """
    device = sp.get_device(data_xyzt)

    if te_ii is None:
        te_ii = 0
        print('Spin echo is not specified. Normalizing images to the first image.')

    magn_threshold = 5e-2

    signal_model = method if method in ('WFS', 'WF') else None
    if signal_model is None:
        raise ValueError(f"method must be 'WFS' or 'WF' for this implementation, got {method}.")

    params, img_keys, img_titles, vmax_scales = _build_wfs_params(
        signal_model=signal_model,
        grase_echo_times=grase_echo_times,
        voxelsize_mm=voxelsize_mm,
        number_fat_peaks=N_peaks,
        range_fm=range_fm,
    )

    nx, ny, nz = data_xyzt.shape[:3]
    signal_xyze = data_xyzt.reshape((nx, ny, nz, -1))
    print("Input Signal shape ", signal_xyze.shape)

    signal_xyze_norm, mask, norm_factor = _normalize_and_mask_signal(
        signal_xyze=signal_xyze,
        te_index=te_ii,
        magn_threshold=magn_threshold,
        device=device,
        plot_mask=plot,
    )

    multi_echo_object = MultiEcho(signal_xyze, mask, params)
    multi_echo_object.options["reg_param"] = lamda
    multi_echo_object.use_gpu = False
    multi_echo_object.r2star_correction = True
    multi_echo_object.perform(perform_method)

    if perform_method == "fieldmap":
        return None, data_xyzt, multi_echo_object

    images = []
    for key in img_keys:
        if key == "fieldmap":
            images.append(multi_echo_object.fieldmap)
        elif key == "r2primemap":
            images.append(multi_echo_object.r2starmap)
        else:
            images.append(multi_echo_object.images[key])

    if get_pdff:
        pdff = multi_echo_object.images['fatFraction_percent']
        bad_mask = (pdff < 0) | (pdff > 125)
        pdff[bad_mask] = 0
        images.append(pdff)

    if plot:
        n_imgs = len(img_keys)
        n_rows = 2
        n_cols = (n_imgs + n_rows - 1) // n_rows
        for slice_index in range(nz):
            fig, axs = plt.subplots(n_cols, n_rows, figsize=(6, 6))
            max_val = 0.8 * np.max(np.abs(np.array(images[:len(vmax_scales)])))
            for img_index, img_key in enumerate(img_keys):
                ax = axs[img_index // n_rows, img_index % n_rows]
                if img_key in ["fieldmap", "r2primemap"]:
                    cmp = ax.imshow(images[img_index][:, :, slice_index], cmap="viridis")
                    plt.colorbar(cmp, ax=ax)
                else:
                    img_mag = np.abs(images[img_index][:, :, slice_index])
                    cmp = ax.imshow(img_mag, vmin=0., vmax=max_val / vmax_scales[img_index])
                    ax.text(img_mag.shape[1] - 40, img_mag.shape[0] - 10,
                            'x' + str(int(vmax_scales[img_index])), color='white', fontsize=12)
                ax.set_title(img_titles[img_index])
                ax.set_axis_off()
            plt.tight_layout()
            if save_plots:
                plt.savefig(fig_name + '_WFSS_sl' + str(slice_index) + '.png', format='png', dpi=300)

    return images, data_xyzt, multi_echo_object


def estimate_coils(ksp_cxyz, n_acs=32, slices_to_recon=None, max_iter=20, espirit_thresh=2e-2, crop_thresh=0.95):
    """
    Estimate coil sensitivity maps using ESPIRiT slice-by-slice (2D).

    Arguments
    ---------
    ksp_cxyz : array
        Calibration k-space data with shape (nc, nx, ny, nz).
    n_acs : int
        ACS width used for calibration.
    slices_to_recon : sequence or None
        Slice indices to run calibration on. If None, runs on all slices.
    max_iter : int
        Maximum iterations for ESPIRiT calibration.
    espirit_thresh : float
        ESPIRiT threshold parameter.
    crop_thresh : float
        ESPIRiT crop parameter.

    Returns
    -------
    coils_cxyz : array
        Coil sensitivity maps with shape (nc, nx, ny, nz).
    """
    device = backend.get_device(ksp_cxyz)

    nc, nx, ny, nz = ksp_cxyz.shape
    if slices_to_recon is None:
        slices_to_recon = range(nz)

    with device:
        xp = device.xp
        coils_cxyz = xp.zeros((nc, nx, ny, nz), dtype=complex)

        for slice_index in slices_to_recon:
            coils_cxyz[..., slice_index] = mr.app.EspiritCalib(
                ksp_cxyz[..., slice_index],
                calib_width=n_acs,
                device=device,
                show_pbar=False,
                max_iter=max_iter,
                thresh=espirit_thresh,
                crop=crop_thresh,
                kernel_width=5,
            ).run()

    return coils_cxyz


class LambdaLinop(sp.linop.Linop):
    """
    Linear operator defined by explicit forward and adjoint callables.

    This is a lightweight wrapper that allows defining a SigPy Linop
    directly from Python functions without writing a full subclass.

    Arguments
    ---------
    ishape : tuple
        Input shape of the linear operator.
    oshape : tuple
        Output shape of the linear operator.
    A : callable
        Forward operation. Should accept an array of shape `ishape`
        and return an array of shape `oshape`.
    AH : callable
        Adjoint operation. Should accept an array of shape `oshape`
        and return an array of shape `ishape`.
    dtype : numpy.dtype, optional
        Data type of the operator. Default is np.complex64.
    """

    def __init__(self, ishape, oshape, A, AH, dtype=np.complex64):
        super().__init__(ishape, oshape, dtype=dtype)
        self._A = A
        self._AH = AH

    def _apply(self, x):
        return self._A(x)

    def _adjoint_linop(self):
        return LambdaLinop(ishape=self.oshape, oshape=self.ishape,
                            A=self._AH, AH=self._A, dtype=self.dtype)


def ShiftLinop(shape, shift, **kwargs):
    """
    Construct a linear operator that applies a spatial shift.

    The forward operator applies `scipy.ndimage.shift` with the given
    shift, while the adjoint applies the corresponding inverse shift
    (i.e., negated shift). This is a practical approximation to the
    true adjoint and is typically sufficient for reconstruction
    algorithms.

    Arguments
    ---------
    shape : tuple
        Input and output shape of the operator.
    shift : float or sequence of float
        Shift applied along each axis. Passed directly to
        `scipy.ndimage.shift`.
    **kwargs : dict
        Additional keyword arguments forwarded to
        `scipy.ndimage.shift`, such as:
            - mode : str
            - order : int
            - prefilter : bool

    Returns
    -------
    shift_operator : sigpy.linop.Linop
        Linear operator implementing the forward and adjoint shift.
    """
    shift = tuple(np.atleast_1d(shift))
    inverse_shift = tuple(-np.asarray(shift))

    def forward(x):
        return ndimage_shift(x, shift=shift, **kwargs)

    def adjoint(y):
        return ndimage_shift(y, shift=inverse_shift, **kwargs)

    dtype = kwargs.get("dtype", np.complex64)

    return LambdaLinop(ishape=shape, oshape=shape, A=forward,
                        AH=adjoint, dtype=dtype)
