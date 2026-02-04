# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os, sys
is_jupyter = hasattr(sys, 'ps1') 

from src import *

import json
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import sigpy.plot as pl
import time

device = sp.Device(0)
xp = device.xp

# small helpers
mvc = lambda x : sp.to_device(x, sp.cpu_device)
mvd = lambda x : sp.to_device(x, device)


# %%
start_time = time.time()

# %%
params = load_parameters(is_jupyter=is_jupyter, config_filename="config.yaml")

date_tag = params["date_tag"]
dataset_number = params["dataset_number"]
base_path = params["base_path"]
data_info_filename = params["data_info_filename"]

R_eff = params["R_eff"]
do_sense_subspace = params["do_sense_subspace"]
do_filter_recon = params["do_filter_recon"]

echoes_to_use = params["echoes_to_use"]  

N_acs = params["N_acs"]
espirit_crop = params["espirit_crop"]
espirit_thresh = params["espirit_thresh"]
max_iter_calib = params["max_iter_calib"]

max_iter_unalias = params["max_iter_unalias"]
lamda_calib = params["lamda_calib"]
lamda_unalias = params["lamda_unalias"]

polarity_index = params.get("polarity_index", None)
do_sort_bipolar_data = params["do_sort_bipolar_data"]

lamda_sparsity = params["lamda_sparsity"]
do_constrain_ngc = params["do_constrain_ngc"]
sparsity_domain = params["sparsity_domain"]

plot_results = params["plot_results"]


# %%
BASE_PATH = base_path
JSON_FILENAME = data_info_filename

with open(JSON_FILENAME, 'r') as json_data:
    metadata = json.load(json_data)[date_tag][dataset_number]

# data dims: (coils, nx, ny, nz, interleaves, echoes)
ksp_cxyzdt, sequence_opts = load_multiecho_data(BASE_PATH, date_tag, metadata)

if ksp_cxyzdt.ndim == 6:
    nc, nx, ny, nz, nt, ne = ksp_cxyzdt.shape
elif ksp_cxyzdt.ndim == 5:
    nc, nx, ny, nz, nt = ksp_cxyzdt.shape
    ne = 1  
     
N_data_sets = metadata['N_data_sets']    
dataset_size = (nt + N_data_sets - 1) // N_data_sets

ksp_cxyzdt = ksp_cxyzdt.reshape((nc, nx, ny, nz, N_data_sets, dataset_size*ne))
# %%
if params["slices_to_process"] is None:
    slices_to_process = list(range(nz))
else:
    start, end = params["slices_to_process"]
    slices_to_process = slice(start, end)
    
ksp_cxyzdt = ksp_cxyzdt[:, :, :, slices_to_process, ...]
nz = ksp_cxyzdt.shape[3]
print('k-space dims: ', ksp_cxyzdt.shape)

# %%
# extract echo timings and sort the time-interleaved data
TE_sets_ms = [] 
grase_esp_te_ms = []

for vol_set in range(nt):
    TE_sets_ms.append(sequence_opts[vol_set]["te_ms"])
    grase_esp_te_ms.append(sequence_opts[vol_set]["tes_per_echo"] + sequence_opts[vol_set]["dTE_ms"])
grase_esp_te_ms = np.vstack(grase_esp_te_ms).reshape((N_data_sets, dataset_size * ne))

# move data to device
with device:
    ksp_cxyzdt_dev = mvd(ksp_cxyzdt)

# find the echo closest to spin echo
index_se_dt = np.argwhere(grase_esp_te_ms == np.min(np.abs(grase_esp_te_ms))) 
se_t_ii = index_se_dt[0][1]
se_d_ii = index_se_dt[0][0]

TE_sets_ms_sorted = np.array(TE_sets_ms).reshape((N_data_sets, dataset_size))

print('Sorted GRE and SE echo times', grase_esp_te_ms, TE_sets_ms)
print("Spin echo indices ", index_se_dt)

# %%
# coils compression is calculated on the spin echo data - apllied to all echoes
if metadata['coil_compr']:
    with device:
        ksp_txyzdc = ksp_cxyzdt_dev.swapaxes(-1, 0)

        if metadata['coil_compr']:
            nc_svd = nc // 2
            ksp_cc_txyzdc = xp.zeros((ne, nx, ny, nz, nt, nc_svd), dtype=np.complex64)
        
            for zz in range(nz):
                cc_mtx = get_cc_matrix(ksp_txyzdc[se_t_ii, :, :, zz, se_d_ii, :] , nc_svd)
                ksp_cc_txyzdc[..., zz, :, :] = apply_cc_matrix(ksp_txyzdc[..., zz, :, :], cc_mtx)
            ksp_txyzdc = ksp_cc_txyzdc
            del ksp_cc_txyzdc, cc_mtx
            nc = nc_svd

        ksp_cxyzdt_dev = ksp_txyzdc.swapaxes(-1, 0)
   

# %%
# get the sampling masks for interleaves in case of bipolar
# maintain the coil dimension
if metadata['readout'] == 'bipolar':
    with device:
        P = estimate_weights(ksp_cxyzdt_dev[None, ...])
        print(f'K-space sampling mask shape {P.shape}')

# %%
# gather positive and negative polarities as interleaves
# exchange every second echo between the inteleaves
with device:
    if do_sort_bipolar_data and metadata['readout']=='bipolar':
            ksp_cxyzdt_dev = sort_bipolar_interleaves(ksp_cxyzdt_dev)
            grase_esp_te_ms = sort_bipolar_interleaves(grase_esp_te_ms)
            P = sort_bipolar_interleaves(P)

# %%
with device:
    # stack polarities or reconstruct just one polarity 
    if metadata['readout'] == 'bipolar':
        if polarity_index is not None: 
            ksp_cxyzt_sorted = ksp_cxyzdt_dev[..., polarity_index, echoes_to_use]
            weights = P[..., polarity_index, echoes_to_use] 
        else:      
            # concatenate posit and negat polarity along the c-dimension 
            ksp_cxyzt_sorted = xp.concatenate((ksp_cxyzdt_dev[..., 0, echoes_to_use], ksp_cxyzdt_dev[..., 1, echoes_to_use]), axis=0)
            weights = xp.concatenate((P[..., 0, echoes_to_use], P[..., 1, echoes_to_use]), axis=0)  
         
        grase_esp_te_ms = grase_esp_te_ms[[0], echoes_to_use].reshape((1, -1))
    else:
        weights = None
        ksp_cxyzt_sorted = ksp_cxyzdt_dev[..., 0, :]

    ne = len(echoes_to_use)
    
    # get spin echo indices
    index_se_dt = np.argwhere(grase_esp_te_ms == np.min(np.abs(grase_esp_te_ms))) 
    se_t_ii = index_se_dt[0][1]
    se_d_ii = index_se_dt[0][0]
    print(f'ksp shape after stacking polarities', ksp_cxyzt_sorted.shape)  

# %%
# fieldmap estimation from ACS
ky_acs_indices = [slice((nx - N_acs)//2, (nx + N_acs)//2, 1), 
                  slice((ny - N_acs)//2, (ny + N_acs)//2, 1)]

if (ne > 3 and do_sense_subspace):
    with device:
        ksp_calib_pad_cxyzt = xp.zeros_like(ksp_cxyzt_sorted)
        ksp_acs_cxyzt = ksp_cxyzt_sorted[..., ky_acs_indices[0], ky_acs_indices[1], :, :]

        # filter k-space - reduce low-res image ringing
        filter_acs_xy = tukey_window(N_acs, 0.5)[:, None] * tukey_window(N_acs, 0.5)[None]
        ksp_calib_pad_cxyzt[:, ky_acs_indices[0], ky_acs_indices[1], ...] = ksp_acs_cxyzt * mvd(filter_acs_xy)[None, :, :, None, None]
        
        coils_cxyz = estimate_coils(ksp_cxyzt_sorted[..., se_t_ii], N_acs, slices_to_recon=None, max_iter=max_iter_calib, \
                                    crop_thresh=espirit_crop, espirit_thresh=espirit_thresh)
        
        if do_constrain_ngc and polarity_index is None:
            # Nyquist ghost phase is estimated only from the 1st coils
            # apllied to the rest of coils
            coils_phase_difference_xyz = xp.conj(coils_cxyz[nc]) * coils_cxyz[0]
            coils_phase_difference_xyz = xp.angle(coils_phase_difference_xyz)
            coils_cxyz[nc:] = coils_cxyz[:nc] * xp.exp(1j * coils_phase_difference_xyz[None])

        img_calib_xyzt = sense_multislice(ksp_calib_pad_cxyzt, coils_cxyz, \
                                          slices_to_recon=None, max_iter=max_iter_unalias, weights=None)
        grase_te_use = grase_esp_te_ms[0, :]
        
        images, signal_xyzt, g = run_wfs(img_calib_xyzt.get(), grase_te_use, "WFS", range_fm=[-3, 2], perform_method='single-res', \
                                         N_peaks=9, lamda=1e-2, get_pdff=False, \
                                         te_ii=se_t_ii, plot=True, save_plots=False, fig_name='')
        
        fieldmap_xyz = np.real(g.fieldmap)
        
        # prep matrices for the joint recon
        cse_matrix_ts = mvd(g.phi)
        fieldmap_xyz_gpu = mvd(fieldmap_xyz)
        te_ms_gpu = mvd(grase_te_use)

    del ksp_calib_pad_cxyzt, img_calib_xyzt, filter_acs_xy, ksp_acs_cxyzt
    
else:
    cse_matrix_ts = None
    fieldmap_xyz_gpu = None
    te_ms_gpu = None

# %%
# coils estimation, filtering k-space before recon
with device:
    if not do_sense_subspace:
        fm_xyzt_rads = None
        cse_matrix_ts = None

        coils_cxyz = estimate_coils(ksp_cxyzt_sorted[..., se_t_ii], N_acs, \
                                    max_iter=max_iter_calib, \
                                    crop_thresh=espirit_crop, espirit_thresh=espirit_thresh)  
        print(f'Espirit output shape {coils_cxyz.shape}')
        pl.ImagePlot(coils_cxyz[..., 0], z=0, x=2, y=1, title='Estimated coil sensitivities')        
     
    if do_filter_recon:
        filter_xy = tukey_window(nx, alpha=0.45)[:, None] * tukey_window(ny, alpha=0.45)[None]   
        filter_xy_gpu = sp.to_device(filter_xy, device)
        ksp_cxyzt_sorted = ksp_cxyzt_sorted * filter_xy_gpu[None, :, :, None, None]  

# %%
# chemical shift-related displacement correction for bipolar data

fov_cm      = 36  # acquisiiton FOV
rbw_kHz     = 250 # readout BW 

pixel_size_cm = abs(fov_cm) / nx

# Chemical-shift frequencies (kHz) for WFS
df_wfs_kHz  = np.array([0.0, 0.434, 0.562])

# Spatial shift (cm) and (pixels)
dx_cm       = fov_cm / (2 * rbw_kHz) * df_wfs_kHz
dx_pixel    = dx_cm / pixel_size_cm
print('Chemical shifts for WFS in pixels: ', dx_pixel)

with device:
    if not polarity_index and metadata['readout']=='bipolar':
        operators = []
        for polarity in (0, 1):
            chemical_shift_polarity = sp.linop.Diag(
                [ShiftLinop((nx, ny, 1, 1),
                            (((-1)**polarity) * dx_shift, 0., 0., 0.))
                 for dx_shift in dx_pixel],
                iaxis=2, oaxis=2
            )
            operators.append(chemical_shift_polarity)

        chemical_shift_operator = sp.linop.Vstack(operators, axis=3)
        print('Chemical shift operator dims: ', chemical_shift_operator.ishape, chemical_shift_operator.oshape)
    else:
        chemical_shift_operator = None


# %%
# get coil-combined images using SENSE
# with do_sense_subspace, does joint recon of water-fat-silicone directly
with ksp_cxyzt_sorted.device:
    img_coilcomb_xyz_dt = sense_multislice(ksp_cxyzt_sorted, coils_cxyz, \
            max_iter=max_iter_unalias, weights=weights, \
            lamda_sparsity=lamda_sparsity, subspace_recon=do_sense_subspace, \
            fieldmap_xyz_Hz=fieldmap_xyz_gpu, te_ms=te_ms_gpu, 
            cse_matrix_ts=cse_matrix_ts, sparsity_domain=sparsity_domain, \
            chemical_shift_operator=chemical_shift_operator)
    
    if do_sense_subspace:
        img_coilcomb_xyz_dt = img_coilcomb_xyz_dt.transpose((1, 2, 3, 4, 0))
        
    print(f'Coil-combined image shape {img_coilcomb_xyz_dt.shape}')

    if metadata['readout'] == 'bipolar':
        N_data_sets = nt = 1

    img_xyzdt = img_coilcomb_xyz_dt.reshape((nx, ny, nz, N_data_sets, -1)) 
    del img_coilcomb_xyz_dt

    pl.ImagePlot(img_xyzdt[:, :, 1, 0, :], z=-1,  x=1, y=0,  vmax=0.8*xp.max(xp.abs(img_xyzdt[:, :, 0, 0, 1])), hide_axes=True)

# %%
fig = pl.ImagePlot(img_xyzdt[:, :, 1, 0, :], z=-1,  x=1, y=0,  vmax=0.8*xp.max(xp.abs(img_xyzdt[:, :, 0, 0, 1])), hide_axes=True).fig
fig.savefig("recon_joint.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# %%
# two-stage reconstruction of WFS 
if not do_sense_subspace:
    interleave_index = 0
    grase_te_use = grase_esp_te_ms[0]
    img_xyzs = np.zeros((nx, ny, nz, 3), dtype=np.complex64)

    img_in = img_xyzdt[..., interleave_index, :].get()
    print('WFS separation input shape: ', img_in.shape)

    images, signal_xyzt, g = run_wfs(img_in, grase_te_use, "WFS", range_fm=[-2.5, 2], N_peaks=9, 
                                     lamda=1e-2, get_pdff=False, te_ii=se_t_ii, plot=True)
    for i in range(3):
        img_xyzs[..., i] = images[i]
    fm_xyz = np.real(g.fieldmap)
    
    pl.ImagePlot(img_xyzs[:, :, 1, :], z=-1,  x=1, y=0,  vmax=0.8*np.max(np.abs(img_xyzs[:, :, 0, 1])), hide_axes=True)

# %%
recon_time_mins = (time.time() - start_time) / 60
print(f"Total recon time--- {recon_time_mins:.2f} minutes ---")
