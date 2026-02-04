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

# add orchestra and other external lib with syspath hack
PATH = "/home/sdkuser/workspace/src/"
external_libraries_to_add = [PATH+'orchestra-sdk-2.1-1.python/', 
                              PATH+'utils/',
                             PATH+'water_fat_silicone/']

for library in external_libraries_to_add:
    if(is_jupyter):
        sys.path.append(library)
    else:
        sys.path.append(os.path.join(os.path.dirname(__file__), library))

from src import *

import argparse
import cfl, json, yaml

import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import sigpy.plot as pl

import time

device = sp.Device(0)
xp = device.xp
mvc = lambda x : sp.to_device(x, sp.cpu_device)
mvd = lambda x : sp.to_device(x, device)

DIR_FIGURES = '/home/sdkuser/workspace/src/water_fat_silicon/figures/'

normalize = lambda x: x / np.max(np.abs(x))

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

echoes_to_use = params["echoes_to_use"]  # np.arange(...) if YAML gave an int

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
BASE_PATH = os.path.join('/home/sdkuser/data/', date_tag+'/') if base_path is None else base_path
JSON_FILENAME = os.path.join('./', 'data_info.json') if data_info_filename is None else data_info_filename

with open(JSON_FILENAME, 'r') as json_data:
    metadata = json.load(json_data)[date_tag][dataset_number]

ksp_cxyzdt, sequence_opts = load_multiecho_data(BASE_PATH, date_tag, [dataset_number], \
    metadata)

if ksp_cxyzdt.ndim == 6:
    nc, nx, ny, nz, nt, ne = ksp_cxyzdt.shape
elif ksp_cxyzdt.ndim == 5:
    nc, nx, ny, nz, nt = ksp_cxyzdt.shape
    ne = 1  
     
N_data_sets = metadata['N_data_sets']    
dataset_size = (nt + N_data_sets - 1) // N_data_sets
map_te_time = [ [[] for tt in range(ne * dataset_size)] for set in range(N_data_sets)]

for set_ii in range(N_data_sets):
    for subset_ii in range(dataset_size):
        map_te_time[set_ii][subset_ii::dataset_size] = np.arange(subset_ii*ne, (subset_ii+1)*ne)

ksp_cxyzdt = ksp_cxyzdt.reshape((nc, nx, ny, nz, N_data_sets, dataset_size*ne))
# %%
if params["slices_to_process"] is None:
    slices_to_process = list(range(nz))
else:
    slices_to_process = params["slices_to_process"]

# %%
# if wanna test a subset
z_ii = 0 # test plots
nz_test = 2
z_begin = 10
z_end = z_begin + nz_test

ksp_cxyzdt = ksp_cxyzdt[..., z_begin:z_end, :, :]
nz = nz_test

# %%
# extract echo timings and sort the time-interleaved data
TE_sets_ms = [] 
grase_esp_te_ms = []

for vol_set in range(nt):
    TE_sets_ms.append(sequence_opts[vol_set]["te_ms"])
    grase_esp_te_ms.append(sequence_opts[vol_set]["tes_per_echo"] + sequence_opts[vol_set]["dTE_ms"])
grase_esp_te_ms = np.vstack(grase_esp_te_ms).reshape((N_data_sets, dataset_size * ne))

grase_esp_te_ms_sorted = np.zeros_like(grase_esp_te_ms)
ksp_cxyzdt_sorted = np.zeros_like(ksp_cxyzdt)
for set_ii in range(N_data_sets):
    grase_esp_te_ms_sorted[set_ii,:] = grase_esp_te_ms[set_ii, map_te_time[set_ii]]
    ksp_cxyzdt_sorted[...,set_ii,:] = ksp_cxyzdt[..., set_ii, map_te_time[set_ii]]

del ksp_cxyzdt

with device:
    ksp_cxyzdt_sorted = mvd(ksp_cxyzdt_sorted)

# find the echo closest to spin echo
index_se_dt = np.argwhere(grase_esp_te_ms_sorted == np.min(np.abs(grase_esp_te_ms_sorted))) 
se_t_ii = index_se_dt[0][1]
se_d_ii = index_se_dt[0][0]

TE_sets_ms_sorted = np.array(TE_sets_ms).reshape((N_data_sets, dataset_size))

if is_jupyter:
    print('Sorted k-space on device shape ', ksp_cxyzdt_sorted.shape)
    print('Sorted GRE and SE echo times', grase_esp_te_ms_sorted, TE_sets_ms)
    print("Spin echo indices ", index_se_dt)

# %%
# coils compression is calculated on the spin echo data
if metadata['coil_compr']:
    with device:
        ksp_txyzdc = ksp_cxyzdt_sorted.swapaxes(-1, 0)

        if metadata['coil_compr']:
            nc_svd = nc // 2
            ksp_cc_txyzdc = xp.zeros((ne, nx, ny, nz, nt, nc_svd), dtype=np.complex64)
        
            for zz in range(nz):
                cc_mtx = get_cc_matrix(ksp_txyzdc[se_t_ii, :, :, zz, se_d_ii, :] , nc_svd)
                ksp_cc_txyzdc[..., zz, :, :] = apply_cc_matrix(ksp_txyzdc[..., zz, :, :], cc_mtx)
            ksp_txyzdc = ksp_cc_txyzdc
            del ksp_cc_txyzdc, cc_mtx
            nc = nc_svd

        ksp_cxyzdt_sorted = ksp_txyzdc.swapaxes(-1, 0)
   

# %%
# get the sampling masks for interleaves in case of bipolar
if metadata['readout'] == 'bipolar':
    with device:
        P = estimate_weights(ksp_cxyzdt_sorted[None, ...])
        print(f'K-space sampling mask shape {P.shape}')

# %%
# combine positive and negative polarities
# exchange every second echo between the inteleaves
with device:
    if do_sort_bipolar_data and metadata['readout']=='bipolar':
            ksp_cxyzdt_sorted = sort_bipolar_interleaves(ksp_cxyzdt_sorted)
            grase_esp_te_ms_sorted = sort_bipolar_interleaves(grase_esp_te_ms_sorted)
            P = sort_bipolar_interleaves(P)

# %%
with device:
    # stack polarities or reconstruct just one polarity 
    if metadata['readout'] == 'bipolar':
        if polarity_index is not None: 
            ksp_cxyzt_sorted = ksp_cxyzdt_sorted[..., polarity_index, echoes_to_use]
            weights = P[..., polarity_index, echoes_to_use] 
        else:      
            # concatenate posit and negat polarity along the c-dimension 
            ksp_cxyzt_sorted = xp.concatenate((ksp_cxyzdt_sorted[..., 0, echoes_to_use], ksp_cxyzdt_sorted[..., 1, echoes_to_use]), axis=0)
            weights = xp.concatenate((P[..., 0, echoes_to_use], P[..., 1, echoes_to_use]), axis=0)  
         
        grase_esp_te_ms_sorted = grase_esp_te_ms_sorted[[0], echoes_to_use].reshape((1, -1))
    else:
        weights = None
        ksp_cxyzt_sorted = ksp_cxyzdt_sorted[..., 0, :]

    ne = len(echoes_to_use)
    
    # get spin echo indices
    index_se_dt = np.argwhere(grase_esp_te_ms_sorted == np.min(np.abs(grase_esp_te_ms_sorted))) 
    se_t_ii = index_se_dt[0][1]
    se_d_ii = index_se_dt[0][0]
    print(f'ksp shape ', ksp_cxyzt_sorted.shape)  
    
    ksp_se_cxyz_gpu = ksp_cxyzt_sorted[...,  se_t_ii]

# %%
# fieldmap estimation from ACS
ky_acs_indices = [slice((nx - N_acs)//2, (nx + N_acs)//2, 1), 
                  slice((ny - N_acs)//2, (ny + N_acs)//2, 1)]

if (ne > 3 and do_sense_subspace):
    with device:
        ksp_calib_pad_cxyzt = xp.zeros_like(ksp_cxyzt_sorted)
        ksp_acs_cxyzt = ksp_cxyzt_sorted[..., ky_acs_indices[0], ky_acs_indices[1], :, :]

        # filter k-space 
        filter_acs_xy = tukey_window(N_acs, 0.5)[:, None] * tukey_window(N_acs, 0.5)[None]
        ksp_calib_pad_cxyzt[:, ky_acs_indices[0], ky_acs_indices[1], ...] = ksp_acs_cxyzt * mvd(filter_acs_xy)[None, :, :, None, None]
        
        coils_cxyz = estimate_coils(ksp_cxyzt_sorted[..., se_t_ii], N_acs, slices_to_recon=None, max_iter=max_iter_calib, \
                                    crop_thresh=0.7, espirit_thresh=0.05)
        
        if do_constrain_ngc and polarity_index is None:
            # Nyquist ghost phase is estimated only from the 1st coils
            coils_phase_difference_xyz = xp.conj(coils_cxyz[nc]) * coils_cxyz[0]
            coils_phase_difference_xyz = xp.angle(coils_phase_difference_xyz)
            coils_cxyz[nc:] = coils_cxyz[:nc] * xp.exp(1j * coils_phase_difference_xyz[None])

        img_calib_xyzt = sense_multislice(ksp_calib_pad_cxyzt, coils_cxyz, \
            slices_to_recon=None, max_iter=max_iter_unalias, weights=None)
        grase_te_use = grase_esp_te_ms_sorted[0, :]
        
        images, signal_xyzt, g = run_wfs(img_calib_xyzt.get(), grase_te_use, "WFS", range_fm=[-3, 2], perform_method='single-res', \
                                         N_peaks=9, lamda=1e-2, get_pdff=False, \
                                         te_ii=se_t_ii, plot=True, save_plots=False, fig_name='')
        
        fieldmap_xyz = np.real(g.fieldmap)
        
        cse_matrix_ts = mvd(g.phi)
        fieldmap_xyz_gpu = mvd(fieldmap_xyz)
        te_ms_gpu = mvd(grase_te_use)


# %%
del ksp_calib_pad_cxyzt, img_calib_xyzt, filter_acs_xy, ksp_acs_cxyzt

# %%
if do_filter_recon:    
    filter_xy = tukey_window(nx, alpha=0.45)[:, None] * tukey_window(ny, alpha=0.45)[None]

# %%
# coils estimation, filtering k-space before recon
with device:
    if not do_sense_subspace:
        fm_xyzt_rads = None
        cse_matrix_ts = None

        coils_cxyz = estimate_coils(ksp_cxyzt_sorted[..., se_t_ii], N_acs, \
                                    slices_to_recon=slices_to_process, max_iter=max_iter_calib, \
                                    crop_thresh=0.7, espirit_thresh=0.05)  # 0.7, 0.05
        print(f'Espirit output shape {coils_cxyz.shape}')
        pl.ImagePlot(coils_cxyz[..., 0], z=0, x=2, y=1)        
        
    if do_filter_recon:
        filter_xy_gpu = sp.to_device(filter_xy, device)
        ksp_cxyzt_sorted = ksp_cxyzt_sorted * filter_xy_gpu[None, :, :, None, None]  

# %%
from wfs_recon_functions import ShiftLinop

with device:
    if not polarity_index:
        operators = []
        for polarity in [0, 1]:
            chemical_shift_polarity = sp.linop.Diag([
                                                    ShiftLinop((nx, ny, 1, 1), 
                                                               ((-1) ** polarity * 2*polarity * dx, 0., 0., 0.)
                                                               ) for dx in dx_pixel
                                                     ], iaxis=2, oaxis=2)
            operators.append(chemical_shift_polarity)
        chemical_shift_operator = sp.linop.Vstack(operators, axis=3)        
        print(chemical_shift_operator.ishape, chemical_shift_operator.oshape)
    else:
        chemical_shift_operator = None

# %%
# coil-combine images using SENSE
with ksp_cxyzt_sorted.device:
    img_coilcomb_xyz_dt = sense_multislice(ksp_cxyzt_sorted, coils_cxyz, \
            slices_to_process, max_iter=30, weights=weights, \
            lamda_sparsity=lamda_sparsity, subspace_recon=do_sense_subspace, \
            fieldmap_xyz_Hz=fieldmap_xyz_gpu, te_ms=te_ms_gpu, 
            cse_matrix_ts=cse_matrix_ts, sparsity_domain=sparsity_domain, \
            chemical_shift_operator=None)
    if do_sense_subspace:
        img_coilcomb_xyz_dt = img_coilcomb_xyz_dt.transpose((1, 2, 3, 4, 0))
    print(f'Coil-combined image shape {img_coilcomb_xyz_dt.shape}')

    if metadata['readout'] == 'bipolar':
        N_data_sets = nt = 1

    img_xyzdt = img_coilcomb_xyz_dt.reshape((nx, ny, nz, N_data_sets, -1)) 
    del img_coilcomb_xyz_dt

    pl.ImagePlot(img_xyzdt[:, :, 1, 0, :], z=-1,  x=1, y=0,  vmax=0.8*xp.max(xp.abs(img_xyzdt[:, :, 0, 0, 1])), hide_axes=True)



# %%
# two-stage reconstruction of WFS
if not do_sense_subspace:
    diet_ii = 0
    grase_te_use = grase_esp_te_ms[0]
    img_xyzs = np.zeros((nx, ny, nz, 3), dtype=np.complex64)
    fm_xyz = np.zeros_like(img_xyzs[..., 0])

    if nz <= 5:
        z_groupsize = nz
        z_overlap = 0
    else:
        z_groupsize = 5
        z_overlap = 1

    zi_begin = 0
    zi_end = zi_begin + z_groupsize
    while zi_begin < nz:
        img_in = img_pad_xyzdt[:, :, zi_begin:zi_end, diet_ii, :]
        print('WFS separation input shape: ', img_in.shape)

        images, signal_xyzt, g = run_wfs(img_in, grase_te_use, "WFS", range_fm=[-2.5, 2], N_peaks=9, lamda=1e-2, get_pdff=False, \
                                        te_ii=se_t_ii, plot=True, save_plots=False, fig_name='', plot_mask=False)
        for i in range(3):
            img_xyzs[:, :, zi_begin:zi_end, i] = images[i]
        fm_xyz[:, :, zi_begin:zi_end] = np.real(g.fieldmap)
        
        zi_begin = zi_end - z_overlap
        zi_end = zi_begin + z_groupsize
    clear_output(wait=True)

# %%
pl.ImagePlot(img_xyzs[:, :, 1, :], z=-1,  x=1, y=0,  vmax=0.8*xp.max(xp.abs(img_xyzs[:, :, 0, 1])), hide_axes=True)


# %%
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
# 
