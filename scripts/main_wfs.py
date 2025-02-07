# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# AN: removed sensitive functionalities: need to test
import os, sys
is_jupyter = hasattr(sys, 'ps1') # https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode
import numpy as np
import sigpy as sp
import sigpy.plot as pl
import cfl
import time
import argparse
import matplotlib.pyplot as plt
import json, yaml

device = sp.Device(0)
xp = device.xp
mvc = lambda x : sp.to_device(x, sp.cpu_device)
mvd = lambda x : sp.to_device(x, device)
from numba import cuda
num_devices = 1 # len(cuda.gpus)

from set_rc_params import set_rc_params
plt.rcParams = set_rc_params(plt.rcParams)

DIR_FIGURES = '/home/sdkuser/workspace/src/water_fat_silicon/figures/'
normalize = lambda x: x / np.max(np.abs(x))

start_time = time.time()

# %%
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

if is_jupyter:
    # Use parameters as loaded from config.yaml
    date_tag = config["date_tag"]
    dataset_number = config["dataset_number"]
    import_archive_and_save_cfl = config["import_archive_and_save_cfl"]
    retro_undersample_interleave = config["retro_undersample_interleave"]

    save_dicoms = config["save_dicoms"]
    save_echo_images = config["save_echo_images"]
    save_coils = config["save_coils"]
    do_pad_images_xy = config["do_pad_images_xy"]
    save_figures = config["save_figures"]

    base_path = config["base_path"]
    data_info_filename = config["data_info_filename"]

    R_eff = config["R_eff"]
    do_sense_subspace = config["do_sense_subspace"]

else:
    # allow to overwrite config from cli
    parser = argparse.ArgumentParser(description="Reconstruction for Water-Fat-Silicone separation")
    
    parser.add_argument("--date_tag", type=str, help="6-digit key for the data.json", default=config["date_tag"])
    parser.add_argument("--dataset_number", type=int, help="Dataset number", default=config["dataset_number"])
    parser.add_argument("--save_cfl", action="store_true", help="Import archives and save CFLs", default=config["import_archive_and_save_cfl"])
    parser.add_argument("--retro_undersample", action="store_true", help="Enable retrospective undersampling", default=config["retro_undersample_interleave"])
    parser.add_argument("--base_path", type=str, help="Full path to data", default=config["base_path"])
    parser.add_argument("--data_info_filename", type=str, help="Filename of the JSON file with data info", default=config["data_info_filename"])
    parser.add_argument("--R_eff", type=int, help="Reduction factor per interleave", default=config["R_eff"])
    parser.add_argument("--do_sense_subspace", action="store_true", help="Enable SENSE subspace processing", default=config["do_sense_subspace"])

    parser.add_argument("--save_dicoms", action="store_true", help="Save DICOM images", default=config["save_dicoms"])
    parser.add_argument("--save_echo_images", action="store_true", help="Save echo images", default=config["save_echo_images"])
    parser.add_argument("--save_coils", action="store_true", help="Save coil images", default=config["save_coils"])
    parser.add_argument("--save_figures", action="store_true", help="Save figures as png", default=config["save_figures"])
    parser.add_argument("--pad_echo_images", action="store_true", help="Enable padding for echo images", default=config["do_pad_images_xy"])
    
    args = parser.parse_args()

    date_tag = args.date_tag
    dataset_number = args.dataset_number
    import_archive_and_save_cfl = args.save_cfl
    retro_undersample_interleave = args.retro_undersample

    save_dicoms = args.save_dicoms
    save_echo_images = args.save_echo_images
    save_coils = args.save_coils
    save_figures = args.save_figures
    do_pad_images_xy = args.pad_echo_images

    base_path = args.base_path
    data_info_filename = args.data_info_filename

    R_eff = args.R_eff
    do_sense_subspace = args.do_sense_subspace

eddy_ss = config["eddy_ss"]
echoes_to_use = np.arange(config["echoes_to_use"])

do_sense_multislice = config["do_sense_multislice"]
do_espirit = config["do_espirit"]
espirit_crop = config["espirit_crop"]
espirit_thresh = config["espirit_thresh"]
N_acs = config["N_acs"]

max_iter_calib = config["max_iter_calib"]
max_iter_unalias = config["max_iter_unalias"]
lamda_calib = config["lamda_calib"]
lamda_unalias = config["lamda_unalias"]
ls_method = config["ls_method"]

polarity_index = config["polarity_index"]
do_sort_bipolar_data = config["do_sort_bipolar_data"]

do_hamming_filter = config["do_hamming_filter"]
do_undersample_prospective = config["do_undersample_prospective"]
dataset_index = config["dataset_index"]


extra_save_dir_suffix  = f'{date_tag}_sense_joint_doublechan_filter_uniform_R{R_eff}test' 
png_folder_ismrm = f'wfs_{extra_save_dir_suffix}'
os.makedirs(f'./figures/ismrm25/{png_folder_ismrm}', exist_ok=True)

# if wanna test a subset
z_ii = 0 # test plots
nz_test = 2
z_begin = 60
z_end = z_begin + nz_test

# %%
BASE_PATH = os.path.join('/home/sdkuser/data/', date_tag+'/') if base_path is None else base_path
JSON_FILENAME = os.path.join('./', 'data_info.json') if data_info_filename is None else data_info_filename

with open(JSON_FILENAME, 'r') as json_data:
    metadata = json.load(json_data)[date_tag][dataset_number]

ksp_cxyzdt, sequence_opts, archive_filenames, metadata = load_wfs_data(date_tag, [dataset_number], import_archive_and_save_cfl, \
    metadata, JSON_FILENAME, save_figures, save_coils, save_dicoms, save_echo_images, \
        DIR_FIGURES, retro_undersample_interleave)

if ksp_cxyzdt.ndim == 6:
    nc, nx, ny, nz, nt, ne = ksp_cxyzdt.shape
elif ksp_cxyzdt.ndim == 5:
    nc, nx, ny, nz, nt = ksp_cxyzdt.shape
    ne = 1  
     
N_data_sets = metadata['N_data_sets']    
dataset_size = (nt + N_data_sets - 1) // N_data_sets
map_te_time = [ [[] for tt in range(ne * dataset_size)] for set in range(N_data_sets)]

ksp_cxyzdt = ksp_cxyzdt.reshape((nc, nx, ny, nz, N_data_sets, dataset_size*ne))

for set_ii in range(N_data_sets):
    for subset_ii in range(dataset_size):
        map_te_time[set_ii][subset_ii::dataset_size] = np.arange(subset_ii*ne, (subset_ii+1)*ne)



ksp_cxyzdt = ksp_cxyzdt[..., z_begin:z_end, :, :]
nz = nz_test
# %%

# sort the time-interleaved data
TE_sets_ms = [] # list of size n_se_te
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

# find closes to spin echo
ind_se_dt = np.argwhere(grase_esp_te_ms_sorted == np.min(np.abs(grase_esp_te_ms_sorted))) 
se_t_ii = ind_se_dt[0][1]
se_d_ii = ind_se_dt[0][0]

TE_sets_ms_sorted = np.array(TE_sets_ms).reshape((N_data_sets, dataset_size))

if is_jupyter:
    print('Sorted k-space on device shape ', ksp_cxyzdt_sorted.shape)
    print('Sorted GRE and SE echo times', grase_esp_te_ms_sorted, TE_sets_ms)
    print("Spin echo indices ", ind_se_dt)

# %%
if do_undersample_prospective:
    with device:
        ksp_cxyzdt_sorted = ksp_cxyzdt_sorted[..., 2*(dataset_index):2*(dataset_index+1), :]
        grase_esp_te_ms_sorted = grase_esp_te_ms_sorted[2*(dataset_index):2*(dataset_index+1), :]
    N_data_sets = 2
    nt = 2

# %%
if metadata['whiten'] or metadata['coil_compr']:
    with device:
        ksp_txyzdc = ksp_cxyzdt_sorted.swapaxes(-1, 0)
        del ksp_cxyzdt_sorted

        if metadata['whiten']:
            # copied Noise stats manually from the scanner
            Psi, _ = get_covariance(os.path.dirname(archive_filenames[0]), archive_path_is_directory_instead_of_file=True)
            ksp_wht_txyzdc = xp.zeros_like(ksp_txyzdc)
            for t_ii in range(nt):
                for e_ii in range(ne):
                    ksp_wht_txyzdc[e_ii, ..., t_ii, :]  = whiten(ksp_txyzdc[e_ii, ..., t_ii, :], Psi, mode='cov') 
            ksp_txyzdc = ksp_wht_txyzdc
            del ksp_wht_txyzdc

        if metadata['coil_compr']:
            nc_svd = nc // 2
            ksp_cc_txyzdc = xp.zeros((ne, nx, ny, nz, nt, nc_svd), dtype=np.complex64)
        
            for zz in range(nz):
                cc_mtx = get_cc_matrix(ksp_txyzdc[se_t_ii, :, :, zz, se_d_ii, :] , nc_svd, do_plot=False)
                ksp_cc_txyzdc[..., zz, :, :] = apply_cc_matrix(ksp_txyzdc[..., zz, :, :], cc_mtx)
            ksp_txyzdc = ksp_cc_txyzdc
            del ksp_cc_txyzdc, cc_mtx
            nc = nc_svd

        ksp_cxyzdt_sorted = ksp_txyzdc.swapaxes(-1, 0)
   
    clean_gpu(device, True)

# %%
# get the sampling masks for interleaves in case of bipolar
if metadata['readout'] == 'bipolar':
    with device:
        P = estimate_weights(ksp_cxyzdt_sorted[None, ...], None, None)
        print(f'K-space sampling mask shape {P.shape}')

        if is_jupyter:
            plot_sampling(P, [0])

    if not do_sense_multislice:
        del P

# %%
# combine positive and negative polarities
# exchange every second echo between the inteleaves
with device:
    if do_sort_bipolar_data:
        if metadata['readout']=='bipolar':
            ksp_cxyzdt_sorted = sort_bipolar_interleaves(ksp_cxyzdt_sorted)
            grase_esp_te_ms_sorted = sort_bipolar_interleaves(grase_esp_te_ms_sorted)
            if do_sense_multislice:
                P = sort_bipolar_interleaves(P)

# %%
with device:
    # sort out bipolar or flyback data 
    if metadata['readout'] == 'bipolar':
        # concatenate posit and negat polarity along the c-dimension 
        if polarity_index is not None: 
            ksp_cxyzt_sorted = ksp_cxyzdt_sorted[..., polarity_index, echoes_to_use]
        else:      
            ksp_cxyzt_sorted = xp.concatenate((ksp_cxyzdt_sorted[..., 0, echoes_to_use], ksp_cxyzdt_sorted[..., 1, echoes_to_use]), axis=0)
        
        if do_sense_multislice:
            if polarity_index is not None: 
                weights = P[..., polarity_index, echoes_to_use]  
            else:   
                weights = xp.concatenate((P[..., 0, echoes_to_use], P[..., 1, echoes_to_use]), axis=0)  
        grase_esp_te_ms_sorted = grase_esp_te_ms_sorted[[0], echoes_to_use].reshape((1, -1))
    else:
        weights = None
        ksp_cxyzt_sorted = ksp_cxyzdt_sorted[..., 0, :]

    ne = len(echoes_to_use)
    se_t_ii = ne // 2
    print(f'ksp shape ', ksp_cxyzt_sorted.shape)  
    ksp_se_cxyz_gpu = ksp_cxyzt_sorted[...,  se_t_ii]

clean_gpu(device, True)

# %%
# dB0-estimation from ACS
for N_acs in [32]:
    ky_acs_indices = [slice((nx - N_acs)//2, (nx + N_acs)//2, 1), slice((ny - N_acs)//2, (ny + N_acs)//2, 1)]

    if (ne > 3 and do_sense_subspace):
        with device:
            ksp_calib_pad_cxyzt = xp.zeros_like(ksp_cxyzt_sorted)
            ksp_acs_cxyzt = ksp_cxyzt_sorted[..., ky_acs_indices[0], ky_acs_indices[1], :, :]

            # filter out k-space
            filter_acs_xy = tukey_window(N_acs, 0.5)[:, None] * tukey_window(N_acs, 0.5)[None]
            ksp_calib_pad_cxyzt[:, ky_acs_indices[0], ky_acs_indices[1], ...] = ksp_acs_cxyzt * mvd(filter_acs_xy)[None, :, :, None, None]
            
            walsh_window_xy_shape = (8, 8)
            walsh_window_xy_stride = (8, 8)
            imgs_ifft_xyzt, _ = recon_fullysampled_walsh_method(ksp_calib_pad_cxyzt, se_t_ii, \
                                            walsh_window_xy_shape, walsh_window_xy_stride)
            grase_te_use = grase_esp_te_ms_sorted[0, :]
            
            img_xyzt = imgs_ifft_xyzt.get()
            images, signal_xyzt, g = run_wfs(img_xyzt, grase_te_use, "WFS", range_fm=[-3, 2], perform_method='breast', \
                                            N_peaks=9, lamda=1e-2, get_pdff=False, \
                                            te_ii=se_t_ii, plot=True, save_plots=False, fig_name='', plot_mask=False)
            
            fieldmap_xyz = np.real(g.fieldmap)

            pl.ImagePlot(fieldmap_xyz[::-1, :, 0], x=1, y=0, colormap='rainbow', hide_axes=True)

            fm_xyzt_rads = 1e-3 * mvd(fieldmap_xyz[..., None] * grase_te_use[None, None, None, :]).astype(np.complex64)
            phi_xyzt = xp.exp(2j * xp.pi * fm_xyzt_rads)
            cse_matrix_ts = sp.to_device(g.phi, device)
            cse_matrix_cxyts = cse_matrix_ts[None, None, None]

# %%
if do_sense_subspace:
    del ksp_calib_pad_cxyzt, imgs_ifft_xyzt, filter_acs_xy, ksp_acs_cxyzt
clean_gpu(device, True)

# %%
# recon
if is_jupyter:
    slices_to_plot = [0]
    coils_to_plot = [4]
    slices_to_process = list(np.arange(nz))
    plot = True
else:
    slices_to_process = list(np.arange(nz))
    plot = False 

# %%
if do_hamming_filter:    
    filter_xy = tukey_window(nx, alpha=0.45)[:, None] * tukey_window(ny, alpha=0.45)[None]

# %%
# PI coils or kernels estimation
with device:
    if do_espirit:
        sens_estim_method = SensitivityEstimationType.ESPIRIT
        
        if not do_sense_subspace:
            fm_xyzt_rads = None
            cse_matrix_ts = None

        coils_cxyz = estimate_coils(ksp_cxyzt_sorted[..., se_t_ii], N_acs, sens_estim_method=sens_estim_method, \
                                    slices_to_recon=slices_to_process, max_iter=max_iter_calib, \
                                    crop_thresh=0.7, espirit_thresh=0.05)  # 0.7, 0.05
        print(f'Espirit output shape {coils_cxyz.shape}')
        pl.ImagePlot(coils_cxyz[..., 0], z=0, x=2, y=1)        
        
    if do_hamming_filter:
        filter_xy_gpu = sp.to_device(filter_xy, device)
        ksp_cxyzt_sorted = ksp_cxyzt_sorted * filter_xy_gpu[None, :, :, None, None]  

# %%
# coil-combine images using SENSE
with ksp_cxyzt_sorted.device:
    if do_sense_multislice:
        img_coilcomb_xyz_dt = sense_multislice(ksp_cxyzt_sorted, coils_cxyz, \
                slices_to_process, max_iter=max_iter_unalias, weights=None, \
                lamda_tv=[0, 0, 0], lamda=1e-5, subspace_recon=do_sense_subspace, \
                fieldmap_xyzt_rads=fm_xyzt_rads, cse_matrix_ts=cse_matrix_ts)
        if do_sense_subspace:
            img_coilcomb_xyz_dt = img_coilcomb_xyz_dt.transpose((1, 2, 3, 4, 0))
        print(f'Coil-combined image shape {img_coilcomb_xyz_dt.shape}')

    if metadata['readout'] == 'bipolar':
        N_data_sets = nt = 1

    img_xyzdt = img_coilcomb_xyz_dt.reshape((nx, ny, nz, N_data_sets, -1)) 
    del img_coilcomb_xyz_dt

    pl.ImagePlot(img_xyzdt[:, :, 1, 0, 0],  x=1, y=0,  vmax=0.4*xp.max(xp.abs(img_xyzdt[:, :, 0, 0, 0])), hide_axes=True)



# %%
del ksp_cxyzt_sorted, ksp_se_cxyz_gpu
for i in range(num_devices):
    clean_gpu(sp.Device(i), True)

# %%
with img_xyzdt.device:
    if do_sense_subspace:
        prefix_recon_folder = 'recon_wfs_images_'
        prefix_file = 'img_xyzs_'
    else:
        prefix_recon_folder = 'recon_images_xyzt_'
        prefix_file = 'img_xyzt_'
    if save_echo_images:
        save_dir = os.path.join(metadata['saved_figures'], f'{prefix_recon_folder}{extra_save_dir_suffix}/') # _undersampl
        os.makedirs(save_dir, exist_ok=True)
    
        img_filename = os.path.join(save_dir, f'{prefix_file}{z_begin}_{z_end}')
        print(img_filename)        
        cfl.writecfl(img_filename, xp.squeeze(img_xyzdt[:, :, slices_to_process, ...]).get())

    if save_coils:
        if do_sense_multislice:
            coils_filename = os.path.join(save_dir,  metadata['data_tag'] + '_coils_cxyz')
            cfl.writecfl(coils_filename, coils_cxyz.get())        

# %%
with device:
    if do_pad_images_xy:
        nx = 512
        ny = 512
        img_pad_xyzdt = pad_images_xy(img_xyzdt, (nx, ny, nz))
    else:
        img_pad_xyzdt = img_xyzdt
    img_pad_xyzdt = mvc(img_pad_xyzdt)

    if is_jupyter:
        pl.ImagePlot(img_pad_xyzdt[:, :, 0, 0, :], z=-1, x=1, y=0, mode='m', title='Padded images, xy-plane')
del img_xyzdt


if do_sense_subspace:
        save_png_wfs(img_pad_xyzdt[90:310, :, :, 0, :], brightness=0.7, contrast=2.0, save_folder=f'./figures/ismrm25/{png_folder_ismrm}', \
                filename=f'img_joint_{extra_save_dir_suffix}_zbegin{z_begin}', wfs_scale_factor=[2, 1, 1.5], \
                        title_str=None, show_figure=True)
        img_xyzs = img_pad_xyzdt[..., 0, :]

# %%
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
if not do_sense_subspace:        
        save_dir = os.path.join(metadata['saved_figures'], f'recon_wfs_images_{extra_save_dir_suffix}/') 
        os.makedirs(save_dir, exist_ok=True)
        img_filename = os.path.join(save_dir, f'img_xyzs_z{z_begin}_{z_end}_test')
        cfl.writecfl(img_filename, img_xyzs)
        print(img_filename)

        # nrmse wrt full two-stage
        if do_undersample_prospective:
            save_png_wfs(img_xyzs[90:310, ...], brightness=1.8, contrast=1.5, save_folder=f'./figures/ismrm25/{png_folder_ismrm}', \
                    filename=f'img_ts_{extra_save_dir_suffix}_zbegin{z_begin}', wfs_scale_factor=[4, 1.5, 1.5], \
                            title_str=None, show_figure=True) 


# %%
if not do_undersample_prospective and save_dicoms:
    save_dcm_dir = metadata['saved_dcm_dir'] + f'{extra_save_dir_suffix}_singleres/'
    os.makedirs(save_dcm_dir, exist_ok=True)
    save_wfs_dicoms([img_xyzs[..., 0], img_xyzs[..., 1], img_xyzs[..., 2]], save_dcm_dir, grad_warp=False, archive_name=archive_filenames[0], \
                    correct_chemical_shift=False) 
                

# %%
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
# 
