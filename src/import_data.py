import os
import pickle
import json
import cfl
import numpy as np
from utils import largest_h5_from_dir

def load_wfs_data(date_tag, datasets_to_process, import_archive_and_save_cfl, metadata, json_filename, save_figures, save_coil_img, save_dicoms, save_echo_img, dir_figures,
    retro_undersample_interleave=False):
    base_path = os.path.join('/home/sdkuser/data/', date_tag)
    root_out_dir = '/home/sdkuser/saved_data/'
    
    for dataset_number in datasets_to_process:
        series_tag = "_".join(f"Series_{s}" for s in metadata['Series'])

        if retro_undersample_interleave:
            series_tag += '_undersampl' 

        saved_cfl_dir = os.path.join(root_out_dir, 'saved_cfl_rawdata', date_tag, series_tag)
        saved_recon_dir = os.path.join(saved_cfl_dir, 'recon_images')
        os.makedirs(saved_cfl_dir, exist_ok=True)
        os.makedirs(saved_recon_dir, exist_ok=True)

        dir_figures_dataset = os.path.join(dir_figures, date_tag, series_tag)
        dir_coil_img = os.path.join(dir_figures_dataset, 'Coil_imgs')
        dir_echo_img = os.path.join(dir_figures_dataset, 'Echo_imgs')
        save_dcm_dir = os.path.join(dir_figures_dataset, 'saved_dicoms')
        save_wfs_dir = os.path.join(dir_figures_dataset, 'saved_wfs')
        save_spectralimages_dir = os.path.join(dir_figures_dataset, 'saved_spectralimages')

        for condition, path in [(save_echo_img, dir_echo_img),
            (save_coil_img, dir_coil_img), (save_figures, save_wfs_dir),
            (save_figures, save_spectralimages_dir), (save_dicoms, save_dcm_dir)]:
            if condition:
                os.makedirs(path, exist_ok=True)

        try:
            with open(json_filename, "r") as json_data:
                json_dict = json.load(json_data)
        except (FileNotFoundError, json.JSONDecodeError):
            json_dict = {}

        metadata.update({
            'saved_cfl_dir': saved_cfl_dir,
            'saved_dcm_dir': save_dcm_dir,
            'saved_figures': dir_figures_dataset
        })
        json_dict.setdefault(date_tag, {})[dataset_number] = metadata

        with open(json_filename, "w") as outfile:
            json.dump(json_dict, outfile, indent=4)

        scans_to_load = [f'Series{s}/' for s in metadata['Series']]
        archive_filenames = [largest_h5_from_dir(os.path.join(base_path, scan)) for scan in scans_to_load]

        ksp_filename = os.path.join(saved_cfl_dir, f"{metadata['data_tag']}_ksp_cxyzdt")

        if import_archive_and_save_cfl:
            rampsample = metadata.get('rampsample')
            rampfile = metadata.get('rampsample_file')

            if rampfile:
                if isinstance(rampfile, str):
                    rampfile = os.path.join(base_path, rampfile)
                else:
                    rampfile = [os.path.join(base_path, r) for r in rampfile]

            # Load scan archive data
            ksp_cxyzet_raw, sequence_opts, _ = load_batch_fse_multiecho_scanarchive(archive_filenames,
                fully_sampled=metadata.get('fully_sampled', False), extract_recon_files=True,
                debug=1, rampfile=rampfile, rampsample=rampsample)

            # Adjust shape if fully sampled
            if metadata.get('fully_sampled'):
                ksp_cxyzet_raw = ksp_cxyzet_raw[..., 0]

            # Rearrange k-space dimensions
            ksp_cxyzdt = np.transpose(ksp_cxyzet_raw, [0, 1, 2, 3, 5, 4])

            # Save k-space data
            cfl.writecfl(ksp_filename, ksp_cxyzdt)

            with open(os.path.join(saved_cfl_dir, f"{metadata['data_tag']}_sequence_opts"), 'wb') as f:
                pickle.dump(sequence_opts, f)
        else:
            # Load previously saved k-space data
            ksp_cxyzdt = cfl.readcfl(ksp_filename)
            print(f'Raw k-space data shape: {ksp_cxyzdt.shape}')

            with open(os.path.join(saved_cfl_dir, f"{metadata['data_tag']}_sequence_opts"), 'rb') as f:
                sequence_opts = pickle.load(f)

        return ksp_cxyzdt, sequence_opts, archive_filenames, metadata
