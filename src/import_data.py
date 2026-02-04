import os
import cfl, pickle

def load_multiecho_data(base_path, date_tag, datasets_to_process, metadata):
    
    for dataset_number in datasets_to_process:
        
        series = metadata['Series']
        series_tag = f'Series_{series[0]}'   
        for i in range(1, len(series)):
            series_tag += f'_{series[i]}'

        saved_cfl_dir = os.path.join(base_path, 'saved_cfl_rawdata', date_tag, series_tag)

        # Load previously saved k-space data
        ksp_filename = os.path.join(saved_cfl_dir, f"{metadata['data_tag']}_ksp_cxyzdt")
        ksp_cxyzdt = cfl.readcfl(ksp_filename)
        print(f'Raw k-space data shape: {ksp_cxyzdt.shape}')

        with open(os.path.join(saved_cfl_dir, f"{metadata['data_tag']}_sequence_opts"), 'rb') as f:
            sequence_options = pickle.load(f)

        return ksp_cxyzdt, sequence_options
    