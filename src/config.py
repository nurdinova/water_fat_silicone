import argparse
import yaml
import numpy as np


def load_parameters(is_jupyter, config_filename="config.yaml"):
    """
    Load reconstruction parameters from YAML and (optionally) CLI.

    YAML is the primary source.
    CLI arguments will override the defaults.

    Arguments
    ---------
    is_jupyter : bool
        Whether the code is running in a Jupyter environment.
    config_filename : str
        Path to YAML configuration file.

    Returns
    -------
    params : dict
        Dictionary containing all reconstruction parameters.
    """
    with open(config_filename, "r") as file:
        config = yaml.safe_load(file)

    params = dict(config)

    if not is_jupyter:
        parser = argparse.ArgumentParser(
            description="Reconstruction for Water-Fat-Silicone separation"
        )

        # Core dataset / path parameters
        parser.add_argument("--date_tag", type=str, default=params["date_tag"])
        parser.add_argument("--dataset_number", type=int, default=params["dataset_number"],
                            help="Dataset index within the specified date.")
        parser.add_argument("--base_path", type=str, default=params["base_path"],
                            help="Base full path.")
        parser.add_argument("--data_info_filename", type=str, default=params["data_info_filename"],
                            help="Name of .json with scan data info.")

        # Reconstruction options
        parser.add_argument("--R_eff", type=int, default=params["R_eff"],
                            help="Acceleration of each data interleave.")
        parser.add_argument("--do_sense_subspace", action="store_true",
                            default=params["do_sense_subspace"],
                            help="Do joint WFS reconstruction.")
        
        # ESPIRiT calibration
        parser.add_argument("--N_acs", type=int, default=params["N_acs"],
                            help="ACS width for ESPIRiT calibration")
        parser.add_argument("--espirit_crop", type=float, default=params["espirit_crop"],
                            help="ESPIRiT crop threshold")
        parser.add_argument("--espirit_thresh", type=float, default=params["espirit_thresh"],
                            help="ESPIRiT eigenvalue threshold")
        parser.add_argument("--max_iter_calib", type=int, default=params["max_iter_calib"],
                            help="Max iterations for ESPIRiT calibration")

        # Unaliasing / reconstruction
        parser.add_argument("--max_iter_unalias", type=int, default=params["max_iter_unalias"],
                            help="Max iterations for unaliasing reconstruction")
        parser.add_argument("--lamda_calib", type=float, default=params["lamda_calib"],
                            help="Calibration regularization")
        parser.add_argument("--lamda_unalias", type=float, default=params["lamda_unalias"],
                            help="Unaliasing regularization")

        # Data handling
        parser.add_argument("--polarity_index", type=int, default=params.get("polarity_index", None),
                            help="Index of polarity dimension (if present)")
        parser.add_argument("--do_sort_bipolar_data", action="store_true",
                            default=params["do_sort_bipolar_data"],
                            help="Sort bipolar interleaves")

        # Filtering
        parser.add_argument("--do_filter_recon", action="store_true",
                            default=params["do_filter_recon"],
                            help="Apply post-reconstruction filtering")

        # Constraints
        parser.add_argument("--lamda_sparsity", type=float, nargs=3,
                            default=params.get("lamda_sparsity", [0., 0., 0.]),
                            help="Sparsity weights [water fat silicone]")
        parser.add_argument("--do_constrain_ngc", action="store_true",
                            default=params.get("do_constrain_ngc", False),
                            help=("Nyquist ghost phase is estimated as phase difference "
                                  "from the first positive-negative polarity coils."))
        parser.add_argument("--sparsity_domain", type=str,
                            default=params.get("sparsity_domain", "wavelet"),
                            help="wavelet | tv | identity | mixed")

        parser.add_argument("--slices_to_process", type=int, nargs="*", default=None,
                            help="Slice indices to process (e.g. --slices_to_process 0 1 2 3)")

        args = parser.parse_args()
        params.update(vars(args))

    if isinstance(params.get("echoes_to_use", None), int):
        params["echoes_to_use"] = np.arange(params["echoes_to_use"])

    params["lamda_sparsity"] = list(params.get("lamda_sparsity", [0., 0., 0.]))
    params["do_constrain_ngc"] = bool(params.get("do_constrain_ngc", False))
    params["sparsity_domain"] = params.get("sparsity_domain", "wavelet")
    params["plot_results"] = bool(params.get("plot_results", False))

    return params
