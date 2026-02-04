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