import sigpy as sp
import numpy as np

def SMS(ishape, coils_cxyz, z_positions_by_gap=None, sms_factor=1, do_caipi=False, weights=None, caipi_factor=2):
    """ SMS forward model """
    if do_caipi and z_positions_by_gap is None:
        raise Exception("CAIPI needs slice z-positions to phase-correct non-isocenter slices!")

    nx, ny, nz, nt, nc, fovz_aliased = *ishape, coils_cxyz.shape[0], nz // sms_factor
    transform_image_cxytsz = sp.linop.Compose([
        sp.linop.Reshape((1, nx, ny, nt, sms_factor, fovz_aliased), sp.linop.Transpose(ishape, (0,1,3,2)).oshape),
        sp.linop.Transpose(ishape, (0,1,3,2))
    ])

    coils_cxy_sz = sp.linop.Reshape((nc, nx, ny, 1, sms_factor, fovz_aliased), coils_cxyz.shape) * coils_cxyz
    FS = sp.linop.FFT(transform_image_cxytsz.oshape, axes=(1,2)) * sp.linop.Multiply(transform_image_cxytsz.oshape, coils_cxy_sz)

    if do_caipi:
        FS = CAIPI_ksp(FS.oshape, z_positions_by_gap, sms_factor, caipi_factor) * FS

    SMS_img = sp.linop.Transpose(FS.oshape, (0,1,2,4,3)) * sp.linop.Sum(FS.oshape, axes=[-2]) * FS * transform_image_cxytsz

    if weights is not None:
        with sp.get_device(weights):
            SMS_img = sp.linop.Multiply(SMS_img.oshape, weights**0.5) * SMS_img

    return SMS_img


def CAIPI_ksp(ishape, z_positions_by_gap, sms_factor=2, caipi_factor=2):
    """ CAIPI linear operator to create 2D-phase modulation maps and multiply slice by slice """
    nx_, ny_, nz_ = ishape[1], ishape[2], len(z_positions_by_gap)
    shift_phase_masks = [
        np.exp(2j * np.pi * (z_positions_by_gap + sms_ii) * 
               (np.arange(-((ny_+1)//2), (ny_+1)//2) % caipi_factor)[:, np.newaxis] / caipi_factor)[np.newaxis,:,:]
        * np.ones((nx_, ny_, nz_))[np.newaxis, :, :, np.newaxis, np.newaxis, :]
        for sms_ii in range(sms_factor)
    ]
    return sp.linop.Multiply(ishape, np.concatenate(shift_phase_masks, axis=-2))


class SMSUnaliasing(sp.app.LinearLeastSquares):
    def __init__(self, y, mps, lamda=0., sms_factor=1, z_positions_by_gap=None, do_caipi=False,
                 caipi_factor=2, device=sp.cpu_device, max_iter=10, weights=None, coord=None, 
                 show_pbar=True, **kwargs):
        
        weights = sp.mri.app._estimate_weights(y, weights, coord)[np.newaxis] if weights is None else weights
        nc, nx, ny, nz, nt = y.shape
        ishape = (nx, ny, sms_factor * nz, nt)

        y, mps = sp.to_device(y, device=device), sp.to_device(mps, device=device)
        A = SMS(ishape, mps, z_positions_by_gap, sms_factor, weights, do_caipi, caipi_factor)

        super().__init__(A, y, lamda=lamda, show_pbar=show_pbar, max_iter=max_iter, **kwargs)
