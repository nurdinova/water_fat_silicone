from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp

def normalize_wfs(img_xys):
    device, xp = sp.get_device(img_xys), sp.get_device(img_xys).xp
    with device:
        return (img_xys - xp.min(img_xys, axis=(0, 1, 2))) / \
               (xp.max(img_xys, axis=(0, 1, 2)) - xp.min(img_xys, axis=(0, 1, 2))) * 255

def save_png_wfs(img_xyzt, brightness, contrast, save_folder='', filename='', title_str=None, 
                 wfs_scale_factor=[1, 1, 1], show_figure=False):
    """ Save WFS images as PNG with brightness and contrast adjustments. """
    nx, ny, nz, _ = img_xyzt.shape
    wfs_scale = np.array(wfs_scale_factor)[None, None, :]

    for z_ii in range(nz):
        img_xys = normalize_wfs(np.abs(img_xyzt[:, :, z_ii, :]) * wfs_scale).transpose((0, 2, 1)).reshape((nx, ny * 3)).astype(np.uint8)
        image_contrast = ImageEnhance.Contrast(ImageEnhance.Brightness(Image.fromarray(img_xys).convert('L')).enhance(brightness)).enhance(contrast)

        plt.figure()
        if title_str:
            plt.title(title_str[z_ii])
        plt.imshow(image_contrast)
        plt.axis('off')  
        plt.savefig(f'{save_folder}/{filename}_z{z_ii}.png', dpi=300, bbox_inches='tight')
        if not show_figure:
            plt.close()
