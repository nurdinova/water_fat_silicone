from ipywidgets import interact
import matplotlib.pyplot as plt
import numpy as np

def plot_3d_image(img_xyzt, n_rows, n_columns, mode="m", max_mult=1., title='', xlabels=None, colorbar=True):
    """ Interactive plotting of 3D volumes. """
    nz = img_xyzt.shape[2]
    
    def plot_image_slices(slice_index):
        vmax = max_mult * np.max(np.abs(img_xyzt[:, :, slice_index, :])) if mode == "m" else \
               max_mult * np.max(img_xyzt[:, :, slice_index, :].real) if mode == 'r' else \
               max_mult * np.max(img_xyzt[:, :, slice_index, :].imag) if mode == 'i' else None
        vmin = -0.1 * vmax if mode == "m" else 0.

        fig, axs = plt.subplots(n_rows, n_columns, figsize=(10,6))
        axs = np.atleast_2d(axs) if (n_rows > 1 and n_columns > 1) else np.array([[axs]]).reshape(n_rows, n_columns)

        for se_te_ii in range(n_rows):
            for ge_te_ii in range(n_columns):
                ax_ = axs[se_te_ii, ge_te_ii] if n_rows > 1 and n_columns > 1 else axs[max(se_te_ii, ge_te_ii)]
                img_xy = np.abs(img_xyzt[:, :, slice_index, ge_te_ii + n_columns * se_te_ii]) if mode == "m" else \
                         np.angle(img_xyzt[:, :, slice_index, ge_te_ii + n_columns * se_te_ii]) if mode == "p" else \
                         img_xyzt[:, :, slice_index, ge_te_ii + n_columns * se_te_ii].real if mode == 'r' else \
                         img_xyzt[:, :, slice_index, ge_te_ii + n_columns * se_te_ii].imag
                if mode == "p": vmax, vmin = np.pi, -np.pi
                
                cmp = ax_.imshow(img_xy, vmax=vmax, vmin=vmin)
                ax_.set_xticks([]), ax_.set_yticks([])
                if xlabels: ax_.set_xlabel(f"{xlabels[ge_te_ii + n_columns * se_te_ii]} Hz", fontsize=10)

        fig.subplots_adjust(right=0.8), fig.suptitle(title)
        if colorbar: fig.colorbar(cmp, cax=fig.add_axes([0.85, 0.15, 0.05, 0.7]))
        plt.show()

    interact(plot_image_slices, slice_index=(0, nz - 1))


def plot_sampling(P_cxyzdt, t_indices=[0]):
    """ Plot sampling pattern. """
    nx, labels, colors = P_cxyzdt.shape[1], ['positive', 'negative'], ['blue', 'orange', 'green', 'red', 'purple', 'yellow']
    for t_ii in t_indices:
        fig, ax = plt.subplots()
        for d_ii in range(P_cxyzdt.shape[-2]):
            ky_pos = np.where(P_cxyzdt[0, nx//2, :, 0, d_ii, t_ii])[0].get()
            ax.vlines(ky_pos, ymin=0, ymax=1, label=labels[d_ii], color=colors[d_ii + 2 * t_ii], alpha=0.5)
        ax.legend(), ax.set_xlabel('ky'), ax.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
