"""Matplotlib helpers.

    molecule(ax, xyz, Z)          top view of the atoms coloured by element, smaller the deeper they lie
    image(ax, img, extent)        one image in grey scale
    show(xyz, Z, images)          one molecule with some of its constant-current and constant-height images (simulate() output)
    gallery(data)                 a random sample of a split file (dataset.read_split output)
"""
import numpy as np
import matplotlib.pyplot as plt

COLOURS = {1: '#e0e0e0', 6: '#404040', 7: '#2a78d6', 8: '#d03b3b', 9: '#1baf7a'}
TOPO_RANGE = (-4.5, 1.4)          # display range of the constant-current images, angstrom
LOG_RANGE = (-3.0, 0.5)           # display range of the constant-height log images


def molecule(ax, xyz, Z, window=6.0, title=None):
    """Top view: atoms coloured by element, drawn smaller the deeper they lie below the topmost atom."""
    xyz, Z = np.asarray(xyz), np.asarray(Z).astype(int)
    top = xyz[:, 2].max()
    for j in np.argsort(xyz[:, 2]):
        size = max(6, 110 * np.exp(-(top - xyz[j, 2]) / 1.5)) * (0.5 if Z[j] == 1 else 1)
        ax.scatter(xyz[j, 0], xyz[j, 1], s=size, color=COLOURS.get(Z[j], 'purple'), edgecolor='k', lw=0.4)
    ax.set(aspect='equal', xlim=(-window, window), ylim=(-window, window), xticks=[], yticks=[])
    if title:
        ax.set_title(title, fontsize=8)


def image(ax, img, extent, window=6.0, vrange=TOPO_RANGE, title=None):
    """One image, grey scale, cropped to +-window angstrom."""
    ax.imshow(img, origin='lower', extent=extent, cmap='gray', vmin=vrange[0], vmax=vrange[1])
    ax.set(xlim=(-window, window), ylim=(-window, window), xticks=[], yticks=[])
    if title:
        ax.set_title(title, fontsize=8)


def show(xyz, Z, images, title='', slices=(0, 2, 5, 8, 10), log_slices=(0, 5, 10)):
    """One row: the molecule, the constant-current images `slices`, the constant-height images `log_slices`."""
    half = images['settings']['size'] / 2
    extent = (-half, half, -half, half)
    fig, axes = plt.subplots(1, 1 + len(slices) + len(log_slices), figsize=(2.3 * (1 + len(slices) + len(log_slices)), 2.6))
    molecule(axes[0], xyz, Z, title=title)
    for ax, k in zip(axes[1:1 + len(slices)], slices):
        image(ax, images['topographs'][k], extent, title=f'constant current\nsetpoint {images["setpoints"][k]:.2g} I_ref')
    planes = images['settings']['z_ref'] + 0.1 * (np.arange(11) - 5)
    for ax, k in zip(axes[1 + len(slices):], log_slices):
        image(ax, images['log_current'][k], extent, vrange=LOG_RANGE, title=f'constant height\n{planes[k]:.1f} A, log10 I')
    fig.tight_layout()
    return fig


def gallery(data, n=12, seed=0, slices=(0, 5, 10), title=''):
    """`n` random molecules of a split: one row each with the molecule and the images `slices`."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data['cid']), min(n, len(data['cid'])), replace=False)
    half = data['setup'].get('size', 18.0) / 2
    extent = (-half, half, -half, half)
    cols = 1 + len(slices)
    fig, axes = plt.subplots(len(idx), cols, figsize=(2.2 * cols, 2.2 * len(idx)))
    for r, i in enumerate(idx):
        xyz = data['xyz'][i]
        xyz = xyz[(xyz != 0).any(1)]                                   # drop the padding rows
        molecule(axes[r, 0], xyz[:, :3], xyz[:, 3], title=f'{data["cid"][i]}' if r == 0 else data['cid'][i])
        for c, k in enumerate(slices):
            image(axes[r, 1 + c], data['X'][i, :, :, k], extent, title=data['labels'][k] if r == 0 else None)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
