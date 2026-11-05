"""Simulate STM images of molecules given as .xyz files (DFT with PySCF, about a minute per molecule).

    python scripts/simulate_xyz.py examples/molecules/*.xyz --out examples/output

Per molecule: <out>/<name>.npz (positions in the imaging frame, atomic numbers, the constant-current and constant-height
images, setpoints, I_ref, z_top, settings) and <out>/<name>.png; plus <out>/gallery.png with all molecules.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from stmsim import Settings, from_xyz, heavy_atom_frame, plot, simulate   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('xyz', nargs='+')
    ap.add_argument('--out', default='output')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for path in a.xyz:
        name = os.path.splitext(os.path.basename(path))[0]
        mol = from_xyz(path)
        mol.xyz = heavy_atom_frame(mol.xyz, mol.Z)
        images = simulate(mol, Settings())
        np.savez(f'{a.out}/{name}.npz', xyz=mol.xyz, Z=mol.Z, name=mol.name, **{k: v for k, v in images.items() if k != 'settings'},
                 settings=str(images['settings']))
        fig = plot.show(mol.xyz, mol.Z, images, title=f'{name}\n{mol.name}')
        fig.savefig(f'{a.out}/{name}.png', dpi=110)
        rows.append((name, mol, images))
        print(f'{name}: {len(mol)} atoms, {len(mol.energies)} orbitals, gap {mol.lumo - mol.homo:.2f} eV -> {a.out}/{name}.png', flush=True)
    fig, axes = plt.subplots(len(rows), 7, figsize=(2.3 * 7, 2.4 * len(rows)))      # gallery: molecule, 5 setpoints, 1 constant height
    axes = np.atleast_2d(axes)
    for r, (name, mol, images) in enumerate(rows):
        half = images['settings']['size'] / 2
        plot.molecule(axes[r, 0], mol.xyz, mol.Z, title=name)
        for c, k in enumerate((0, 2, 5, 8, 10)):
            plot.image(axes[r, 1 + c], images['topographs'][k], (-half, half, -half, half), title=f'setpoint {images["setpoints"][k]:.2g} I_ref' if r == 0 else None)
        plot.image(axes[r, 6], images['log_current'][5], (-half, half, -half, half), vrange=plot.LOG_RANGE, title='constant height 3 A, log10 I' if r == 0 else None)
    fig.tight_layout()
    fig.savefig(f'{a.out}/gallery.png', dpi=110)
    print(f'-> {a.out}/gallery.png')


if __name__ == '__main__':
    main()
