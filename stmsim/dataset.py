"""Write and read the HDF5 split files of simulated images (the layout the training code reads).

    write_split(out_dir, name, split, jobs, load, settings, workers)   simulate `jobs` in parallel into two files
    read_split(path, rows=None) -> dict                                arrays, labels and settings of one file

Per split: <out_dir>/<name>_cc/<split>.h5 (constant current) and <out_dir>/<name>_log/<split>.h5 (constant height),
same rows:
    X        (n, px, px, n_slices) float16   cc: tip height minus the reference height, angstrom; log: log10(I / I_ref)
    xyz      (n, max_atoms, 5) float32       x, y, z, atomic number, 0; all-zero rows are padding
    cid (n,) str; qm9 (n,) int; n_states, i_ref, z_top (n,)
    attrs    setup (Settings as JSON), setpoints, heights, frame, slice_order, value, n_written, mean, std, transform

`load(job) -> Molecule` runs inside the worker processes and must return positions already in the imaging frame.
Rows are written in blocks of BLOCK molecules; a rerun resumes at attrs['n_written']. mean/std of the train images
are stored in every file for the reader. Field meanings: README "What is stored per molecule".
"""
import json
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np

from .images import Settings, simulate

N_NORM = 512          # train images the mean / std are computed from
BLOCK = 256           # molecules per write
_LOAD = _SETTINGS = None   # per-worker globals, set by _init (a Pool initializer cannot return state any other way)


def _init(load, settings):
    """Pool initializer: keep the loader and the settings in the worker."""
    global _LOAD, _SETTINGS
    _LOAD, _SETTINGS = load, settings


def _one(job):
    """Simulate one job in a worker and return its row of every dataset as plain arrays."""
    mol = _LOAD(job)                                   # positions already in the image frame
    img = simulate(mol, _SETTINGS)
    xyz = np.zeros((_MAX_ATOMS, 5), np.float32)        # padded to a fixed number of rows
    xyz[:len(mol.Z), :3], xyz[:len(mol.Z), 3] = mol.xyz, mol.Z
    return dict(cc=img['topographs'].transpose(1, 2, 0).astype(np.float32),      # (slices, y, x) -> stored (y, x, slices)
                log=img['log_current'].transpose(1, 2, 0).astype(np.float32),
                xyz=xyz, cid=str(mol.name), qm9=int(getattr(mol, 'qm9', 0)), n_states=len(mol.energies),
                i_ref=img['I_ref'], z_top=img['z_top'])


_MAX_ATOMS = 29       # rows of the xyz array (QM9 molecules have at most 29 atoms); write_split may change it


def _open(path, n, kind, settings, frame):
    """Open a split file for appending; create its datasets and attributes if it is new (kind: 'cc' or 'log')."""
    f = h5py.File(path, 'a')
    if 'X' not in f:
        s = settings
        f.create_dataset('X', (n, s.pixels, s.pixels, s.n_setpoints if kind == 'cc' else len(s.log_planes)), 'f2')
        f.create_dataset('xyz', (n, _MAX_ATOMS, 5), 'f4')
        f.create_dataset('cid', (n,), h5py.string_dtype('ascii'))
        for k, dt in (('qm9', 'i4'), ('n_states', 'i4'), ('i_ref', 'f8'), ('z_top', 'f4')):
            f.create_dataset(k, (n,), dt)
        f.attrs.update(setup=json.dumps(s.__dict__), setpoints=s.setpoints, n_written=0,
                       heights=np.round(-np.log10(s.setpoints), 3) if kind == 'cc' else s.log_planes,
                       frame=f'{frame}; arrays [y, x]; x, y = linspace(-{s.size / 2}, {s.size / 2}, {s.pixels})',
                       slice_order='slice 0 = largest setpoint (tip closest); heights attr = nominal offset from the reference plane, angstrom' if kind == 'cc'
                       else 'slice k = plane heights[k] above the topmost atom',
                       value='tip height at the setpoint minus the reference height, angstrom (clipped to the scanned range)' if kind == 'cc'
                       else 'log10(current / its maximum on the reference plane), floored')
    return f


def write_split(out_dir, name, split, jobs, load, settings=Settings(), workers=4, max_atoms=29, frame='heavy-atom PCA frame'):
    """Simulate `jobs` in `workers` processes into <out_dir>/<name>_cc/<split>.h5 and <out_dir>/<name>_log/<split>.h5.

    `jobs` is whatever `load(job) -> Molecule` understands (QH9 ids, .xyz paths ...); `load` must return positions in
    the imaging frame, and `frame` describes that frame in words (stored as an attribute). A rerun resumes at
    attrs['n_written']. After the split, the train file's mean/std are stored in this file for the reader.
    """
    global _MAX_ATOMS
    _MAX_ATOMS = max_atoms
    files = {}
    for kind in ('cc', 'log'):
        os.makedirs(f'{out_dir}/{name}_{kind}', exist_ok=True)
        files[kind] = _open(f'{out_dir}/{name}_{kind}/{split}.h5', len(jobs), kind, settings, frame)
    start = int(files['cc'].attrs['n_written'])                   # resume point (0 for a new file)
    print(f'{split}: {len(jobs)} molecules, resuming at {start}', flush=True)
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(load, settings)) as pool:
        for b0 in range(start, len(jobs), BLOCK):                 # one block of molecules per write, so a crash loses at most a block
            out = pool.map(_one, jobs[b0:b0 + BLOCK])
            b1 = b0 + len(out)
            for kind, f in files.items():
                f['X'][b0:b1] = np.stack([o[kind] for o in out])
                f['xyz'][b0:b1] = np.stack([o['xyz'] for o in out])
                f['cid'][b0:b1] = [o['cid'] for o in out]
                for key in ('qm9', 'n_states', 'i_ref', 'z_top'):
                    f[key][b0:b1] = [o[key] for o in out]
                f.attrs['n_written'] = b1                         # the resume point; updated only after the rows are written
                f.flush()
            print(f'{split} {b1}/{len(jobs)}  {(time.time() - t0) / (b1 - start):.3f} s/molecule', flush=True)
    for kind, f in files.items():                                 # standardisation constants: always from the train split
        train_path = f'{out_dir}/{name}_{kind}/train.h5'
        train = f if split == 'train' else (h5py.File(train_path, 'r+') if os.path.exists(train_path) else None)
        if train is not None:
            if 'mean' not in train.attrs:                         # computed once, from N_NORM evenly spaced train images
                idx = np.linspace(0, len(train['X']) - 1, min(N_NORM, len(train['X']))).astype(int)
                x = np.stack([train['X'][i] for i in idx]).astype(np.float32)
                train.attrs.update(mean=float(x.mean()), std=float(x.std()))
            f.attrs.update(mean=float(train.attrs['mean']), std=float(train.attrs['std']),
                           transform='(x - mean) / std, applied by the reader; x stored unscaled')
            if train is not f:
                train.close()
        f.close()


def read_split(path, rows=None):
    """Arrays of one split file as a dict (all written rows, or the given row indices); 'labels' names the slices."""
    with h5py.File(path, 'r') as f:
        n = int(f.attrs['n_written'])
        rows = np.arange(n) if rows is None else np.asarray(rows)
        a = f.attrs
        qm9 = f['qm9'][rows] if 'qm9' in f else np.zeros(len(rows), int)   # files written before the column existed
        if str(a.get('value', '')).startswith('log10'):                      # constant-height stack
            labels = [f'height {h:.1f} Å' for h in a['heights']]
        elif 'setpoints' in a:
            labels = [f'setpoint {s:.2g} I_ref' for s in a['setpoints']]
        else:
            labels = [f'slice {k}' for k in range(f['X'].shape[-1])]
        return dict(X=f['X'][rows], xyz=f['xyz'][rows], cid=f['cid'][rows].astype(str), qm9=qm9, z_top=f['z_top'][rows],
                    setpoints=a.get('setpoints'), heights=a.get('heights'), labels=labels, setup=json.loads(a['setup']))
