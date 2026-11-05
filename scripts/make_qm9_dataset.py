"""Build the QM9 data set from QH9Stable.db for the molecules listed in splits/{train,val,test}.txt.

    python scripts/make_qm9_dataset.py --qh9 QH9Stable.db --out data [--workers 32] [--pixels 64] [--limit N]
    python scripts/make_qm9_dataset.py --qh9 QH9Stable.db --out data --name qm9s --rotations qm9s_rotations.npz

Split lines read "QH9 id  QM9 number" (the DiffSpectra split of QM9 restricted to the 125 165 molecules QH9 covers).
Output: data/<name>_cc/<split>.h5 and data/<name>_log/<split>.h5 (layout in stmsim/dataset.py). Frame: the heavy-atom
PCA frame, or with --rotations the TERS simulator's rotation table looked up by QM9 number (stmsim/orient.py).
About 1-2 s per molecule per worker; a rerun resumes where it stopped.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from stmsim import QH9, Settings, heavy_atom_frame, read_table, table_frame, write_split

_DB = None            # one database connection per worker, opened lazily by load()
_PATH = None          # path of QH9Stable.db
_TABLE = None         # {QM9 number: (rotation, centroid)} when --rotations is given


def load(job):
    """(QH9 id, QM9 number) -> Molecule with positions in the image frame. Runs inside the worker processes."""
    global _DB
    if _DB is None:
        _DB = QH9(_PATH)
    qh9_id, qm9 = job
    mol = _DB[qh9_id]
    mol.name, mol.qm9 = str(qh9_id), qm9
    mol.xyz = heavy_atom_frame(mol.xyz, mol.Z) if _TABLE is None else table_frame(mol.xyz, *_TABLE[qm9])
    return mol


def main():
    global _PATH, _TABLE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--qh9', required=True, help='QH9Stable.db')
    ap.add_argument('--out', default='data')
    ap.add_argument('--splits', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'splits'))
    ap.add_argument('--name', default='qm9')
    ap.add_argument('--rotations', default=None, help='qm9s_rotations.npz of the TERS simulator (default: heavy-atom PCA frame)')
    ap.add_argument('--pixels', type=int, default=64, help='image resolution over the 18 A window (cost grows with pixels^2)')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    ap.add_argument('--limit', type=int, default=None, help='molecules per split (test runs)')
    a = ap.parse_args()
    _PATH = a.qh9
    frame = 'heavy-atom PCA frame'
    if a.rotations:
        _TABLE = read_table(a.rotations)
        frame = f'frame of the TERS simulator table {os.path.basename(a.rotations)}: all-atom PCA of the QM9S geometry, coordinates @ R, not centred'
    for split in ('train', 'val', 'test'):
        jobs = [tuple(int(v) for v in line.split()) for line in open(f'{a.splits}/{split}.txt') if line.strip() and not line.startswith('#')]
        write_split(a.out, a.name, split, jobs[:a.limit], load, Settings(pixels=a.pixels), workers=a.workers, frame=frame)
    print('done')


if __name__ == '__main__':
    main()
