"""Draw a random sample of a split file:  python scripts/gallery.py data/qm9_cc/test.h5 [--n 12] [--seed 0] [--out gallery.png]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from stmsim import plot, read_split   # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('h5')
ap.add_argument('--n', type=int, default=12)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='gallery.png')
a = ap.parse_args()
fig = plot.gallery(read_split(a.h5), n=a.n, seed=a.seed, title=os.path.basename(os.path.dirname(a.h5)) + ' / ' + os.path.basename(a.h5))
fig.savefig(a.out, dpi=110)
print('->', a.out)
