"""Molecule: atoms plus the orbital data that current.py needs.

    Molecule(xyz, Z, energies, coefs)   the container consumed by current.py (fields documented in the class)
    from_xyz(path) / from_dft(xyz, Z)   run B3LYP/def2-SVP with PySCF (QH9's settings), then from_fock
    from_fock(F, xyz, Z)                Fock matrix (Hartree, PySCF basis-function order) -> Molecule
    QH9(path)[i]                        one entry of QH9Stable.db (Yu et al., NeurIPS 2023) -> Molecule
    read_xyz / write_xyz                plain .xyz files

Orbital energies are stored in eV with zero at the middle of the HOMO-LUMO gap, and only orbitals within
ENERGY_WINDOW of the frontier are kept.
"""
import sqlite3
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from pyscf import gto
from pyscf import dft

HARTREE = 27.211386                   # 1 Hartree in eV
ORBITALS = ('s', 'px', 'py', 'pz')    # the four parts kept per atom in this order in Molecule.coefs
ENERGY_WINDOW = 5.0                   # eV: orbitals further below the HOMO or above the LUMO are dropped
ELEMENTS = ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']


@dataclass
class Molecule:
    xyz: np.ndarray        # (n_atoms, 3) positions, angstrom
    Z: np.ndarray          # (n_atoms,) atomic numbers
    energies: np.ndarray   # (n_orbitals,) orbital energies in eV, 0 = middle of the HOMO-LUMO gap
    coefs: np.ndarray      # (n_orbitals, n_atoms, 4) weight of the s, px, py, pz part of each orbital on each atom
    name: str = ''

    @property
    def homo(self):
        """Energy of the highest occupied orbital (eV)."""
        return float(self.energies[self.energies < 0].max())

    @property
    def lumo(self):
        """Energy of the lowest unoccupied orbital (eV)."""
        return float(self.energies[self.energies > 0].min())

    def __len__(self):
        return len(self.Z)


def read_xyz(path):
    """Positions (angstrom), atomic numbers and the comment line of an .xyz file."""
    with open(path) as f:
        n = int(f.readline())
        comment = f.readline().strip()
        rows = [f.readline().split() for _ in range(n)]
    Z = np.array([ELEMENTS.index(r[0]) if r[0].isalpha() else int(r[0]) for r in rows])
    xyz = np.array([[float(v) for v in r[1:4]] for r in rows])
    return xyz, Z, comment


def write_xyz(path, xyz, Z, comment=''):
    """Write an .xyz file (element symbols, angstrom)."""
    with open(path, 'w') as f:
        f.write(f'{len(Z)}\n{comment}\n')
        for (x, y, z), el in zip(xyz, Z):
            f.write(f'{ELEMENTS[int(el)]:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n')


def from_xyz(path, **dft_settings):
    """Molecule from an .xyz file: runs the DFT calculation (from_dft)."""
    xyz, Z, comment = read_xyz(path)
    return from_dft(xyz, Z, name=comment or str(path), **dft_settings)


def from_dft(xyz, Z, name='', xc='b3lyp', basis='def2-svp'):
    """Run a closed-shell DFT calculation with PySCF (QH9's settings by default) and convert its Fock matrix."""
    mol = _pyscf_molecule(xyz, Z, basis)
    mf = dft.RKS(mol, xc=xc).run()
    return from_fock(mf.get_fock(), xyz, Z, basis=basis, name=name)


def from_fock(F, xyz, Z, basis='def2-svp', name=''):
    """Orbitals from a Fock matrix: solve F C = S C e, zero the energies at mid-gap, keep the frontier window,
    and collapse the basis functions of each atom into one s and one (px, py, pz) set; d functions are dropped."""
    mol = _pyscf_molecule(xyz, Z, basis)
    S = mol.intor('int1e_ovlp')                                # overlap of the basis functions (they are not orthogonal)
    e, C = eigh(F, S)                                          # generalised eigenproblem: e ascending, columns of C = orbitals
    n_occ = mol.nelectron // 2                                 # closed shell: orbitals 0 .. n_occ-1 are occupied
    e = (e - 0.5 * (e[n_occ - 1] + e[n_occ])) * HARTREE        # eV, zero at the middle of the HOMO-LUMO gap (= bias 0)
    keep = (e > e[n_occ - 1] - ENERGY_WINDOW) & (e < e[n_occ] + ENERGY_WINDOW)   # the frontier window
    coefs = np.zeros((int(keep.sum()), len(Z), 4))
    for i, (atom, _, shell, m) in enumerate(mol.ao_labels(fmt=False)):   # one label per basis function, e.g. (0, 'C', '2p', 'x')
        kind = shell[-1] + m                                              # 's', 'px', 'py', 'pz', 'dxy', ...
        if kind in ORBITALS:                                              # shells of the same kind add up (as PP-STM's reader does)
            coefs[:, atom, ORBITALS.index(kind)] += C[i, keep]
    return Molecule(np.asarray(xyz, float), np.asarray(Z, int), e[keep], coefs, name)


def _pyscf_molecule(xyz, Z, basis):
    """PySCF molecule object for the geometry (angstrom) and basis set."""
    return gto.M(atom=[(int(z), tuple(map(float, p))) for z, p in zip(Z, xyz)], basis=basis, unit='Angstrom', verbose=0)


class QH9:
    """Read-only access to QH9Stable.db (sqlite table `data`: id, N, Z, pos, Ham): `QH9(path)[i]` -> Molecule."""

    def __init__(self, path):
        self.db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)

    def __getitem__(self, i):
        n, Z, xyz, F = self.db.execute('select N, Z, pos, Ham from data where id = ?', (int(i),)).fetchone()
        Z = np.frombuffer(Z, np.int32)
        xyz = np.frombuffer(xyz, np.float64).reshape(n, 3)
        F = np.frombuffer(F, np.float64)                                   # the Fock matrix, flattened square
        return from_fock(F.reshape(int(np.sqrt(F.size)), -1), xyz, Z, name=f'QH9 {i}')
