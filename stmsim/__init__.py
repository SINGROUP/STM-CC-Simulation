"""stmsim: simulated scanning-tunnelling-microscope (STM) images of small molecules, computed from their orbitals.

Public API, re-exported from the modules in the order the data flows:

    molecule.py   Molecule, QH9, from_xyz, from_dft, from_fock, read_xyz, write_xyz   atoms + orbital coefficients
    orient.py     heavy_atom_frame, table_frame, read_table                           rotate into the imaging frame
    current.py    tunnelling_current, state_weights, decay_constant                   current at tip positions
    images.py     Settings, current_stack, constant_current, simulate                 plane stack -> images
    dataset.py    write_split, read_split                                             HDF5 files of many molecules
    plot.py       molecule, image, show, gallery                                      matplotlib figures

    mol = from_xyz('molecule.xyz')                 # DFT with PySCF, about a minute
    mol.xyz = heavy_atom_frame(mol.xyz, mol.Z)     # z becomes the tip axis
    images = simulate(mol, Settings())             # images['topographs']: (11, 64, 64) heights in angstrom, [y, x]
"""
from .molecule import Molecule, QH9, from_dft, from_fock, from_xyz, read_xyz, write_xyz
from .orient import heavy_atom_frame, table_frame, read_table
from .current import tunnelling_current, state_weights, decay_constant
from .images import Settings, simulate, current_stack, constant_current
from .dataset import write_split, read_split
from . import plot

__all__ = ['Molecule', 'QH9', 'from_dft', 'from_fock', 'from_xyz', 'read_xyz', 'write_xyz', 'heavy_atom_frame', 'table_frame', 'read_table',
           'tunnelling_current', 'state_weights', 'decay_constant', 'Settings', 'simulate', 'current_stack',
           'constant_current', 'write_split', 'read_split', 'plot']
