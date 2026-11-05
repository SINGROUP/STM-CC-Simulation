"""Rotate a molecule into the imaging frame: z is the tip axis and the tip approaches from +z.

    heavy_atom_frame(xyz, Z)         center on PCA of the heavy atoms: x = widest, z = thinnest axis
    table_frame(xyz, R, centroid)    the TERS simulator's frame: (xyz - mean(xyz) + centroid) @ R, not centred afterwards
    read_table(path)                 qm9s_rotations.npz -> {QM9 number: (R, centroid)}

The PCA frame's face (+z or -z) and its 180-degree in-plane turn follow the eigenvector signs: reproducible but
arbitrary. The table frame puts the STM image pixel for pixel onto the TERS image of the same molecule.
"""
import numpy as np


def heavy_atom_frame(xyz, Z):
    """Rotated and centred positions of all atoms (molecules with fewer than three heavy atoms use all atoms)."""
    xyz = np.asarray(xyz, float)
    heavy = xyz[np.asarray(Z) > 1]
    if len(heavy) < 3:                                                   # a plane needs three points (H2, H2O ...)
        heavy = xyz
    centre = heavy.mean(0)
    _, axes = np.linalg.eigh(np.cov((heavy - centre).T, bias=True))   # eigenvectors of the covariance, spread ascending
    rotation = axes[:, ::-1].copy()                                  # reorder: x = largest spread, z = smallest
    rotation[:, 0] *= np.sign(np.linalg.det(rotation))               # det +1: a rotation, never a mirror image
    return (xyz - centre) @ rotation                                 # all atoms, hydrogens included, into the new axes


def table_frame(xyz, rotation, centroid):
    """Positions in the frame of one rotation-table row: our copy of the geometry is moved onto the table's
    centroid (it may be stored with another offset) and rotated as the TERS simulator does, `coordinates @ R`."""
    xyz = np.asarray(xyz, float)
    return (xyz - xyz.mean(0) + np.asarray(centroid)) @ np.asarray(rotation)


def read_table(path):
    """qm9s_rotations.npz -> {QM9 number: (rotation (3, 3), centroid (3,))}."""
    with np.load(path) as t:
        return {int(n): (R, c) for n, R, c in zip(t['qm9_index'], t['rotation_matrix'], t['centroid'])}
