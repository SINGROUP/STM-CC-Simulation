"""Tunnelling current of a rigid s-wave tip: PP-STM's model (Krejci et al., Phys. Rev. B 95, 045407, 2017) in numpy.

    decay_constant(work_function)        -> kappa in 1/angstrom
    state_weights(energies, bias, eta)   -> (n_orbitals,) Lorentzian weight of each orbital at the bias
    tunnelling_current(mol, points)      -> (n_points,) current at each tip position, arbitrary units

    I(R)   = sum_n  w_n * G_n(R)**2, where w_n = eta / ((E_n - bias)**2 + eta**2)
    G_n(R) = sum_atoms  exp(-kappa r) * (c_s + sqrt(3) * (c_px dx + c_py dy + c_pz dz) / r)

`mol` needs `.xyz` (n_atoms, 3) in angstrom, `.energies` (n_orbitals,) in eV and `.coefs` (n_orbitals, n_atoms, 4)
in the order s, px, py, pz (molecule.Molecule).
"""
import numpy as np

BOHR = 1.889725989      # 1 angstrom in bohr
HARTREE = 27.211386     # 1 Hartree in eV


def decay_constant(work_function=5.0):
    """kappa in 1/angstrom: sqrt(2 W) in atomic units, converted (1.146 for 5 eV)."""
    return np.sqrt(2.0 * work_function / HARTREE) * BOHR


def state_weights(energies, bias=0.0, eta=0.1):
    """Lorentzian weight of every orbital around the bias voltage (all in eV)."""
    return eta / ((np.asarray(energies) - bias) ** 2 + eta ** 2)


def tunnelling_current(mol, points, work_function=5.0, bias=0.0, eta=0.1, chunk=4096):
    """Current at every tip position. `points`: (n_points, 3) in angstrom. Returns (n_points,).

    Evaluated in chunks of `chunk` points so that the (points, atoms, 3) temporaries stay in the CPU cache to
    reduce the simulation cost. The cost grows linearly with the number of points.
    """
    points = np.asarray(points)
    if len(points) > chunk:
        return np.concatenate([tunnelling_current(mol, points[i:i + chunk], work_function, bias, eta, chunk)
                               for i in range(0, len(points), chunk)])
    kappa = decay_constant(work_function)
    d = points[:, None, :] - mol.xyz[None, :, :]                       # (points, atoms, 3): vector from each atom to the tip
    r = np.sqrt((d * d).sum(-1))                                       # (points, atoms): its length
    radial = np.exp(-kappa * r)                                        # exponential tail of every atomic orbital at the tip
    basis = np.empty((len(d), len(mol.xyz), 4))                        # (points, atoms, 4): s, px, py, pz tails at the tip
    basis[..., 0] = radial
    basis[..., 1:] = (np.sqrt(3.0) * radial / r)[..., None] * d        # p lobes: the tail times the direction cosine (PP-STM's sqrt(3))
    G = basis.reshape(len(d), -1) @ mol.coefs.reshape(len(mol.energies), -1).T   # (points, orbitals): amplitude G_n(R) of each orbital
    return (G ** 2) @ state_weights(mol.energies, bias, eta)           # weighted sum of the squared amplitudes
