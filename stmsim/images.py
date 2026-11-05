"""From the current to the images that are stored.

    Settings                                   scan window, plane stack, reference height, setpoint ladder, model parameters
    current_stack(mol, settings)               -> (n_planes, px, px) current on horizontal planes above the topmost atom
    constant_current(stack, z, setpoints)      -> (n_setpoints, px, px) height at which each column first reaches each setpoint
    simulate(mol, settings)                    -> dict: 'topographs', 'log_current', 'setpoints', 'I_ref', 'z_top', 'settings'

Conventions: image arrays are indexed [y, x] on settings.grid with plane heights are relative to the topmost atom.
The reference current I_ref is the maximum on the plane z_ref and the setpoint slice 0 is the largest current
(tip closest to a molecule). Stored heights are relative to the plane z_ref.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .current import tunnelling_current


@dataclass
class Settings:
    """Everything that defines one simulation; stored as JSON in the dataset files (attribute `setup`)."""
    size: float = 18.0              # edge of the square scan window, angstrom, centred on the origin
    pixels: int = 64
    z_min: float = -1.5             # planes relative to the topmost atom, angstrom
    z_max: float = 5.0
    dz: float = 0.1
    z_ref: float = 3.0              # reference plane: defines I_ref and the zero of the stored heights
    n_setpoints: int = 11
    decade_step: float = 0.25       # spacing of the setpoints in decades of current
    work_function: float = 5.0      # eV, sets the decay of the current with distance
    bias: float = 0.0               # eV relative to the middle of the gap (0 = in-gap imaging)
    eta: float = 0.1                # eV, half-width of the orbital weighting
    log_floor: float = -6.0         # floor of the constant-height log images

    @property
    def grid(self):
        """Pixel centres along x and y."""
        return np.linspace(-self.size / 2, self.size / 2, self.pixels)

    @property
    def z_planes(self):
        """Plane heights above the topmost atom, ascending (66 with the defaults)."""
        return np.round(np.arange(self.z_min, self.z_max + 1e-6, self.dz), 3)

    @property
    def setpoints(self):
        """Setpoints as multiples of I_ref: 10**(decade_step * (n//2 - k)), largest first."""
        return 10.0 ** (-self.decade_step * (np.arange(self.n_setpoints) - self.n_setpoints // 2))

    @property
    def log_planes(self):
        """Planes of the constant-height images: z_ref - 0.5 ... z_ref + 0.5 in steps of 0.1."""
        return np.round(np.arange(self.z_ref - 0.5, self.z_ref + 0.5 + 1e-6, 0.1), 3)


def current_stack(mol, settings=Settings()):
    """Current on the plane stack: (n_planes, pixels, pixels) indexed [z, y, x]; plane k lies settings.z_planes[k]
    above the topmost atom, x and y run over settings.grid."""
    g = settings.grid
    X, Y = np.meshgrid(g, g)                                            # [y, x]
    z_top = mol.xyz[:, 2].max()
    stack = np.empty((len(settings.z_planes), len(g), len(g)))
    for k, dz in enumerate(settings.z_planes):
        points = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_top + dz)])
        stack[k] = tunnelling_current(mol, points, settings.work_function, settings.bias, settings.eta).reshape(X.shape)
    return stack


def constant_current(stack, z, setpoints):
    """Tip height at which each pixel's current first reaches each setpoint when descending from the top plane.

    stack: (n_planes, ny, nx) currents; z: (n_planes,) plane heights, ascending; setpoints: (n_setpoints,) currents.
    Returns (n_setpoints, ny, nx) heights in the units of z. The crossing is interpolated linearly in log10(I)
    between the last plane below and the first plane at or above the setpoint. A pixel that never reaches the
    setpoint gets the lowest plane (floor) while one already above it on the top plane gets the top plane (ceiling).
    """
    # Work from the top plane downwards and in log10(I): the current is close to exponential in z, so a straight
    # line in log I between two neighbouring planes is a sufficiently accurate interpolant.
    logI = np.log10(np.maximum(stack, 1e-300))[::-1]                    # (planes, ny, nx); index 0 is now the TOP plane
    z = np.asarray(z, float)[::-1]                                      # plane heights in the same order (descending)
    rows, cols = np.indices(stack.shape[1:])                            # pixel index grids for the per-pixel lookups below
    out = np.empty((len(setpoints), *stack.shape[1:]))
    for k, I0 in enumerate(setpoints):
        above = logI >= np.log10(I0)                                    # (planes, ny, nx): current at or above the setpoint?
        first = above.argmax(0)                                         # per pixel: index of the first True from the top
        #                                                                 (argmax of booleans = first True; 0 if there is none)
        i1 = np.clip(first, 1, len(z) - 1)                              # the plane at or above the setpoint (clipped so i0 exists)
        i0 = i1 - 1                                                     # the plane just above it, still below the setpoint
        l0, l1 = logI[i0, rows, cols], logI[i1, rows, cols]             # log currents on those two planes, per pixel
        f = np.clip((np.log10(I0) - l0) / np.where(l1 != l0, l1 - l0, 1.0), 0.0, 1.0)   # fraction of the way from plane i0 to i1
        h = z[i0] + f * (z[i1] - z[i0])                                 # interpolated crossing height
        h[first == 0] = z[0]                                            # already at/above the setpoint on the top plane: ceiling
        h[~above.any(0)] = z[-1]                                        # never reaches the setpoint: floor (the lowest plane)
        out[k] = h
    return out


def simulate(mol, settings=Settings()):
    """All images of one molecule whose positions are already in the imaging frame.

    Returns a dict: {'topographs': (n_setpoints, px, px) tip heights relative to the reference plane in angstrom,
    where slice 0 = largest setpoint; 'log_current': (11, px, px) log10(I / I_ref) on settings.log_planes, floored;
    'setpoints': (multiples of I_ref); 'I_ref'; 'z_top'; 'settings' (as a dict). Image arrays are [y, x].
    """
    stack = current_stack(mol, settings)
    z = settings.z_planes
    i_ref = int(np.argmin(np.abs(z - settings.z_ref)))                 # the plane closest to z_ref
    I_ref = float(stack[i_ref].max())                                  # reference current: the brightest pixel of that plane
    topographs = constant_current(stack, z, I_ref * settings.setpoints) - z[i_ref]   # heights relative to the reference plane
    planes = [int(np.argmin(np.abs(z - h))) for h in settings.log_planes]
    log_current = np.maximum(np.log10(np.maximum(stack[planes], 1e-300) / I_ref), settings.log_floor)   # constant height, floored
    return dict(topographs=topographs, log_current=log_current, setpoints=settings.setpoints, I_ref=I_ref,
                z_top=float(mol.xyz[:, 2].max()), settings=asdict(settings))
