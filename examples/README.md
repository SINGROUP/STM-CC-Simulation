# Examples

`molecules/`: six QM9 molecules as `.xyz` files (QM9 numbers 26844, 24855, 10100, 93832, 117814 and
5971, from planar to three-dimensional).

`output/`: what `python scripts/simulate_xyz.py examples/molecules/*.xyz --out examples/output` writes:
`<name>.npz` (positions and atomic numbers in the imaging frame, the 11 constant-current images
`topographs` (11, 64, 64) in ångström relative to the 3 Å reference plane, the constant-height
`log_current` images, the setpoints, I_ref, z_top and the settings), `<name>.png`, and `gallery.png`
with all six molecules.
