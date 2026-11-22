#!/usr/bin/env python3
"""Common radial label for the Dommaschk examples (structured and unstructured).

The label is s = d_in / (d_in + d_out), with d_in and d_out the distances of a
point (R, Z) in poloidal plane j to the inner and outer traced flux surfaces
of that plane (0 on the inner, 1 on the outer surface). It is a property of
the position only, so the structured (``../dommaschk-3D``) and unstructured
(``../dommaschk-3D-unstructured``) runs can start from the same physical
initial profile, e.g. ``function = 1.0 - 0.9 * gridvar:flux_label``.

    python dommaschk_label.py GRIDFILE [GRIDFILE ...]

adds (or overwrites) the variable ``flux_label`` in each grid file, which must
have the 8 planes of one field period of these examples. The unstructured
``create_grid.py`` also writes it.
"""

import argparse

import netCDF4
import numpy as np
import zoidberg
from scipy.spatial import KDTree
from zoidberg.fieldtracer import trace_poincare

# Field and domain of both examples (see their create_grid.py)
C = np.zeros((6, 5, 4))
C[5, 2, 1] = -1.489
C[5, 2, 2] = -1.489
BTOR = 1.5  # Magnetic field strength [T]
R0 = 2.0  # Major radius [m]
A1 = R0 + 0.08  # Inner flux surface at the outboard midplane [m]
A2 = R0 + 0.1  # Outer flux surface at the outboard midplane [m]
SYMMETRY = 5
NY = 5 * 8 // SYMMETRY
YPERIOD = 2.0 * np.pi / SYMMETRY

# Points on each traced surface used for the distance (spacing ~0.1 mm)
N_CURVE = 4000


def field():
    return zoidberg.field.DommaschkPotentials(C, R_0=R0, B_0=BTOR)


def flux_surface_lines(fld, ycoords, r_start):
    """Closed line of the flux surface through (r_start, 0) in each plane,
    from a Poincare trace (as in ../dommaschk-3D/create_grid.py)."""
    rz, _ = trace_poincare(
        fld, r_start, 0.0, YPERIOD, y_slices=ycoords, revs=500, nplot=1, nover=20
    )
    return [
        zoidberg.rzline.line_from_points(rz[:, j, 0, 0], rz[:, j, 0, 1], spline_order=1)
        for j in range(len(ycoords))
    ]


def flux_surface_curves(fld, ycoords, r_start, n_points, lines=None):
    """Closed (R, Z) curve of the flux surface through (r_start, 0) in each
    plane, with n_points equally spaced points. ``lines``: the result of
    flux_surface_lines, if already computed."""
    if lines is None:
        lines = flux_surface_lines(fld, ycoords, r_start)
    curves = []
    for line in lines:
        line = line.equallySpaced(n=n_points)
        curves.append(np.column_stack((line.R, line.Z)))
    return curves


def distance_to_curve(points, curve):
    """Distance from each point to the closed polyline ``curve``: the nearest
    vertex and its two adjacent segments."""
    n = len(curve)
    _, k = KDTree(curve).query(points)
    best = np.full(len(points), np.inf)
    for a, b in ((k - 1) % n, k), (k, (k + 1) % n):
        p, q = curve[a], curve[b]
        d = q - p
        t = np.clip(np.sum((points - p) * d, axis=1) / np.sum(d * d, axis=1), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(points - (p + t[:, None] * d), axis=1))
    return best


class FluxLabel:
    """s(R, Z, plane) for the NY planes of one field period."""

    def __init__(self, n_points=N_CURVE, lines=(None, None)):
        """``lines``: flux_surface_lines of the inner and outer flux surface,
        if already computed."""
        ycoords = np.linspace(0.0, YPERIOD, NY, endpoint=False)
        fld = field()
        self.inner = flux_surface_curves(fld, ycoords, A1, n_points, lines[0])
        self.outer = flux_surface_curves(fld, ycoords, A2, n_points, lines[1])

    def __call__(self, R, Z, plane):
        points = np.column_stack((np.ravel(R), np.ravel(Z)))
        d_in = distance_to_curve(points, self.inner[plane])
        d_out = distance_to_curve(points, self.outer[plane])
        return (d_in / (d_in + d_out)).reshape(np.shape(R))


def add_to_grid(filename, label):
    with netCDF4.Dataset(filename, "a") as grid:
        R, Z = grid["R"][:], grid["Z"][:]
        if R.shape[1] != NY:
            raise ValueError(f"{filename}: expected {NY} planes, found {R.shape[1]}")
        s = np.stack([label(R[:, j, :], Z[:, j, :], j) for j in range(NY)], axis=1)
        if "flux_label" not in grid.variables:
            grid.createVariable("flux_label", "f8", grid["R"].dimensions)
        grid["flux_label"][:] = s
        evolving = s[int(grid["MXG"][...]) : R.shape[0] - int(grid["MXG"][...])]
        print(f"{filename}: flux_label in [{evolving.min():.4f}, {evolving.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grids", nargs="+", help="grid files to add flux_label to")
    args = parser.parse_args()
    label = FluxLabel()
    for filename in args.grids:
        add_to_grid(filename, label)


if __name__ == "__main__":
    main()
