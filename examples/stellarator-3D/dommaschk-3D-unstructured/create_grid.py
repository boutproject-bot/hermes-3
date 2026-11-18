#!/usr/bin/env python3
"""Unstructured FCI grid for the Dommaschk stellarator example.

Same magnetic field and domain as ``../dommaschk-3D/create_grid.py``: the
region between two traced flux surfaces of a 5-fold symmetric Dommaschk
field, one field period with 8 poloidal planes. Every poloidal plane is
meshed independently with gmsh (triangles); the two flux surfaces are
Dirichlet walls whose nodes are stored in the x guard cells (set them with
the ``dirichlet_node`` boundary condition). See ``zoidberg.unstructured``
for the grid-file contract.

    python create_grid.py                  # mesh size 3 mm
    python create_grid.py --mesh-size 2e-3

Requires zoidberg with ``zoidberg.unstructured`` and gmsh (``pip install
gmsh``). Existing files are kept unless ``-f`` is given.
"""

import argparse
from pathlib import Path

import numpy as np
import zoidberg
from scipy.spatial import KDTree
from zoidberg.fieldtracer import trace_poincare
from zoidberg.unstructured import (
    UnstructuredGrid,
    build_grid_data,
    gmsh_planes,
    write_unstructured,
)

# Field and domain: as in ../dommaschk-3D/create_grid.py
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


def flux_surface_curves(field, ycoords, r_start, n_points=400):
    """Closed (R, Z) curve of the flux surface through (r_start, 0) in each
    plane, from a Poincare trace."""
    rz, _ = trace_poincare(
        field, r_start, 0.0, YPERIOD, y_slices=ycoords, revs=500, nplot=1, nover=20
    )
    curves = []
    for j in range(len(ycoords)):
        line = zoidberg.rzline.line_from_points(
            rz[:, j, 0, 0], rz[:, j, 0, 1], spline_order=1
        ).equallySpaced(n=n_points)
        curves.append(np.column_stack((line.R, line.Z)))
    return curves


def radial_coordinate(pol):
    """0 on the inner wall, 1 on the outer wall: d_in / (d_in + d_out),
    with d the distance to the nearest wall node. Used for the initial
    profile (BOUT++'s index x is not geometric on unstructured grids)."""
    w_in, w_out = pol.walls
    d_in, _ = KDTree(pol.points[w_in]).query(pol.points)
    d_out, _ = KDTree(pol.points[w_out]).query(pol.points)
    return d_in / (d_in + d_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mesh-size", type=float, default=3e-3, help="[m]")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite")
    args = parser.parse_args()

    filename = Path(__file__).resolve().parent / (
        f"dommaschk_unstructured_{NY}_{args.mesh_size:g}.fci.nc"
    )
    if filename.exists() and not args.force:
        print(filename, "exists. Use -f to recreate it.")
        return

    field = zoidberg.field.DommaschkPotentials(C, R_0=R0, B_0=BTOR)
    ycoords = np.linspace(0.0, YPERIOD, NY, endpoint=False)

    print("Tracing flux surfaces")
    inner = flux_surface_curves(field, ycoords, A1)
    outer = flux_surface_curves(field, ycoords, A2)

    print("Meshing the poloidal planes with gmsh")
    planes = gmsh_planes(outer, inner, mesh_size=args.mesh_size)
    print(f"  {planes[0].n_nodes} nodes per plane")

    print("Computing maps, operators and metric")
    grid = UnstructuredGrid(planes, ycoords, YPERIOD, dirichlet_walls=True)
    data, stats = build_grid_data(grid, field)
    print(f"  {stats}")
    # Field lines from evolving nodes that leave the domain get parallel
    # boundary conditions; with flux-surface walls there should be none
    # (the hit counts in stats include the non-evolved wall nodes)
    evolving = slice(grid.MXG, grid.nx - grid.MXG)
    for name in ("forward", "backward"):
        xt = data[f"{name}_xt_prime"][evolving]
        n_hits = int(np.sum((xt < 0) | (xt >= grid.nx)))
        print(f"  {name} wall hits from evolving nodes: {n_hits}")
    data["xs"] = grid.to3d([radial_coordinate(p) for p in planes])

    write_unstructured(str(filename), grid, data)
    print("Created", filename)


if __name__ == "__main__":
    main()
