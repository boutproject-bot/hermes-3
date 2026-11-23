#!/usr/bin/env python3
"""Unstructured FCI grid for the TCV-X21 divertor example.

Same field and domain as the structured grid of zoidberg's
``examples/tcv.py`` (``../tcv-x21-structured``): the GEQDSK equilibrium of
TCV shot 65402 and the region between the core flux surface through the
outboard midplane at R = 1.09 m and an outer line made of two flux surfaces
and two straight divertor plates, 1/16 of the torus with 2 poloidal planes.
Every poloidal plane is meshed independently with gmsh, graded from a fine
spacing at the outer line to a coarser one inside; both lines are Dirichlet
walls whose nodes are stored in the x guard cells (set them with the
``dirichlet_node`` boundary condition). Field lines that end on the outer
line are sheath boundary points.

    python create_grid.py --tcv-data /path/to/TCV-X21

Requires zoidberg with ``zoidberg.unstructured`` and gmsh (``pip install
gmsh``), the zoidberg repository (for ``examples/tcv.py``) and the TCV-X21
data (https://github.com/SPCData/TCV-X21). Existing files are kept unless
``-f`` is given.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import zoidberg
from zoidberg.unstructured import gmsh_planes, make_unstructured_grid

# Parameters of zoidberg's examples/tcv.py (fac = 0.5)
FAC = 0.5
SYMMETRY = 16
NY = 2
X1, X2 = 0.74, 0.865  # Lower plate, inner and outer radial position [m]
Y_CUTOFF = -0.70  # Height of the lower plate [m]
X_CUTOFF = 0.62  # Radial position of the inner (vertical) plate [m]
CORE = 1.09  # Core flux surface through this R at the outboard midplane [m]


def load_tcv(path):
    """The tcv_x21 class of zoidberg's examples/tcv.py"""
    spec = importlib.util.spec_from_file_location("tcv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.tcv_x21


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tcv-data", required=True, help="path of the TCV-X21 repository"
    )
    parser.add_argument(
        "--zoidberg",
        default=Path(zoidberg.__file__).resolve().parent.parent,
        help="path of the zoidberg repository (default: the source tree of an "
        "editable install)",
    )
    parser.add_argument("--mesh-size", type=float, default=4e-3, help="[m]")
    parser.add_argument(
        "--wall-size", type=float, default=1e-3, help="at the outer line [m]"
    )
    parser.add_argument(
        "--wall-distance",
        type=float,
        default=12e-3,
        help="distance over which the spacing grows to --mesh-size [m]",
    )
    parser.add_argument("-f", "--force", action="store_true", help="overwrite")
    parser.add_argument(
        "--processes",
        type=int,
        default=2,
        help="worker processes for the per-plane operators (identical result)",
    )
    args = parser.parse_args()

    filename = Path(__file__).resolve().parent / (
        f"tcv_x21_unstructured_{NY}_{args.mesh_size:g}_{args.wall_size:g}.fci.nc"
    )
    if filename.exists() and not args.force:
        print(filename, "exists. Use -f to recreate it.")
        return

    print("Tracing the boundary lines")
    tcv_x21 = load_tcv(Path(args.zoidberg) / "examples" / "tcv.py")
    tcv = tcv_x21(tcv_datafolderpath=str(Path(args.tcv_data)) + "/")
    tcv.divertor(
        1,
        nx=int(FAC * 128 + 4),
        ny=NY,
        nz=int(1024 * FAC),
        x1=X1,
        x2=X2,
        y_cutoff=Y_CUTOFF,
        x_cutoff=X_CUTOFF,
        plotting=0,
        tracemultiplier=2,
    )
    tcv.core(CORE)
    outer = np.column_stack((tcv.outer_line.R, tcv.outer_line.Z))
    inner = np.column_stack((tcv.inner_line.R, tcv.inner_line.Z))

    print("Meshing the poloidal planes with gmsh")
    planes = gmsh_planes(
        [outer] * NY,
        [inner] * NY,
        mesh_size=args.mesh_size,
        wall_size=args.wall_size,
        wall_distance=args.wall_distance,
    )
    print(f"  {planes[0].n_nodes} nodes per plane")

    print("Computing maps, operators and metric")
    # map_walls=False and near_wall_hits=True: heat carried along the field
    # lines leaves through the walls only via the sheath, and end points
    # between the last plasma nodes and the wall are sheath points, as
    # beyond the last cell centre on the structured grid
    make_unstructured_grid(
        str(filename),
        planes,
        tcv.magnetic_field,
        ny=NY,
        Ly=2 * np.pi / SYMMETRY,
        dirichlet_walls=True,
        map_walls=False,
        near_wall_hits=True,
        processes=args.processes,
    )
    print("Created", filename)


if __name__ == "__main__":
    main()
