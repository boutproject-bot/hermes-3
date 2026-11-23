#!/usr/bin/env python3
"""Add the cell numbering for the support-operator conduction methods.

Writes a copy of a structured FCI grid file with the cell numbering and the
parallel-map operators of ``zoidberg.weights.modify_maps``, computed from the
sub-sampled maps of the file. Use it on the grid of zoidberg's
``examples/tcv.py`` after its wall-hit correction:

    python add_cell_numbers.py cor_TCV_X21_68_2_512_16_1.fci.grid.nc \\
        tcv_x21_structured.fci.nc
"""

import shutil
import sys

import netCDF4
import numpy as np
import zoidberg.weights


def main():
    src, dst = sys.argv[1:3]
    shutil.copy(src, dst)
    with netCDF4.Dataset(dst, "r+") as grid:
        maps = {"MXG": int(grid["MXG"][...]) if "MXG" in grid.variables else 2}
        for name in grid.variables:
            if name in ("R", "Z") or name.startswith(("forward_", "backward_")):
                maps[name] = np.array(grid[name][...])
        maps["subcell_weights"] = np.array(grid["subcell_weights"][...])
        known = set(maps)
        maps = zoidberg.weights.modify_maps(maps, partial_boundaries=False)
        shape = grid["forward_xt_prime"].shape
        for name in (k for k in maps if k not in known):
            value = np.asarray(maps[name])
            if value.dtype.kind == "i":
                value = value.astype(np.int32)  # as zoidberg's MapWriter
            if value.shape == shape:
                dims = grid["forward_xt_prime"].dimensions
            else:
                dims = []
                for n, size in enumerate(value.shape):
                    dims.append(f"{name}_dim{n}")
                    grid.createDimension(dims[-1], size)
            grid.createVariable(name, value.dtype, dims)[...] = value
    print("Created", dst)


if __name__ == "__main__":
    main()
