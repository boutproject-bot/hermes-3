#!/usr/bin/env python3
"""Compare the radial Pe and Ph+ profiles of the structured and unstructured
Dommaschk examples.

Both runs start from the same profile in the common radial label
``flux_label`` (``dommaschk_label.py``). This script bins the evolving cells
of both runs by ``flux_label`` and compares the volume-weighted means per bin
at one output time:

    python compare_profiles.py STRUCTURED_RUN STRUCTURED_GRID \\
        UNSTRUCTURED_RUN UNSTRUCTURED_GRID [--time T] [--bins N] [--tol TOL]

The run directories hold the BOUT.dmp.*.nc files. Volumes: J dx dy dz on the
structured grid, dy * dagp_us_volume on the unstructured grid (wall nodes in
the x guard cells are excluded). The checked quantity is the binned change
since t = 0: the cells of the two grids sit at different labels inside a bin,
so the binned profiles differ already at t = 0 (up to 0.021 in the edge
bins); the change removes this sampling difference. The exit status is 1 if a
bin differs by more than ``--tol`` (default 0.015 in normalised pressure; measured 0.0117 for the
unstructured example with conduction_method = petscoperator) or if both runs
have no common output time.
"""

import argparse
import sys

import netCDF4
import numpy as np
from boutdata.collect import collect

SPECIES = ("Pe", "Ph+")


def structured(run, grid):
    get = lambda n: collect(n, path=run, info=False, xguards=False, yguards=False)
    with netCDF4.Dataset(grid) as g:
        mxg = int(g["MXG"][...])
        label = np.asarray(g["flux_label"][:])[mxg:-mxg]
    dV = get("J") * get("dx") * get("dy") * get("dz")
    fields = {s: get(s) for s in SPECIES}
    t = collect("t_array", path=run, info=False)
    return (
        t,
        label.ravel(),
        dV.ravel(),
        {s: f.reshape(f.shape[0], -1) for s, f in fields.items()},
    )


def unstructured(run, grid):
    with netCDF4.Dataset(grid) as g:
        mxg = int(g["MXG"][...])
        dV = (g["dy"][:] * g["dagp_us_volume"][:])[:, :, 0]
        cell = np.asarray(g["cell_number"][:, :, 0])
        label = np.asarray(g["flux_label"][:])[:, :, 0]
    evolving = np.zeros(dV.shape, bool)
    evolving[mxg:-mxg] = True
    evolving &= cell >= 0
    get = lambda n: collect(n, path=run, info=False, xguards=True, yguards=False)[
        ..., 0
    ]
    fields = {s: get(s)[:, evolving] for s in SPECIES}
    t = collect("t_array", path=run, info=False)
    return t, label[evolving], np.asarray(dV)[evolving], fields


def binned(label, dV, values, edges):
    idx = np.clip(np.digitize(label, edges) - 1, 0, len(edges) - 2)
    vol = np.bincount(idx, weights=dV, minlength=len(edges) - 1)
    return np.bincount(idx, weights=values * dV, minlength=len(edges) - 1) / vol


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("structured_run")
    parser.add_argument("structured_grid")
    parser.add_argument("unstructured_run")
    parser.add_argument("unstructured_grid")
    parser.add_argument(
        "--time", type=float, default=None, help="output time (default: last common)"
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--tol", type=float, default=0.015)
    args = parser.parse_args()

    ts, ls, vs, fs = structured(args.structured_run, args.structured_grid)
    tu, lu, vu, fu = unstructured(args.unstructured_run, args.unstructured_grid)
    common = np.intersect1d(np.round(ts, 6), np.round(tu, 6))
    if common.size == 0:
        print("No common output time")
        return 1
    time = common[-1] if args.time is None else args.time
    i_s = int(np.argmin(np.abs(ts - time)))
    i_u = int(np.argmin(np.abs(tu - time)))
    if abs(ts[i_s] - time) > 1e-6 * max(1.0, time) or abs(tu[i_u] - time) > 1e-6 * max(
        1.0, time
    ):
        print(f"Time {time} is not an output time of both runs")
        return 1

    edges = np.linspace(0.0, 1.0, args.bins + 1)
    print(f"t = {time:g}, {args.bins} bins in flux_label")
    print("Change since t = 0, binned (removes the sampling difference of the")
    print("common initial profile within a bin); profiles themselves for reference")
    header = " ".join(f"{s + ' str':>9} {s + ' uns':>9} {'diff':>8}" for s in SPECIES)
    print(f"{'bin':>11} {header}   (change since t = 0)")
    worst = 0.0
    rows = {}
    for s in SPECIES:
        a = binned(ls, vs, fs[s][i_s] - fs[s][0], edges)
        b = binned(lu, vu, fu[s][i_u] - fu[s][0], edges)
        rows[s] = (a, b)
        worst = max(worst, float(np.max(np.abs(b - a))))
    for k in range(args.bins):
        row = " ".join(
            f"{rows[s][0][k]:+9.5f} {rows[s][1][k]:+9.5f} {rows[s][1][k] - rows[s][0][k]:+8.5f}"
            for s in SPECIES
        )
        print(f"{edges[k]:5.2f}-{edges[k + 1]:4.2f} {row}")
    profile_diff = max(
        float(
            np.max(
                np.abs(
                    binned(lu, vu, fu[s][i_u], edges)
                    - binned(ls, vs, fs[s][i_s], edges)
                )
            )
        )
        for s in SPECIES
    )
    print(f"Largest difference of the binned profiles themselves: {profile_diff:.5f}")
    print(
        f"Largest difference of the binned change since t = 0: {worst:.5f} (tolerance {args.tol})"
    )
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
