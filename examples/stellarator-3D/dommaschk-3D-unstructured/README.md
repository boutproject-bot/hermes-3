3D FCI example on unstructured meshes: conduction and diffusion in a Dommaschk potential
=========================================================================================

Unstructured-mesh version of [`../dommaschk-3D`](../dommaschk-3D): same magnetic
field (5-fold symmetric Dommaschk potential), same domain (between two traced
flux surfaces, one field period, 8 poloidal planes) and same physics (ion and
electron pressure with Braginskii conduction and anomalous diffusion). Every
poloidal plane is a triangle mesh made independently with
[gmsh](https://gmsh.info); the two flux surfaces are Dirichlet walls.

# Grid generation

Requires zoidberg with `zoidberg.unstructured` and gmsh:
```bash
pip install gmsh
python create_grid.py                    # mesh size 3 mm
python create_grid.py --mesh-size 2e-3   # finer
```
The script traces the two flux surfaces, meshes each cross-section with gmsh
(the same number of nodes in every plane, as the grid format stores the nodes
along x), traces the field lines from every node to the neighbouring planes and
computes the interpolation and perpendicular-diffusion matrices. With 3 mm it
gives 6845 nodes per plane (309 and 424 of them on the inner and outer flux
surface), takes about 270 s, needs about 1.5 GB of memory at peak (both
including the `k2_*` matrices, see below) and writes the 524 MB file
`dommaschk_unstructured_8_0.003.fci.nc`. It also writes the radial label
`flux_label` used for the initial profile (see below).

The unknowns sit at the centroids of their dual cells, which gives second-order
perpendicular diffusion on the unrelaxed gmsh mesh. The wall nodes are stored in
the x guard cells (`MXG` = 424 here), so with `NXPE > 1` each processor needs at
least `MXG` nodes; BOUT++ chooses a valid split automatically.

# Differences in the input file

* `[mesh:paralleltransform:xzinterpolation] type = precomputed` (the
  interpolation weights come from the grid file).
* The wall values are set with `dirichlet_node(...)`, which evaluates the value
  at each wall node, instead of `dirichlet(...)`.
* BOUT++'s index coordinate `x` is not geometric on unstructured grids, so the
  initial profile uses the grid variable `flux_label` (0 on the inner, 1 on the
  outer flux surface), read through `[input:grid_variables]`. It is the same
  function of position in both examples (`../dommaschk_label.py`, written by
  both `create_grid.py` scripts), so the two runs start from the same physical
  profile. For an existing grid, `python ../dommaschk_label.py GRIDFILE` adds it.

# Running

```bash
mpirun -np 8 hermes-3 -d .
```
On unstructured grids the parallel heat conduction uses the
flux-reconstructed conservative divergence (`conduction_method =
petscoperatork2`) by default when the grid file contains its matrices
(`k2_*`, written by `create_grid.py` with current zoidberg). It conserves
energy exactly and converges at second order on test meshes whose unknowns
sit at the dual-cell centroids, as on the gmsh meshes, but has no discrete
energy estimate. Grid files without these matrices fall back to the
support-operator method (`petscoperator`): exactly conservative with an
energy estimate, but its solution error converges only at order 0.6-1.4 on
irregular meshes. `conduction_method = original` converges at order 2.8-3.8
but does not conserve energy exactly.

On 4 processes, t = 10^4 (the default `nout = 100`, `timestep = 100`) took
about 7 min with `petscoperatork2`, against about 17 min for the structured
example on 4 processes under similar load.

# Comparing with the structured example

`../compare_profiles.py` bins Pe and Ph+ of both runs by `flux_label` and
compares the change since t = 0 per bin (the structured example must be run
to the same output times, e.g. `nout=100 timestep=100`):

```bash
python ../compare_profiles.py ../dommaschk-3D ../dommaschk-3D/dommaschk_68_8_512.fci.grid.nc \
    . dommaschk_unstructured_8_0.003.fci.nc
```

At t = 10^4 the largest difference per bin (10 bins) was 0.008 with
`petscoperatork2`, 0.012 with `petscoperator` and 0.033 with `original`; the
script fails above 0.015. With `original`, the inner half heats up relative to
the structured run (the method does not conserve energy exactly), so the
conservative methods agree better with the structured profiles in this
example.
