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
surface), takes about 2.5 minutes (1.5 minutes with `--processes 4`), needs
about 1.4 GB of memory (2 GB at 2.5 mm) and writes
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
On unstructured grids the parallel heat conduction uses the conservative
support-operator method (`conduction_method = petscoperator`) by default; the
input sets it explicitly. It conserves energy exactly, but on irregular meshes
its solution error converges only at order 0.7-1.4, against order 2 for
`conduction_method = original`, which does not conserve energy exactly.
Choose `original` when accuracy matters more than exact conservation. On 8
cores, t = 10^4 (`nout=100 timestep=100` on the command line) took about 10 min, against
about 19 min for the structured example (see the unstructured-FCI follow-up
reports for a detailed comparison). The grid generation needs about 1.4 GB of
memory at the default 3 mm mesh size (2 GB at 2.5 mm).
