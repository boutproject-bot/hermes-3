3D FCI example on unstructured meshes: TCV-X21 divertor geometry
===============================================================

Unstructured-mesh version of [`../tcv-x21-structured`](../tcv-x21-structured):
same magnetic field (GEQDSK equilibrium of TCV shot 65402), same domain (the
core flux surface at psin = 0.93 to an outer line of two flux surfaces and
two straight divertor plates, 1/16 of the torus with 2 poloidal planes) and
same physics (the reference electron-ion input with sheaths). Every poloidal
plane is a triangle mesh made independently with [gmsh](https://gmsh.info);
the core surface and the outer line are Dirichlet walls.

# Grid generation

Requires zoidberg with `zoidberg.unstructured` and gmsh, the zoidberg
repository (the boundary lines come from its `examples/tcv.py`) and the
TCV-X21 data (https://github.com/SPCData/TCV-X21):

```bash
pip install gmsh
python create_grid.py --tcv-data /path/to/TCV-X21
python create_grid.py --tcv-data /path/to/TCV-X21 --zoidberg /path/to/zoidberg
```

The mesh is graded: 1 mm at the outer line, growing to 4 mm at 12 mm from it
(`--wall-size`, `--wall-distance`, `--mesh-size`), so that the layer next to
the wall is resolved as finely as on the structured grid (0.9 mm across the
flux surfaces there). This gives 21692 nodes per plane, takes about 7 min
with the default 2 worker processes, needs about 7 GB of memory at peak and
writes the 553 MB file `tcv_x21_unstructured_2_0.004_0.001.fci.nc`. On this
mesh the volume of the cells whose field lines end on the outer line matches
the structured grid (forward 3.5 %, backward 5.1 % against 5.9 %). An
isotropic 4 mm mesh is cheaper but makes this layer 4 to 10 times thicker.

The grid is built with `map_walls=False` and `near_wall_hits=True`: heat
carried along the field lines leaves through the walls only via the sheath,
and end points between the last plasma nodes and the wall are sheath points,
as end points beyond the last cell centre on the structured grid. The wall
nodes are stored in the x guard cells (`MXG` = 2574 here), so with
`NXPE > 1` each processor needs at least `MXG` nodes.

# Differences in the input file

* `[mesh:paralleltransform:xzinterpolation] type = precomputed` (the
  interpolation weights come from the grid file).
* The wall values are set with `dirichlet_node(...)` instead of
  `dirichlet(...)`.
* The profiles use the grid variable `psi` (written by zoidberg from the
  equilibrium), as in the structured example.
* The parallel heat conduction uses the unstructured default, the
  flux-reconstructed divergence (`petscoperatork2`, the grid file contains
  its matrices); the structured example sets the support operators
  (`conduction_method = petscoperator`).

# Running

```bash
mpirun -np 4 hermes-3 -d .
```

The first outputs agree with the structured run (flow at the vertical plate,
temperature extremes), but the solver currently needs 10 to 100 times more
work than on the structured grid. The cause is the parallel ion viscosity at
the sheath points on the vertical plate; without `braginskii_ion_viscosity`
the run is about as cheap as the structured one. This is being investigated.
