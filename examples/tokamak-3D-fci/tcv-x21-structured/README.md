3D FCI example: TCV-X21 divertor geometry (structured grid)
==========================================================

Electron-ion transport (the reference input: h+ and e with density, momentum
and pressure, Braginskii collisions, friction, heat exchange, ion viscosity
and conduction, anomalous diffusion, sheaths) in the edge of the TCV-X21
reference scenario: between the core flux surface at psin = 0.93 and an outer
line made of two flux surfaces and two straight divertor plates, 1/16 of the
torus with 2 poloidal planes. Field lines that leave through the outer line
(the scrape-off layer and the plates) get sheath boundary conditions.
[`../tcv-x21-unstructured`](../tcv-x21-unstructured) is the same case on an
unstructured grid.

# Grid generation

The grid comes from zoidberg's `examples/tcv.py`, which needs the TCV-X21
data (https://github.com/SPCData/TCV-X21: the GEQDSK file of shot 65402 and
the divertor polygon). Set `tcv_datafolderpath` (the TCV-X21 repository) and
`write_folder` (this directory) in the script and run it:

```bash
python /path/to/zoidberg/examples/tcv.py
```

It writes `TCV_X21_68_2_512_16_1.fci.grid.nc` and, after marking the end points
beyond the last cell centre as wall hits, `cor_TCV_X21_68_2_512_16_1.fci.grid.nc`,
which this input uses (68 x 2 x 512; about 4 min and 2 GB). The grid contains
the poloidal flux `psi`, which the input uses for the initial profiles and
the sources.

# Running

```bash
mpirun -np 2 hermes-3 -d .
```

With the default conduction method (`original`) the run slows down strongly
after t = 90 (in units of 1/Omega_ci). The support-operator method
(`conduction_method = petscoperator`) needs the cell numbering of the grid
file, which `tcv.py` does not write yet.
