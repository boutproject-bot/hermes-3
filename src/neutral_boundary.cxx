#include "hermes_utils.hxx"
#include "bout/mesh.hxx"
#include <bout/constants.hxx>
using bout::globals::mesh;

#include "../include/neutral_boundary.hxx"

NeutralBoundary::NeutralBoundary(std::string name, Options& alloptions,
                                 [[maybe_unused]] Solver* solver)
    : NamedComponent(name,
                     {writeBoundaryFinal("species:{name}:{outputs}"),
                      writeBoundaryFinalIfSet("species:{name}:{conditional_outputs}"),
                      readWrite("species:{name}:energy_source"),
                      readOnly("species:{name}:AA")}) {

  auto& options = alloptions[name];
  const Options& units = alloptions["units"];
  Tnorm = units["eV"];

  diagnose =
      options["diagnose"].doc("Save additional diagnostics?").withDefault<bool>(false);
  lower_y = options["neutral_boundary_lower_y"]
                .doc("Boundary on lower y?")
                .withDefault<bool>(true);
  upper_y = options["neutral_boundary_upper_y"]
                .doc("Boundary on upper y?")
                .withDefault<bool>(true);
  yboundary = bout::boundary::YBoundary(*mesh, lower_y, upper_y);
  sol = options["neutral_boundary_sol"].doc("Boundary on SOL?").withDefault<bool>(false);
  pfr = options["neutral_boundary_pfr"].doc("Boundary on PFR?").withDefault<bool>(false);
  requireStructuredMesh("neutral_boundary_sol", sol);
  requireStructuredMesh("neutral_boundary_pfr", pfr);

  target_energy_refl_factor = options["target_energy_refl_factor"]
                                  .doc("Fraction of energy retained by neutral particles "
                                       "after wall reflection at target")
                                  .withDefault<BoutReal>(0.75);

  sol_energy_refl_factor = options["sol_energy_refl_factor"]
                               .doc("Fraction of energy retained by neutral particles "
                                    "after wall reflection at SOL")
                               .withDefault<BoutReal>(0.75);

  pfr_energy_refl_factor = options["pfr_energy_refl_factor"]
                               .doc("Fraction of energy retained by neutral particles "
                                    "after wall reflection at PFR")
                               .withDefault<BoutReal>(0.75);

  target_fast_refl_fraction =
      options["target_fast_refl_fraction"]
          .doc("Fraction of neutrals that are undergoing fast reflection at the target")
          .withDefault<BoutReal>(0.8);

  sol_fast_refl_fraction =
      options["sol_fast_refl_fraction"]
          .doc("Fraction of neutrals that are undergoing fast reflection at the sol")
          .withDefault<BoutReal>(0.8);

  pfr_fast_refl_fraction =
      options["pfr_fast_refl_fraction"]
          .doc("Fraction of neutrals that are undergoing fast reflection at the pfr")
          .withDefault<BoutReal>(0.8);

  substitutePermissions("name", {name});
  substitutePermissions("outputs", {"density", "temperature", "pressure"});
  substitutePermissions("conditional_outputs", {"velocity", "momentum"});
}

void NeutralBoundary::transform_impl(GuardedOptions& state) {
  auto species = state["species"][objectName()];
  const BoutReal AA = get<BoutReal>(species["AA"]);

  // FCI meshes have no field-aligned form: there the boundary values are
  // the parallel slices of the fields
  const bool fci = mesh->isFci();
  const auto aligned = [fci](const Field3D& f) { return fci ? f : toFieldAligned(f); };
  const auto unaligned = [fci](const Field3D& f) {
    return fci ? f : fromFieldAligned(f);
  };

  Field3D Nn = aligned(GET_NOBOUNDARY(Field3D, species["density"]));
  Field3D Pn = aligned(GET_NOBOUNDARY(Field3D, species["pressure"]));
  Field3D Tn = aligned(GET_NOBOUNDARY(Field3D, species["temperature"]));

  Field3D Vn = IS_SET_NOBOUNDARY(species["velocity"])
                   ? aligned(getNoBoundary<Field3D>(species["velocity"]))
                   : zeroFrom(Nn);

  Field3D NVn = IS_SET_NOBOUNDARY(species["momentum"])
                    ? aligned(getNoBoundary<Field3D>(species["momentum"]))
                    : zeroFrom(Nn);

  // Get the energy source, or create if not set
  Field3D energy_source = species.isSet("energy_source")
                              ? aligned(getNonFinal<Field3D>(species["energy_source"]))
                              : zeroFrom(Nn);

  Coordinates* coord = mesh->getCoordinates();
  target_energy_source = 0;
  wall_energy_source = 0;

  // Targets: on each point, current() is the cell next to the target and
  // next() the boundary cell beyond it
  yboundary.iter([&](auto& pnt) {
    const auto i = pnt.ind();

    // Free boundary condition on Nn, Pn, Tn
    // This is problematic when Nn, Pn or Tn are zero
    // pnt.next(Nn) = SQ(pnt.current(Nn)) / pnt.prev(Nn);
    // pnt.next(Pn) = SQ(pnt.current(Pn)) / pnt.prev(Pn);
    // pnt.next(Tn) = SQ(pnt.current(Tn)) / pnt.prev(Tn);

    // Neumann boundary condition: do not extrapolate, but
    // assume the target value is same as the final cell centre.
    // Shouldn't affect results much and more resilient to positivity issues
    pnt.next(Nn) = pnt.current(Nn);
    pnt.next(Pn) = pnt.current(Pn);
    pnt.next(Tn) = pnt.current(Tn);

    // No-flow boundary condition
    pnt.next(Vn) = -pnt.current(Vn);
    pnt.next(NVn) = -pnt.current(NVn);

    // Calculate midpoint values at wall
    const BoutReal nnsheath = 0.5 * (pnt.next(Nn) + pnt.current(Nn));
    const BoutReal tnsheath = 0.5 * (pnt.next(Tn) + pnt.current(Tn));

    // Thermal speed
    const BoutReal v_th =
        0.25 * sqrt(8 * tnsheath / (PI * AA)); // Stangeby p.69 eqns. 2.21, 2.24

    // Approach adapted from D. Power thesis 2023
    BoutReal T_FC = 3 / Tnorm; // Franck-Condon temp (hardcoded for now)

    // Outgoing neutral heat flux [W/m^2]
    // This is rearranged from Power for clarity - note definition of v_th.
    // Uses standard Stangeby 1D static Maxwellian particle/heat fluxes for fast terms
    // and simply Q = T * particle flux for the monoenergetic thermal reflected
    // population.
    BoutReal q = 2 * nnsheath * tnsheath * v_th // Incident energy
                 - (target_energy_refl_factor * target_fast_refl_fraction) * 2 * nnsheath
                       * tnsheath * v_th // Fast reflected energy
                 - (1 - target_fast_refl_fraction) * T_FC * nnsheath
                       * v_th; // Thermal reflected energy

    BoutReal cooling_source = 0.0;
    if (fci) {
      // Only where the cell touches the target, through its face area, as
      // in sheath_boundary_fci
      if (std::abs(pnt.offset()) != 1) {
        return;
      }
      const BoutReal flow =
          q * (pnt.dir() > 0 ? coord->cell_area_yhigh()[i] : coord->cell_area_ylow()[i]);
      cooling_source = flow / coord->cell_volume()[i];
    } else {
      // Cross-sectional area in XZ plane:
      BoutReal da = (pnt.current(coord->J()) + pnt.next(coord->J()))
                    / (sqrt(pnt.current(coord->g_22())) + sqrt(pnt.next(coord->g_22())))
                    * 0.5 * (pnt.current(coord->dx()) + pnt.next(coord->dx())) * 0.5
                    * (pnt.current(coord->dz()) + pnt.next(coord->dz())); // [m^2]

      // Multiply by area to get energy flow (power)
      BoutReal flow = q * da; // [W]

      // Divide by cell volume to get source [W/m^3]
      cooling_source =
          flow / (coord->dx()[i] * coord->dy()[i] * coord->dz()[i] * coord->J()[i]);
    }

    // Subtract from cell next to boundary
    energy_source[i] -= cooling_source;
    target_energy_source[i] -= cooling_source;
  });

  // SOL edge
  if (sol) {
    if (mesh->lastX()) { // Only do this for the processor which has the edge region
      for (int iy = 0; iy < mesh->LocalNy; iy++) {
        for (int iz = 0; iz < mesh->LocalNz; iz++) {

          auto i = indexAt(Nn, mesh->xend, iy, iz);      // Final domain cell
          auto ig = indexAt(Nn, mesh->xend + 1, iy, iz); // Guard cell

          // Calculate midpoint values at wall
          const BoutReal nnsheath = 0.5 * (Nn[ig] + Nn[i]);
          const BoutReal tnsheath = 0.5 * (Tn[ig] + Tn[i]);

          // Thermal speed of static Maxwellian in one direction
          const BoutReal v_th =
              0.25 * sqrt(8 * tnsheath / (PI * AA)); // Stangeby p.69 eqns. 2.21, 2.24

          // Approach adapted from D. Power thesis 2023
          BoutReal T_FC = 3 / Tnorm; // Franck-Condon temp (hardcoded for now)

          // Outgoing neutral heat flux [W/m^2]
          // This is rearranged from Power for clarity - note definition of v_th.
          BoutReal q = 2 * nnsheath * tnsheath * v_th // Incident energy
                       - (sol_energy_refl_factor * sol_fast_refl_fraction) * 2 * nnsheath
                             * tnsheath * v_th // Fast reflected energy
                       - (1 - sol_fast_refl_fraction) * T_FC * nnsheath
                             * v_th; // Thermal reflected energy

          // Multiply by radial cell area to get power
          // Expanded form of the calculation for clarity

          // Converts dy to poloidal length: dl = dy / sqrt(g22) = dy * h_theta
          BoutReal dpol = 0.5 * (coord->dy()[i] + coord->dy()[ig]) * 1
                          / (0.5 * (sqrt(coord->g22()[i]) + sqrt(coord->g22()[ig])));

          // Converts dz to toroidal length:  = dz*sqrt(g_33) = dz * R = 2piR
          BoutReal dtor = 0.5 * (coord->dz()[i] + coord->dz()[ig]) * 0.5
                          * (sqrt(coord->g_33()[i]) + sqrt(coord->g_33()[ig]));

          BoutReal da = dpol * dtor; // [m^2]

          // Multiply by area to get energy flow (power)
          BoutReal flow = q * da; // [W]

          // Divide by cell volume to get source [W/m^3]
          BoutReal cooling_source = flow
                                    / (coord->J()[i] * coord->dx()[i] * coord->dy()[i]
                                       * coord->dz()[i]); // [W m^-3]

          // Subtract from cell next to boundary
          energy_source[i] -= cooling_source;
          wall_energy_source[i] -= cooling_source;
        }
      }
    }
  }

  // PFR edge
  if (pfr) {
    if ((mesh->firstX())
        and (!mesh->periodicY(
            mesh->xstart))) { // do loop if inner edge and not periodic (i.e. PFR)
      for (int iy = 0; iy < mesh->LocalNy; iy++) {
        for (int iz = 0; iz < mesh->LocalNz; iz++) {

          auto i = indexAt(Nn, mesh->xstart, iy, iz);      // Final domain cell
          auto ig = indexAt(Nn, mesh->xstart - 1, iy, iz); // Guard cell

          // Calculate midpoint values at wall
          const BoutReal nnsheath = 0.5 * (Nn[ig] + Nn[i]);
          const BoutReal tnsheath = 0.5 * (Tn[ig] + Tn[i]);

          // Thermal speed of static Maxwellian in one direction
          const BoutReal v_th =
              0.25 * sqrt(8 * tnsheath / (PI * AA)); // Stangeby p.69 eqns. 2.21, 2.24

          // Approach adapted from D. Power thesis 2023
          BoutReal T_FC = 3 / Tnorm; // Franck-Condon temp (hardcoded for now)

          // Outgoing neutral heat flux [W/m^2]
          // This is rearranged from Power for clarity - note definition of v_th.
          BoutReal q = 2 * nnsheath * tnsheath * v_th // Incident energy
                       - (pfr_energy_refl_factor * pfr_fast_refl_fraction) * 2 * nnsheath
                             * tnsheath * v_th // Fast reflected energy
                       - (1 - pfr_fast_refl_fraction) * T_FC * nnsheath
                             * v_th; // Thermal reflected energy

          // Multiply by radial cell area to get power
          // Expanded form of the calculation for clarity

          // Converts dy to poloidal length: dl = dy / sqrt(g22) = dy * h_theta
          BoutReal dpol = 0.5 * (coord->dy()[i] + coord->dy()[ig]) * 1
                          / (0.5 * (sqrt(coord->g22()[i]) + sqrt(coord->g22()[ig])));

          // Converts dz to toroidal length:  = dz*sqrt(g_33) = dz * R = 2piR
          BoutReal dtor = 0.5 * (coord->dz()[i] + coord->dz()[ig]) * 0.5
                          * (sqrt(coord->g_33()[i]) + sqrt(coord->g_33()[ig]));

          BoutReal da = dpol * dtor; // [m^2]

          // Multiply by area to get energy flow (power)
          BoutReal flow = q * da; // [W]

          // Divide by cell volume to get source [W/m^3]
          BoutReal cooling_source = flow
                                    / (coord->J()[i] * coord->dx()[i] * coord->dy()[i]
                                       * coord->dz()[i]); // [W m^-3]

          // Subtract from cell next to boundary
          energy_source[i] -= cooling_source;
          wall_energy_source[i] -= cooling_source;
        }
      }
    }
  }

  // Set density, pressure and temperature, now with boundary conditions
  setBoundary(species["density"], unaligned(Nn));
  setBoundary(species["temperature"], unaligned(Tn));
  setBoundary(species["pressure"], unaligned(Pn));
  if (IS_SET_NOBOUNDARY(species["velocity"])) {
    setBoundary(species["velocity"], unaligned(Vn));
  }
  if (IS_SET_NOBOUNDARY(species["momentum"])) {
    setBoundary(species["momentum"], unaligned(NVn));
  }

  // Set energy source (negative in cell next to sheath)
  // Note: energy_source includes any sources previously set in other components
  set(species["energy_source"], unaligned(energy_source));
}

void NeutralBoundary::outputVars(Options& state) {

  // Normalisations
  auto Nnorm = get<BoutReal>(state["Nnorm"]);
  auto Omega_ci = get<BoutReal>(state["Omega_ci"]);
  auto Tnorm = get<BoutReal>(state["Tnorm"]);
  BoutReal Pnorm = SI::qe * Tnorm * Nnorm; // Pressure normalisation
  const std::string& name = objectName();

  if (diagnose) {

    // Save particle and energy source for the species created during recycling

    // Target recycling

    if ((sol) or (pfr)) {
      set_with_attrs(
          state[{std::string("E") + name + std::string("_wall_refl")}],
          wall_energy_source,
          {{"time_dimension", "t"},
           {"units", "W m^-3"},
           {"conversion", Pnorm * Omega_ci},
           {"standard_name", "energy source"},
           {"long_name", std::string("Wall reflection energy source of ") + name},
           {"source", "neutral_boundary"}});
    }

    set_with_attrs(
        state[{std::string("E") + name + std::string("_target_refl")}],
        target_energy_source,
        {{"time_dimension", "t"},
         {"units", "W m^-3"},
         {"conversion", Pnorm * Omega_ci},
         {"standard_name", "energy source"},
         {"long_name", std::string("Wall reflection energy source of ") + name},
         {"source", "neutral_boundary"}});
  }
}
