#include "gtest/gtest.h"

#include "fake_mesh_fixture.hxx"
#include "test_extras.hxx" // FakeMesh

#include "../../include/div_ops.hxx"

#include <cmath>

/// Global mesh
namespace bout {
namespace globals {
extern Mesh* mesh;
} // namespace globals
} // namespace bout

using namespace bout::globals;

// FCI::dagp_fv on an analytic sheared slab: index coordinates (x, z) with
// R = hx x and Z = hz z + s R. For this affine geometry the face factors are
// fac_XX = hz / hx, fac_ZZ = (1 + s^2) hx / hz, fac_XZ = fac_ZX = -s / 2 (the
// operator uses the sum of two central differences, i.e. twice the averaged
// one-cell difference, for the cross terms) and the cell area is hx hz.
namespace {
constexpr int NX = 10;
constexpr int NY = 3;
constexpr int NZ = 12;
constexpr BoutReal hx = 0.1;
constexpr BoutReal hz = 0.05;
constexpr BoutReal shear = 0.3;

BoutReal Rof(int x) { return hx * x; }
BoutReal Zof(int x, int z) { return hz * z + shear * Rof(x); }

// Quadratic test function and its derivatives
constexpr BoutReal alpha = 1.3, beta = 0.7, gamma_ = -0.4, delta = 0.2, eps = -0.5;
BoutReal fquad(BoutReal R, BoutReal Z) {
  return alpha * R * R + beta * R * Z + gamma_ * Z * Z + delta * R + eps * Z;
}
BoutReal dfdR(BoutReal R, BoutReal Z) { return 2 * alpha * R + beta * Z + delta; }
} // namespace

class DagpFvTest : public FakeMeshFixture_tmpl<NX, NY, NZ, false, 1> {
public:
  DagpFvTest() {
    Options values;
    values["dagp_fv_XX"] = hz / hx;
    values["dagp_fv_XZ"] = -0.5 * shear;
    values["dagp_fv_ZX"] = -0.5 * shear;
    values["dagp_fv_ZZ"] = (1 + shear * shear) * hx / hz;
    values["dagp_fv_volume"] = hx * hz;
    static_cast<FakeMesh*>(mesh)->setGridDataSource(new FakeGridDataSource(values));
  }

  template <typename F>
  static Field3D make(F&& func) {
    Field3D result{0.0};
    result.allocate();
    for (int x = 0; x < mesh->LocalNx; x++) {
      for (int y = 0; y < mesh->LocalNy; y++) {
        for (int z = 0; z < mesh->LocalNz; z++) {
          result(x, y, z) = func(x, z);
        }
      }
    }
    return result;
  }
};

TEST_F(DagpFvTest, ConstantGivesZero) {
  FCI::dagp_fv dagp(*mesh);
  const Field3D a = make(
      [](int x, int z) { return 1.0 + 0.3 * x + 0.5 * std::cos(2 * M_PI * z / NZ); });
  const Field3D f{3.0};
  const Field3D result = dagp(a, f, false);
  BOUT_FOR(i, result.getRegion("RGN_NOBNDRY")) { EXPECT_NEAR(result[i], 0.0, 1e-12); }
}

TEST_F(DagpFvTest, QuadraticExactOnShearedSlab) {
  // With a = 1 the scheme is exact for quadratic f on this affine grid:
  // div(grad f) = 2 alpha + 2 gamma. z is periodic but f is not, so only
  // cells whose stencil does not wrap around in z are checked.
  FCI::dagp_fv dagp(*mesh);
  const Field3D a{1.0};
  const Field3D f = make([](int x, int z) { return fquad(Rof(x), Zof(x, z)); });
  const Field3D result = dagp(a, f, false);
  const BoutReal expected = 2 * alpha + 2 * gamma_;
  for (int x = mesh->xstart; x <= mesh->xend; x++) {
    for (int y = mesh->ystart; y <= mesh->yend; y++) {
      for (int z = 1; z < NZ - 1; z++) {
        EXPECT_NEAR(result(x, y, z), expected, 1e-9) << "x=" << x << " z=" << z;
      }
    }
  }
}

TEST_F(DagpFvTest, ConservativeWithVariableCoefficient) {
  // Orthogonal case (s = 0 in the fluxes through the x boundaries) with f
  // independent of x: no flux through the x boundaries, z periodic, so the
  // volume integral of the divergence vanishes for any a.
  Options values;
  values["dagp_fv_XX"] = hz / hx;
  values["dagp_fv_XZ"] = 0.0;
  values["dagp_fv_ZX"] = 0.0;
  values["dagp_fv_ZZ"] = hx / hz;
  values["dagp_fv_volume"] = hx * hz;
  static_cast<FakeMesh*>(mesh)->setGridDataSource(new FakeGridDataSource(values));
  FCI::dagp_fv dagp(*mesh);
  const Field3D a = make(
      [](int x, int z) { return 1.0 + 0.2 * x + 0.5 * std::cos(2 * M_PI * z / NZ); });
  const Field3D f = make([](int, int z) {
    return std::sin(2 * M_PI * z / NZ) + 0.3 * std::cos(4 * M_PI * z / NZ);
  });
  const Field3D result = dagp(a, f, false);
  BoutReal integral = 0.0, total = 0.0;
  BOUT_FOR_SERIAL(i, result.getRegion("RGN_NOBNDRY")) {
    integral += result[i] * hx * hz;
    total += std::abs(result[i]) * hx * hz;
  }
  ASSERT_GT(total, 0.0);
  EXPECT_LT(std::abs(integral), 1e-13 * total);
}

TEST_F(DagpFvTest, FlowsAreFaceFluxesDividedByVolume) {
  // flow_xlow holds the flux through the lower x face of each cell divided by
  // the cell volume (P-062 in the master report: the flow diagnostics should
  // be the face fluxes themselves). For quadratic f and a = 1 the discrete
  // face flux equals the exact one, -hz * df/dR at the face centre.
  FCI::dagp_fv dagp(*mesh);
  const Field3D a{1.0};
  const Field3D f = make([](int x, int z) { return fquad(Rof(x), Zof(x, z)); });
  Field3D flow_xlow, flow_zlow;
  dagp(a, f, flow_xlow, flow_zlow, false);
  for (int x = mesh->xstart + 1; x <= mesh->xend; x++) {
    for (int y = mesh->ystart; y <= mesh->yend; y++) {
      for (int z = 1; z < NZ - 1; z++) {
        const BoutReal Rm = hx * (x - 0.5);
        const BoutReal Zm = hz * z + shear * Rm;
        const BoutReal face_flux = -hz * dfdR(Rm, Zm);
        EXPECT_NEAR(flow_xlow(x, y, z) * hx * hz, face_flux, 1e-10)
            << "x=" << x << " z=" << z;
      }
    }
  }
}
