#pragma once
#ifndef INTEGRATE_H
#define INTEGRATE_H

#include <bout/bout_types.hxx>
#include <bout/coordinates.hxx>
#include <bout/field3d.hxx>
#include <bout/fv_ops_impl.hxx>

#include "../include/hermes_build_config.hxx"

/// Get the first argument from a parameter pack
template <typename Head, typename... Tail>
auto firstArg(const Head& head, Tail...) {
  return head;
}

inline auto threePointStencil(BoutReal c, BoutReal m, BoutReal p) {
  // TODO(peter): We can remove this #ifdef guard after switching to C++20
#if __cpp_designated_initializers >= 201707L
  return FV::Stencil1D{
      .c = c, .m = m, .p = p, .mm = BoutNaN, .pp = BoutNaN, .L = BoutNaN, .R = BoutNaN};
#else
  return FV::Stencil1D{c, m, p, BoutNaN, BoutNaN, BoutNaN, BoutNaN};
#endif
}

/// Return the value at the left of a cell,
/// given cell centre values at this cell and two neighbours
template <typename CellEdges>
BoutReal cellLeft(BoutReal c, BoutReal m, BoutReal p) {
  CellEdges cellboundary;
  FV::Stencil1D s = threePointStencil(c, m, p);
  cellboundary(s);
  return s.L;
}

/// Return the value at the right of a cell,
/// given cell centre values at this cell and two neighbours
template <typename CellEdges>
BoutReal cellRight(BoutReal c, BoutReal m, BoutReal p) {
  CellEdges cellboundary;
  FV::Stencil1D s = threePointStencil(c, m, p);
  cellboundary(s);
  return s.R;
}

/// The field that holds the value in the next cell along the magnetic field
/// from cell i, at index i.yp() (dir = +1) or i.ym() (dir = -1): on FCI
/// meshes the parallel slice, otherwise the field itself
template <typename T>
const T& alongField(const T& f, int dir, bool fci) {
  return fci ? f.ynext(dir) : f;
}

/// On FCI meshes, averages along the magnetic field need the parallel slices
template <typename T>
void requireParallelSlices(const T& f, bool fci) {
  if constexpr (std::is_base_of_v<Field3D, T>) {
    if (fci and not f.hasParallelSlices()) {
      throw BoutException("Averaging along the magnetic field on an FCI mesh needs the "
                          "parallel slices of every field it uses");
    }
  }
}

/// Take a function of BoutReals, and a region. Return
/// a function which takes fields (e.g. Field2D, Field3D),
/// and for every cell in the region evaluates the function
/// at quadrature points with weights.
/// These weights sum to 1, resulting in volume averaged values.
///
/// Uses a limiter to calculate values at cell edges. This is
/// needed so that as Ne goes to zero in a cell then atomic
/// rates also go to zero.
///
/// Example
///   Field3D Ne = ..., Te = ...;
///
///   Field3D result = cellAverage(
///          [](BoutReal Ne, BoutReal Te) {return Ne*Te;} // The function to evaluate
///          Ne.getRegion("RGN_NOBNDRY")  // The region to iterate over
///          )(Ne, Te);                   // The input fields
///
/// Note that the order of the arguments to the lambda function
/// is the same as the input fields.
///
template <typename CellEdges = hermes::Limiter, typename Function, typename RegionType>
auto cellAverage(Function func, const RegionType& region) {
  // Note: Capture by value or func and region go out of scope
  return [=](const auto&... args) {
    // Use the first argument to set the result mesh etc.
    Field3D result{emptyFrom(firstArg(args...))};
    result.allocate();

    // Get the coordinate Jacobian
    auto J = result.getCoordinates()->J();
    // Neighbours along the magnetic field: on FCI meshes in the parallel slices
    const bool fci = result.isFci();
    requireParallelSlices(J, fci);
    (requireParallelSlices(args, fci), ...);
    BOUT_FOR(i, region) {
      // Offset indices
      auto yp = i.yp();
      auto ym = i.ym();
      auto Ji = J[i];
      auto Jm = alongField(J, -1, fci)[ym];
      auto Jp = alongField(J, +1, fci)[yp];

      // Integrate in Y using Simpson's rule
      // Using limiter to calculate cell edge values
      result[i] =
          4. / 6 * func((args[i])...)
          + (Ji + Jm) / (12. * Ji)
                * func(cellLeft<CellEdges>(args[i], alongField(args, -1, fci)[ym],
                                           alongField(args, +1, fci)[yp])...)
          + (Ji + Jp) / (12. * Ji)
                * func(cellRight<CellEdges>(args[i], alongField(args, -1, fci)[ym],
                                            alongField(args, +1, fci)[yp])...);
    }
    return result;
  };
}

#endif // INTEGRATE_H
