#pragma once
#include <limits>

namespace ugkp { namespace gasNumerics {

// Value-only startup controls, copied with each gas resident state/view.
// Geometry, physical validity and diagnostic severity are unchanged.
template<class Real> struct ScalarSolverControls {
    int laminarMaximumIterations = 10;
    Real laminarRelativeTolerance = Real(1.0e-8);
    int spaldingMaximumIterations = 64;
    Real spaldingTolerance = Real(32)*std::numeric_limits<Real>::epsilon();
    int thermalMaximumIterations = 80;
    Real thermalRelativeTolerance = Real(8)*std::numeric_limits<Real>::epsilon();
    int inletMachMaximumIterations = 48;
};

template<class To, class From>
inline ScalarSolverControls<To> castScalarSolverControls(const ScalarSolverControls<From>& in) {
    ScalarSolverControls<To> out;
    out.laminarMaximumIterations=in.laminarMaximumIterations;
    out.laminarRelativeTolerance=To(in.laminarRelativeTolerance);
    out.spaldingMaximumIterations=in.spaldingMaximumIterations;
    out.spaldingTolerance=To(in.spaldingTolerance);
    out.thermalMaximumIterations=in.thermalMaximumIterations;
    out.thermalRelativeTolerance=To(in.thermalRelativeTolerance);
    out.inletMachMaximumIterations=in.inletMachMaximumIterations;
    return out;
}
}}
