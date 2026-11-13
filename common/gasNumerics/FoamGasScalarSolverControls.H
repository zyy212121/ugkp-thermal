#pragma once
#include "GasScalarSolverControls.H"
#include "../FoamSolverScheme.H"

namespace ugkp { namespace gasNumerics {
template<class Real>
inline ScalarSolverControls<Real> readGasScalarSolverControls(
    const ugkp::solverControls::Scheme& scheme,
    const Foam::word& region=Foam::word::null) {
    ScalarSolverControls<Real> controls;
    const char* source="effectiveDefault";
    scheme.read("wallFunctions","laminarMaximumIterations",controls.laminarMaximumIterations,source,region);
    scheme.read("wallFunctions","laminarRelativeTolerance",controls.laminarRelativeTolerance,source,region);
    scheme.read("wallFunctions","spaldingMaximumIterations",controls.spaldingMaximumIterations,source,region);
    scheme.read("wallFunctions","spaldingTolerance",controls.spaldingTolerance,source,region);
    scheme.read("wallFunctions","thermalMaximumIterations",controls.thermalMaximumIterations,source,region);
    scheme.read("wallFunctions","thermalRelativeTolerance",controls.thermalRelativeTolerance,source,region);
    scheme.read("gasBoundary","inletMachMaximumIterations",controls.inletMachMaximumIterations,source,region);
    return controls;
}
}}
