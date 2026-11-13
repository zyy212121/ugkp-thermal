#pragma once
#include "FoamSolverScheme.H"
#include "IStringStream.H"
namespace ugkp { namespace solverControls {
// Keep the existing OpenFOAM-shaped selector grammar, but its sole configurable
// source is the immutable scheme snapshot, never system/fvSchemes.
inline Foam::dictionary readGasNumericsDictionary(const Foam::Time& time,
    const Foam::word& region=Foam::word::null) {
    Foam::IStringStream defaults(
        "fluxScheme Kurganov; gasLimiter none; "
        "ddtSchemes { default Euler; } "
        "gradSchemes { default Gauss linear; } "
        "interpolationSchemes { default linear; } "
        "snGradSchemes { default corrected; } "
        "laplacianSchemes { default Gauss linear corrected; } "
        "divSchemes { default none; div(phi,U) Gauss upwind; "
        "div(phi,e) Gauss upwind; div(phi,K) Gauss upwind; "
        "div(phi,(p|rho)) Gauss upwind; "
        "div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear; }");
    Foam::dictionary result(defaults);
    result.merge(startupScheme(time).module("gasNumerics",region));
    return result;
}
}}
