#pragma once
#include "dictionary.H"
#include "error.H"
#include <cmath>
namespace Foam {
inline scalar readParticleDiameterShape(const dictionary& input) {
    if (input.found("dSigma"))
        FatalIOErrorInFunction(input) << "dSigma is replaced by RR shape dn; set dMin=dMax for monodisperse particles" << exit(FatalIOError);
    const scalar n=input.lookupOrDefault<scalar>("dn",1);
    if (!(std::isfinite(n) && n>0))
        FatalIOErrorInFunction(input) << "RR shape dn must be finite and positive" << exit(FatalIOError);
    return n;
}
}
