#pragma once
#include "MixtureThermo.H"
namespace ugkwp {
// Geometric occlusion applies only to gas conductive heat. It does not scale
// species transport, surface reactions, injection enthalpy or wall traction.
template<class Real>
UGKWP_GAS_HD inline bool wallHeatExposedFraction
(Real faceArea,Real representedContactArea,Real contactAreaScale,Real& exposed) noexcept
{
    if(!gasFinite(faceArea)||!gasFinite(representedContactArea)||!gasFinite(contactAreaScale)
        ||!(faceArea>Real(0))||representedContactArea<Real(0)
        ||!(contactAreaScale>Real(0))||contactAreaScale>Real(1))return false;
    const Real occupied=representedContactArea*contactAreaScale;
    if(!gasFinite(occupied)||occupied<Real(0))return false;
    const Real fraction=Real(1)-occupied/faceArea;
    exposed=fraction<Real(0)?Real(0):(fraction>Real(1)?Real(1):fraction);
    return true;
}
}
