#pragma once
#include "MixtureThermo.H"

namespace ugkwp
{
// Inputs gradNormalArea already contain oriented face area. Output is kg/s,
// with formation-inclusive species enthalpy transport in J/s. Failure is atomic.
template<class Real, int Ns>
UGKWP_GAS_HD inline bool correctedSpeciesDiffusion
(
    const Real (&massFraction)[Ns], const Real (&gradNormalArea)[Ns],
    Real density, const Real* molecularD, Real eddyViscosity, Real schmidt,
    Real temperature, const SpeciesThermoView<Real,Ns>& thermo,
    Real (&flux)[Ns], Real& enthalpyFlux
)
{
    if (!gasFinite(density) || !(density>Real(0)) || !gasFinite(eddyViscosity)
        || eddyViscosity<Real(0) || !gasFinite(schmidt) || !(schmidt>Real(0))) return false;
    Real raw[Ns], corrected[Ns], sumRaw=Real(0), sumY=Real(0);
    for(int k=0;k<Ns;++k)
    {
        const Real D=molecularD?molecularD[k]:Real(0);
        if(!gasFinite(D)||D<Real(0)||!gasFinite(massFraction[k])||massFraction[k]<Real(0)
            ||!gasFinite(gradNormalArea[k]))return false;
        sumY+=massFraction[k];
        raw[k]=-density*(D+eddyViscosity/schmidt)*gradNormalArea[k];
        sumRaw+=raw[k];
    }
    if(!gasFinite(sumY)||gasThermoDetail::absValue(sumY-Real(1))>
        Real(64*Ns)*std::numeric_limits<Real>::epsilon())return false;
    Real sumCorrected=Real(0), energy=Real(0);
    for(int k=0;k<Ns;++k)
    {
        corrected[k]=raw[k]-massFraction[k]*sumRaw;
        if(k==Ns-1)corrected[k]=-sumCorrected; // roundoff-only mass closure
        sumCorrected+=corrected[k];
        const Real h=speciesH(k,temperature,thermo);
        if(!gasFinite(h)||!gasFinite(corrected[k]))return false;
        energy+=h*corrected[k];
    }
    if(!gasFinite(energy))return false;
    for(int k=0;k<Ns;++k)flux[k]=corrected[k];
    enthalpyFlux=energy;
    return true;
}
}
