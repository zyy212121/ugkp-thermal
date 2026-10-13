#pragma once
#include "MixtureThermo.H"

namespace ugkwp
{
// Inputs gradNormalArea already contain oriented face area. Output is kg/s,
// with formation-inclusive species enthalpy transport in J/s. Failure is atomic.
template<class Real, int Ns, class TableReal>
UGKWP_GAS_HD inline bool correctedSpeciesDiffusion
(
    const Real (&massFraction)[Ns], const Real (&gradNormalArea)[Ns],
    Real density, const Real* molecularD, Real eddyViscosity, Real schmidt,
    Real temperature, const SpeciesThermoView<TableReal,Ns>& thermo,
    Real (&flux)[Ns], Real& enthalpyFlux, const bool trialEligibility=false
)
{
    if (!gasFinite(density) || !gasFinite(eddyViscosity) || !gasFinite(schmidt)) return false;
    if(trialEligibility && (!(density>Real(0)) || eddyViscosity<Real(0) || !(schmidt>Real(0))))return false;
    Real raw[Ns], corrected[Ns], sumRaw=Real(0),sumY=Real(0);
    for(int k=0;k<Ns;++k)
    {
        const Real D=molecularD?molecularD[k]:Real(0);
        if(!gasFinite(D)||!gasFinite(massFraction[k])
            ||!gasFinite(gradNormalArea[k]))return false;
        if(trialEligibility && (D<Real(0) || massFraction[k]<Real(0)))return false;
        if(trialEligibility)sumY+=massFraction[k];
        raw[k]=-density*(D+eddyViscosity/schmidt)*gradNormalArea[k];
        sumRaw+=raw[k];
    }
    if(trialEligibility && (!gasFinite(sumY)||gasThermoDetail::absValue(sumY-Real(1))>
        Real(64*Ns)*std::numeric_limits<Real>::epsilon()))return false;
    int dependent=Ns-1;
    // Preserve the established arithmetic unless its carrier is absent.
    if(massFraction[dependent]==Real(0))
        for(int k=0;k<Ns;++k)if(massFraction[k]>massFraction[dependent])dependent=k;
    Real sumCorrected=Real(0), energy=Real(0);
    for(int k=0;k<Ns;++k)
    {
        corrected[k]=raw[k]-massFraction[k]*sumRaw;
        if(k!=dependent)sumCorrected+=corrected[k];
    }
    // Do not assign closure roundoff to zero inventory. Its physical gradient
    // can still be nonzero (for example at a fixed-composition boundary).
    corrected[dependent]=-sumCorrected;
    for(int k=0;k<Ns;++k)
    {
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
