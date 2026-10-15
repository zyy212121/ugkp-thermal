#pragma once
#include "MixtureThermo.H"

namespace ugkwp
{
// Inputs gradNormalArea already contain oriented face area. Output is kg/s,

template<class Real, int Ns, class TableReal>
UGKWP_GAS_HD inline bool correctedSpeciesDiffusion
(
    const Real (&massFraction)[Ns], const Real (&gradNormalArea)[Ns],
    Real density, const Real* molecularD, Real eddyViscosity, Real schmidt,
    Real temperature, const SpeciesThermoView<TableReal,Ns>& thermo,
    Real (&flux)[Ns], Real& enthalpyFlux, const bool trialEligibility=false, bool* evaluated=nullptr
)
{
    if(evaluated)*evaluated=false;
    bool valid=gasFinite(density) && gasFinite(eddyViscosity) && gasFinite(schmidt);
    if(trialEligibility && !valid)return false;
    if(trialEligibility && (!(density>Real(0)) || eddyViscosity<Real(0) || !(schmidt>Real(0))))return false;
    Real raw[Ns], corrected[Ns], sumRaw=Real(0),sumY=Real(0);
    for(int k=0;k<Ns;++k)
    {
        const Real D=molecularD?molecularD[k]:Real(0);
        const bool inputValid=gasFinite(D)&&gasFinite(massFraction[k])&&gasFinite(gradNormalArea[k]);
        if(trialEligibility && !inputValid)return false;
        valid=inputValid&&valid;
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
        gasThermoDetail::SpeciesValues<Real> values;bool propertyEvaluated=false;
        const auto property=trialEligibility
            ?gasThermoDetail::evaluateSpecies<Real,Ns,TableReal,false>(k,temperature,thermo,values,&propertyEvaluated)
            :gasThermoDetail::evaluateSpecies<Real,Ns,TableReal,true,true>(k,temperature,thermo,values,&propertyEvaluated);
        if(!propertyEvaluated)return false;
        const Real h=values.h;
        const bool propertyValid=property==ThermoStatus::Success&&gasFinite(h)&&gasFinite(corrected[k]);
        if(trialEligibility && !propertyValid)return false;
        valid=propertyValid&&valid;
        energy+=h*corrected[k];
    }
    if(trialEligibility && !gasFinite(energy))return false;
    valid=gasFinite(energy)&&valid;
    for(int k=0;k<Ns;++k)flux[k]=corrected[k];
    enthalpyFlux=energy;
    if(evaluated)*evaluated=true;
    return valid;
}
}
