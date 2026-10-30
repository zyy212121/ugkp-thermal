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
    using namespace ugkp::diagnostics;
    (void)trialEligibility; // compatibility only: there is one physical evaluation.
    if(evaluated)*evaluated=false;
    if(!gasFinite(density)||!gasFinite(eddyViscosity)||!gasFinite(schmidt)||schmidt==Real(0)) {
        gasThermoDetail::failure(thermo.diagnostics,ThermoStatus::NonFiniteInput,(!gasFinite(density)||!gasFinite(eddyViscosity)||!gasFinite(schmidt))?Code::NonFinite:Code::ZeroDenominator,-1,!gasFinite(density)?density:(!gasFinite(eddyViscosity)?eddyViscosity:schmidt),!gasFinite(density)?Quantity::Density:(!gasFinite(eddyViscosity)?Quantity::Unspecified:Quantity::Denominator));
        return false;
    }
    Real raw[Ns],corrected[Ns],Y[Ns],sumRaw=0,sumY=0;
    for(int k=0;k<Ns;++k) {
        const Real D=molecularD?molecularD[k]:Real(0);
        if(!gasFinite(D)||!gasFinite(massFraction[k])||!gasFinite(gradNormalArea[k])) {
            gasThermoDetail::failure(thermo.diagnostics,ThermoStatus::NonFiniteInput,Code::NonFinite,k,!gasFinite(massFraction[k])?massFraction[k]:(!gasFinite(D)?D:gradNormalArea[k]));
            return false;
        }
        Y[k]=gasThermoDetail::maxValue(Real(0),gasThermoDetail::minValue(Real(1),massFraction[k]));
        if(Y[k]!=massFraction[k])recordWarning(thermo.diagnostics,makeRecord(Code::FiniteYClip,2305,0,0,EntityKind::Species,k,k,massFraction[k],0,Y[k],Quantity::MassFraction));
        sumY+=Y[k];raw[k]=-density*(D+eddyViscosity/schmidt)*gradNormalArea[k];sumRaw+=raw[k];
    }
    const Real tolerance=Real(64*Ns)*std::numeric_limits<Real>::epsilon();
    if(gasThermoDetail::absValue(sumY-Real(1))>tolerance)
        recordWarning(thermo.diagnostics,makeRecord(Code::ConservationResidual,10005,0,0,EntityKind::None,-1,-1,sumY,1,sumY,Quantity::Conservation));
    // Close on the largest carrier, including when the last species is a
    // nonzero trace. Subtracting carrier-sized raw fluxes first loses the
    // trace channel to roundoff; assigning that remainder to a trace creates
    // an outflow unrelated to its inventory and can suppress the bulk flux.
    int dependent=0;
    for(int k=1;k<Ns;++k)if(Y[k]>Y[dependent])dependent=k;
    Real sumCorrected=0,energy=0;
    for(int k=0;k<Ns;++k){corrected[k]=raw[k]-Y[k]*sumRaw;if(k!=dependent)sumCorrected+=corrected[k];}
    // Original conservative diffusion closure, not composition renormalization.
    corrected[dependent]=-sumCorrected;
    for(int k=0;k<Ns;++k) {
        gasThermoDetail::SpeciesValues<Real> values;
        const auto status=gasThermoDetail::evaluateSpecies<Real,Ns,TableReal,false,true>(k,temperature,thermo,values);
        if(status!=ThermoStatus::Success)return false;
        if(!gasFinite(corrected[k])){gasThermoDetail::failure(thermo.diagnostics,ThermoStatus::NonFiniteInput,Code::NonFinite,k,corrected[k]);return false;}
        energy+=values.h*corrected[k];
    }
    if(!gasFinite(energy)){gasThermoDetail::failure(thermo.diagnostics,ThermoStatus::NonFiniteInput,Code::NonFinite,-1,energy);return false;}
    for(int k=0;k<Ns;++k)flux[k]=corrected[k];
    enthalpyFlux=energy;if(evaluated)*evaluated=true;return true;
}
}
