#pragma once
#include "MixtureThermo.H"
#include "GasStateView.H"

namespace ugkwp
{


template<class State,class Primitive,class R>
UGKWP_GAS_HD inline void gasFaceSpeciesTransportProperties(const State& s,int f,int own,int nei,
    const Primitive& left,const Primitive& right,bool includeTurbulence,
    R* diffusion,R& density,R& eddyViscosity,R& temperature)
{
    const R w=s.faceWeight[f]<R(0)?R(0):(s.faceWeight[f]>R(1)?R(1):s.faceWeight[f]);
    temperature=nei>=0?w*left.T+(R(1)-w)*right.T:right.T;
    const R pressure=nei>=0?w*left.p+(R(1)-w)*right.p:right.p;
    density=nei>=0?w*left.rho+(R(1)-w)*right.rho:right.rho;
    eddyViscosity=includeTurbulence&&s.turbulenceModel!=0
        ?(nei>=0?w*s.nut[own]+(R(1)-w)*s.nut[nei]:s.nut[own]):R(0);
    for(int k=0;k<GasStateTraits<State>::speciesCount;++k)
        diffusion[k]=gasLocalMolecularDiffusivity(s,k,temperature,pressure);
}

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
    (void)trialEligibility;
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




    int dependent=0;
    for(int k=1;k<Ns;++k)if(Y[k]>Y[dependent])dependent=k;
    Real sumCorrected=0,energy=0;
    for(int k=0;k<Ns;++k){corrected[k]=raw[k]-Y[k]*sumRaw;if(k!=dependent)sumCorrected+=corrected[k];}

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
