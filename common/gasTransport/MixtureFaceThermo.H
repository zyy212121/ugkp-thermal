#pragma once
#include "MixtureThermo.H"

namespace ugkwp
{




template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline ThermoStatus evaluateMixtureFaceEnergy
(
    const Real* energyWeights,const Real* capacityWeights,Real temperature,
    const SpeciesThermoView<TableReal,Ns>& view,Real& energy,Real& heatCapacity,
    const bool nativeRefinement=false,bool* evaluated=nullptr
) noexcept
{
    if(evaluated)*evaluated=false;
    if(!energyWeights || !capacityWeights)return ThermoStatus::InvalidComposition;
    Real localEnergy=0,localCapacity=0;
    for(int k=0;k<Ns;++k)
    {
        if(!nativeRefinement && !gasFinite(energyWeights[k]))
            return gasThermoDetail::failure(view.diagnostics,ThermoStatus::InvalidComposition,
                ugkp::diagnostics::Code::NonFinite,k,energyWeights[k]);
        gasThermoDetail::SpeciesValues<Real> values;
        const auto status=gasThermoDetail::evaluateSpecies<Real,Ns,TableReal,false,true>
            (k,temperature,view,values);
        if(status!=ThermoStatus::Success)
        {
            if(!nativeRefinement)return status;


            values.e=gasThermoDetail::invalidValue<Real>();
            values.cv=speciesCv(k,temperature,view);
        }
        else if(nativeRefinement && !gasFinite(values.e))
        {
            gasThermoDetail::failure(view.diagnostics,ThermoStatus::NonFiniteInput,
                ugkp::diagnostics::Code::NonFinite,k,values.e);
            values.e=gasThermoDetail::invalidValue<Real>();
        }
        localEnergy+=energyWeights[k]*values.e;
        if(!nativeRefinement && !gasFinite(capacityWeights[k]))
            return gasThermoDetail::failure(view.diagnostics,ThermoStatus::NonFiniteInput,
                ugkp::diagnostics::Code::NonFinite,k,capacityWeights[k]);
        localCapacity+=capacityWeights[k]*values.cv;
    }
    if(!nativeRefinement && (!gasFinite(localEnergy)||!gasFinite(localCapacity)))
        return gasThermoDetail::failure(view.diagnostics,ThermoStatus::NonFiniteInput,
            ugkp::diagnostics::Code::NonFinite,-1,!gasFinite(localEnergy)?localEnergy:localCapacity);
    energy=gasFinite(localEnergy)?localEnergy:gasThermoDetail::invalidValue<Real>();
    heatCapacity=gasFinite(localCapacity)?localCapacity:gasThermoDetail::invalidValue<Real>();
    if(evaluated)*evaluated=true;
    return ThermoStatus::Success;
}
}
