#ifndef UGKWP_MIXTURE_HEAT_EXCHANGE_H
#define UGKWP_MIXTURE_HEAT_EXCHANGE_H
#include "MixtureThermo.H"

namespace ugkwp
{




template<class Real,int Ns,class TableReal>
struct MixtureHeatExchangeEvaluator
{
    MixtureEnergyEvaluator<Real,Ns,TableReal,false> mixture;
    Real oldEnergy,linearCapacity,referenceTemperature,lower,upper;
    UGKWP_GAS_HD ThermoStatus operator()(Real T,Real& energy,Real& capacity) const noexcept
    {
        if(!gasFinite(T))return ThermoStatus::NonFiniteInput;
        if(T<lower || T>upper)return ThermoStatus::TemperatureOutOfRange;
        const auto status=mixture(T,energy,capacity);
        if(status!=ThermoStatus::Success)return status;
        energy=(energy-oldEnergy)+linearCapacity*(T-referenceTemperature);
        capacity+=linearCapacity;
        return ThermoStatus::Success;
    }
};



template<class Real,class Evaluator>
UGKWP_GAS_HD inline ThermoStatus invertAcceptedHeatExchange
(
    const Evaluator& evaluator,Real heatTarget,Real oldEnergy,Real lo,Real hi,
    const ThermoInversionControls<Real>& controls,Real& temperature,
    ugkp::diagnostics::DeviceContext* diagnostics
) noexcept
{
    using namespace gasThermoDetail;


    auto solveControls=controls;
    const Real energyRoundoff=Real(4)*std::numeric_limits<Real>::epsilon()
        *(absValue(oldEnergy)+absValue(heatTarget));
    solveControls.absoluteEnergyTolerance+=energyRoundoff;




    return invertCaloricEnergy(evaluator,heatTarget,lo,hi,solveControls,
                              temperature,nullptr,diagnostics);
}
template<class Real,int Ns,class TableReal>
UGKWP_GAS_HD inline ThermoStatus invertMixtureHeatExchange
(
    const Real* speciesMass,Real oldEnergy,Real linearCapacity,Real heatTarget,
    const SpeciesThermoView<TableReal,Ns>& view,
    const ThermoInversionControls<Real>& controls,Real& temperature
) noexcept
{
    using namespace gasThermoDetail;
    if(!view.species || !speciesMass)return ThermoStatus::InvalidModel;
    Real lo,hi;mixtureTemperatureBounds(view,controls,lo,hi);
    if(!(lo<hi))return ThermoStatus::InvalidControls;
    const MixtureHeatExchangeEvaluator<Real,Ns,TableReal> evaluator
        {{speciesMass,view},oldEnergy,linearCapacity,temperature,lo,hi};
    return invertAcceptedHeatExchange(evaluator,heatTarget,oldEnergy,lo,hi,
        controls,temperature,view.diagnostics);
}





template<class Real,int Ns,class TableReal,class Response>
struct MixtureParticleHeatExchangeEvaluator
{
    MixtureEnergyEvaluator<Real,Ns,TableReal,false> mixture;
    const Response& response;
    Real oldEnergy,lower,upper;
    UGKWP_GAS_HD ThermoStatus operator()(Real T,Real& energy,Real& capacity) const noexcept
    {
        if(!gasFinite(T))return ThermoStatus::NonFiniteInput;
        if(T<lower || T>upper)return ThermoStatus::TemperatureOutOfRange;
        const auto status=mixture(T,energy,capacity);
        if(status!=ThermoStatus::Success)return status;
        Real heat,derivative;
        const auto particleStatus=response(T,heat,derivative);
        if(particleStatus!=ThermoStatus::Success)return particleStatus;
        energy=(energy-oldEnergy)+heat;
        capacity+=derivative;
        return ThermoStatus::Success;
    }
};
template<class Real,int Ns,class TableReal,class Response>
UGKWP_GAS_HD inline ThermoStatus invertMixtureParticleHeatExchange
(
    const Real* speciesMass,Real oldEnergy,const Response& response,
    const SpeciesThermoView<TableReal,Ns>& view,
    const ThermoInversionControls<Real>& controls,Real& temperature
) noexcept
{
    using namespace gasThermoDetail;
    if(!view.species || !speciesMass)return ThermoStatus::InvalidModel;
    Real lo,hi;mixtureTemperatureBounds(view,controls,lo,hi);
    if(!(lo<hi))return ThermoStatus::InvalidControls;
    Real referenceHeat,derivative;
    const auto status=response(temperature,referenceHeat,derivative);
    if(status!=ThermoStatus::Success)return status;





    const Response endpointResponse=response;
    struct CenteredResponse {
        const Response& source;
        const Response& endpoints;
        Real reference,lower,upper;
        UGKWP_GAS_HD ThermoStatus operator()(Real T,Real& q,Real& dq)const noexcept {
            const auto result=(T==lower||T==upper)
                ?endpoints(T,q,dq):source(T,q,dq);
            if(result==ThermoStatus::Success)q-=reference;
            return result;
        }
    } centered{response,endpointResponse,referenceHeat,lo,hi};
    const MixtureParticleHeatExchangeEvaluator<Real,Ns,TableReal,CenteredResponse>
        evaluator{{speciesMass,view},centered,oldEnergy,lo,hi};
    return invertAcceptedHeatExchange(evaluator,-referenceHeat,oldEnergy,lo,hi,
        controls,temperature,view.diagnostics);
}
}
#endif
