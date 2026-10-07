#ifndef UGKWP_CHEMISTRY_TYPES_H
#define UGKWP_CHEMISTRY_TYPES_H
#include "../gasTransport/MixtureThermo.H"
namespace ugkwp {
enum class ChemistryCode : unsigned char {
    Success, InvalidModel, InvalidComposition, InvalidControls, NonFiniteInput,
    TemperatureFailure, RateFailure, SingularJacobian, NewtonFailure,
    StepUnderflow, StepLimit, ConservationFailure
};
// Failed attempted-work diagnostics travel in the return value, never the
// accepted result/audit. Neither a failure nor a rejected substep publishes state.
struct ChemistryStatus {
    ChemistryCode code = ChemistryCode::Success;
    ThermoStatus thermoCode = ThermoStatus::Success;
    int reaction = -1;
    int attemptedSteps = 0;
    int rejectedSteps = 0;
    int nonlinearIterations = 0;
    UGKWP_GAS_HD explicit operator bool() const noexcept { return code==ChemistryCode::Success; }
};
template<class Real> struct ChemistryControls {
    Real relativeTolerance = Real(1e-6);
    Real absoluteMassFractionTolerance = Real(1e-12);
    Real absoluteTemperatureTolerance = Real(1e-4); // K
    Real initialStep = Real(0); // seconds; zero selects the complete interval
    Real minimumStep = Real(1e-20);
    Real maximumStep = Real(0); // zero means interval
    Real newtonTolerance = Real(0.03); // scaled local-error norm
    int maximumSteps = 10000;
    int maximumRejectedSteps = 100;
    int maximumNewtonIterations = 16;
    int maximumLineSearchIterations = 24;
    Real absoluteMassTolerance = Real(1e-14); // kg
    Real relativeConservationTolerance = Real(1e-10);
    Real absoluteElementTolerance = Real(1e-12); // mol atoms, NOT kg
    ThermoInversionControls<Real> thermo;
};
template<class Real,int Ns> struct ChemistryAudit {
    Real speciesMassChange[Ns] = {};
    Real massResidual = Real(0); // kg
    Real maximumElementResidual = Real(0); // mol atoms
    Real energyResidual = Real(0); // J
    Real maximumLocalError = Real(0);
    Real integratedTime = Real(0);
    int acceptedSteps = 0;
    int rejectedSteps = 0;
    int nonlinearIterations = 0;
};
template<class R> UGKWP_GAS_HD inline bool validateChemistryControls(const ChemistryControls<R>& c) noexcept {
 return gasFinite(c.relativeTolerance) && c.relativeTolerance>R(0)
 && gasFinite(c.absoluteMassFractionTolerance) && c.absoluteMassFractionTolerance>R(0)
 && gasFinite(c.absoluteTemperatureTolerance) && c.absoluteTemperatureTolerance>R(0)
 && gasFinite(c.initialStep) && c.initialStep>=R(0) && gasFinite(c.maximumStep) && c.maximumStep>=R(0)
 && gasFinite(c.minimumStep) && c.minimumStep>R(0) && gasFinite(c.newtonTolerance) && c.newtonTolerance>R(0) && c.newtonTolerance<R(1)
 && c.maximumSteps>0 && c.maximumRejectedSteps>=0 && c.maximumNewtonIterations>0 && c.maximumLineSearchIterations>0
 && gasFinite(c.absoluteMassTolerance) && c.absoluteMassTolerance>=R(0)
 && gasFinite(c.absoluteElementTolerance) && c.absoluteElementTolerance>=R(0)
 && gasFinite(c.relativeConservationTolerance) && c.relativeConservationTolerance>=R(0)
 && (c.maximumStep==R(0) || c.maximumStep>=c.minimumStep)
 && c.thermo.maximumIterations>0
 && gasFinite(c.thermo.minTemperature) && c.thermo.minTemperature>=R(0)
 && gasFinite(c.thermo.maxTemperature) && c.thermo.maxTemperature>=R(0)
 && (c.thermo.maxTemperature==R(0) || c.thermo.maxTemperature>c.thermo.minTemperature)
 && gasFinite(c.thermo.absoluteEnergyTolerance) && c.thermo.absoluteEnergyTolerance>=R(0)
 && gasFinite(c.thermo.relativeEnergyTolerance) && c.thermo.relativeEnergyTolerance>=R(0)
 && gasFinite(c.thermo.absoluteTemperatureTolerance) && c.thermo.absoluteTemperatureTolerance>=R(0)
 && gasFinite(c.thermo.relativeTemperatureTolerance) && c.thermo.relativeTemperatureTolerance>=R(0);
}

}
#endif
