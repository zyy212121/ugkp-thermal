#ifndef UGKWP_CHEMISTRY_TYPES_H
#define UGKWP_CHEMISTRY_TYPES_H
#include "../gasTransport/MixtureThermo.H"
namespace ugkwp {
enum class ChemistryCode : unsigned char {
    Success, InvalidModel, InvalidComposition, InvalidControls, NonFiniteInput,
    TemperatureFailure, RateFailure, SingularJacobian, NewtonFailure,
    StepUnderflow, StepLimit, ConservationFailure
};


struct ChemistryStatus {
    ChemistryCode code = ChemistryCode::Success;
    ThermoStatus thermoCode = ThermoStatus::Success;
    int reaction = -1;
    int attemptedSteps = 0;
    int rejectedSteps = 0;
    double observed = 0;
    double reference = 0;
    int nonlinearIteration = 0;
    UGKWP_GAS_HD explicit operator bool() const noexcept { return code==ChemistryCode::Success; }
};
template<class Real> struct ChemistryControls {
    Real relativeTolerance = Real(1e-6);
    Real absoluteMassFractionTolerance = Real(1e-12);
    Real absoluteTemperatureTolerance = Real(1e-4); // K
    Real initialStep = Real(0); // Legacy serialized field; single-pass advancement ignores it.
    Real minimumStep = Real(1e-20);
    Real maximumStep = Real(0); // zero means interval
    Real newtonTolerance = Real(0.03); // scaled local-error norm
    int maximumSteps = 10000;
    int maximumRejectedSteps = 100;
    int maximumNewtonIterations = 16;
    int maximumLineSearchIterations = 24; // Legacy wire/checkpoint slot, no line search.
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
// Local scratch may use energy precision while immutable tables and published
// species/audits retain their independently declared storage precision.
template<class To,class From>
UGKWP_GAS_HD inline ChemistryControls<To> castChemistryControls
(const ChemistryControls<From>& input) noexcept
{
    ChemistryControls<To> out;
    out.relativeTolerance=input.relativeTolerance;
    out.absoluteMassFractionTolerance=input.absoluteMassFractionTolerance;
    out.absoluteTemperatureTolerance=input.absoluteTemperatureTolerance;
    out.initialStep=input.initialStep;
    out.minimumStep=input.minimumStep;
    out.maximumStep=input.maximumStep;
    out.newtonTolerance=input.newtonTolerance;
    out.maximumSteps=input.maximumSteps;
    out.maximumRejectedSteps=input.maximumRejectedSteps;
    out.maximumNewtonIterations=input.maximumNewtonIterations;
    out.maximumLineSearchIterations=input.maximumLineSearchIterations;
    out.absoluteMassTolerance=input.absoluteMassTolerance;
    out.relativeConservationTolerance=input.relativeConservationTolerance;
    out.absoluteElementTolerance=input.absoluteElementTolerance;
    out.thermo=castThermoControls<To>(input.thermo);
    return out;
}


template<class R> UGKWP_GAS_HD inline bool validateChemistryControls(const ChemistryControls<R>& c) noexcept {


 (void)c;return true;
}

}
#endif
