#pragma once
#include "MieDoRadiationSolver.H"
namespace Foam { namespace gpuThermal {
inline MieDoControls readRadiationControls(const dictionary& d){
    MieDoControls controls;
    // Angular quadrature remains a model input.
    // Iteration budgets, residuals and relaxation are resolved from scheme by
    // the solver owner; old model-dictionary numerical keys are not consumed.
    controls.nTheta=d.lookup<label>("nTheta");
    controls.nPhi=d.lookup<label>("nPhi");
    return controls;
}
}}
