#pragma once
#include "MieDoRadiationSolver.H"
namespace Foam { namespace gpuThermal {
inline MieDoControls readRadiationControls(const dictionary& d){
    return {d.lookup<label>("nTheta"),d.lookup<label>("nPhi"),d.lookup<label>("maxSourceIterations"),
        d.lookup<scalar>("sourceRelativeTolerance"),d.lookup<scalar>("phaseBalanceTolerance"),
        d.lookup<label>("maxPhaseBalanceIterations"),d.lookup<scalar>("radiationActiveEpsilonMin"),
        d.lookup<scalar>("conservationRelativeTolerance"),d.lookup<scalar>("maxPhaseScalingCorrection"),
        d.lookupOrDefault<label>("angleParallelThreads",1)};
}
}}
