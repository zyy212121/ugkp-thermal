#pragma once
#include "MieDoRadiationSolver.H"
namespace Foam { namespace gpuThermal {
inline MieDoControls readRadiationControls(const dictionary& d){
    MieDoControls controls;



    controls.nTheta=d.lookup<label>("nTheta");
    controls.nPhi=d.lookup<label>("nPhi");
    return controls;
}
}}
