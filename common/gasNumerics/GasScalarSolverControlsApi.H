#pragma once
#include "GasScalarSolverControls.H"



extern "C" int ugkwpGpuResidentStrictConfigureGasScalarSolvers(
    void* handle, const ugkp::gasNumerics::ScalarSolverControls<double>* controls);
