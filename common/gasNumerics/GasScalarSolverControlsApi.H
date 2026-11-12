#pragma once
#include "GasScalarSolverControls.H"

// Host ABI uses binary64 storage; the backend casts once into its native scalar
// state. The frontend derives defaults in the compiled device precision.
extern "C" int ugkwpGpuResidentStrictConfigureGasScalarSolvers(
    void* handle, const ugkp::gasNumerics::ScalarSolverControls<double>* controls);
