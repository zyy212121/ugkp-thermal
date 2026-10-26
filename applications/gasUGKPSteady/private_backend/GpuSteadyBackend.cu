// Compile the resident implementation independently; only this binary owns the
// new steady and frozen-mixture/particle adapter. Normal gasUGKP keeps its gate.
#define UGKWP_STEADY_APPLICATION 1
#include "SteadyMixturePorosity.cuh"
#define GPU_GAS_POROSITY_SPECIES(s,c,scale) steadyScalePorositySpecies(s,c,scale)
#include "../../gasUGKP/private_backend/GpuResidentStrict.cu"
