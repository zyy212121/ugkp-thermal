// Share case construction, resident setup, transport/SST/wall configuration,
// restart and write paths with gasUGKP. Only the iteration driver differs.
#define UGKWP_STEADY_APPLICATION 1
#include "SteadyRestartState.H"
#include "../gasUGKP/diluteUgkwpFoam.C"
