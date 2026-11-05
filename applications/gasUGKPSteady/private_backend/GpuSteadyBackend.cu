// The steady driver and transient gasUGKP use the same physical operators.
// Only pseudo-time and statistical sampling belong to this application.
#define UGKWP_STEADY_APPLICATION 1
#include "../../gasUGKP/private_backend/GpuResidentStrict.cu"
