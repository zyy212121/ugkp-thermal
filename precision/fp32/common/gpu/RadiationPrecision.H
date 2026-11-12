#ifndef UGKP_GPU_RADIATION_PRECISION_H
#define UGKP_GPU_RADIATION_PRECISION_H
#include "../GpuPrecisionTypes.H"
#include <limits>
#ifdef __CUDACC__
#define UGKP_RADIATION_PRECISION_HD __host__ __device__
#else
#define UGKP_RADIATION_PRECISION_HD
#endif
namespace ugkp { namespace gpu {
// One physical scalar per field. Host OpenFOAM interfaces convert at entry/exit.
using RadiationReal = GpuReal;
using RadiationResidual = RadiationReal;
using RadiationEnergy = RadiationReal;
inline constexpr RadiationReal radiationEpsilon=std::numeric_limits<RadiationReal>::epsilon();
// Preset precision floors, not residual substitutes. The FP64 request is
// unchanged. FP32 compares the measured residual against its representable
// arithmetic scale; relative-to-initial stopping remains separately configured.
UGKP_RADIATION_PRECISION_HD inline RadiationReal radiationConvergenceTolerance(
    RadiationReal requested,RadiationReal epsilonUnits=32){
    if constexpr(sizeof(RadiationReal)<sizeof(double)){
        const RadiationReal floor=epsilonUnits*radiationEpsilon;
        return requested<floor?floor:requested;
    }
    return requested;
}
UGKP_RADIATION_PRECISION_HD inline RadiationReal radiationPhaseTolerance(RadiationReal requested,int directions){
    const RadiationReal accumulated=RadiationReal(8)*RadiationReal(directions);
    return radiationConvergenceTolerance(requested,accumulated>32?accumulated:RadiationReal(32));
}
}}
#undef UGKP_RADIATION_PRECISION_HD
#endif
