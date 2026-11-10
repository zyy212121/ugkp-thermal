#pragma once
#include "BoundaryInjectionSampling.H"
#include "ParticleResidentInterval.H"
#if defined(__CUDACC__)
#define UGKP_INJECTION_SEQUENCE_HD __host__ __device__
#else
#define UGKP_INJECTION_SEQUENCE_HD
#endif
namespace ugkwp {
// A configured sample sequence is independent of the frozen boundary clock.
// Unconfigured callers retain the existing physical-time position stream.
template<class State>
UGKP_INJECTION_SEQUENCE_HD inline unsigned long long particleInjectionPositionSeed
(const State& state,unsigned long long seed,int face,double time,int ordinal,
 unsigned long long sequence=0,bool sequenceEnabled=false) {
    if constexpr(HasParticleResidentInterval<State>::value) {
        const auto& interval=state.particleResidentInterval;
        if(interval.active) {
            sequence=interval.positionSequence;
            sequenceEnabled=interval.positionSequenceEnabled;
        }
    }
    if(!sequenceEnabled)return injectionPositionSeed(seed,face,time,ordinal);
    return injectionMix(seed ^ injectionMix(static_cast<unsigned long long>(face))
        ^ injectionMix(sequence ^ 0x94d049bb133111ebULL)
        ^ injectionMix(static_cast<unsigned long long>(ordinal) + 0xd1b54a32d192ed03ULL));
}
}
#undef UGKP_INJECTION_SEQUENCE_HD
