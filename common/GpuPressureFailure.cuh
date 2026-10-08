#pragma once
enum PressureFailureCode : unsigned int
{
    pressureBadInitial=1, pressureBadFace=2, pressureBadFinal=4,
    pressureBadParticle=8, pressureUnrealizableParticles=16
};
__device__ inline void recordPressureFailure(DeviceState& s,unsigned int reason,int c)
{
    atomicOr(s.pressureFailure,reason);
    if(atomicCAS(s.pressureFailure+1,0u,static_cast<unsigned int>(c+1))==0u)
        s.pressureFailure[2]=reason;
}
