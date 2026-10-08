
#pragma once
#include "GpuPressureUnsortedAlgebra.cuh"
// One pressure traversal; model adapters own only constraints and moment closure.
// Directory modes describe storage, never application identity.
enum class PressureDirectory { Full, Split, Base, Injection };
struct PressureParameters
{
    PressureReal ux0,uy0,uz0,ux1,uy1,uz1,theta1,thermalScale,thetaScale;
    PressureReal px1,py1,pz1,e1;
    bool active,resolved;
};
__device__ __forceinline__ void pressureDeltaFromLimitedFaces
(DeviceState& s,const int c,const PressureTime dt,PressureReal (&delta)[4])
{
    // Keep accumulation local; publish the caller's array only after recovery.
    PressureReal dpx,dpy,dpz,de;
    accumulatePressureFaceDelta(s,c,dt,dpx,dpy,dpz,de);
    delta[0]=dpx; delta[1]=dpy; delta[2]=dpz; delta[3]=de;
    s.pressureDeltaMomX[c]=delta[0]; s.pressureDeltaMomY[c]=delta[1];
    s.pressureDeltaMomZ[c]=delta[2]; s.pressureDeltaEnergy[c]=delta[3];
}
template<class ParameterStorage>
__device__ __forceinline__ void makePressureParameters
(const DeviceState& s,const int c,const PressureReal (&delta)[4],ParameterStorage& q)
{
    const PressureReal rhoP=clampMin(finiteOr(s.momRhoP[c],PressureReal(0)),PressureReal(0));
    q.active=!(rhoP<=s.epsSMin*s.rhoSolid);
    if(q.active)
    {
    const PressureReal px0=finiteOr(s.momRhoUPx[c],PressureReal(0));
    const PressureReal py0=finiteOr(s.momRhoUPy[c],PressureReal(0));
    const PressureReal pz0=finiteOr(s.momRhoUPz[c],PressureReal(0));
    const PressureReal e0=clampMin(finiteOr(s.momRhoEP[c],PressureReal(0)),PressureReal(0));
    const PressureReal px1=px0+delta[0],py1=py0+delta[1],pz1=pz0+delta[2],e1=e0+delta[3];
    const PressureReal ux0=px0/rhoP,uy0=py0/rhoP,uz0=pz0/rhoP;
    const PressureReal ux1=px1/rhoP,uy1=py1/rhoP,uz1=pz1/rhoP;
    const PressureReal theta0=clampMin(pressureKickInternalEnergy(rhoP,px0,py0,pz0,e0)/(PressureReal(1.5)*rhoP),PressureReal(0));
    const PressureReal theta1=clampMin(pressureKickInternalEnergy(rhoP,px1,py1,pz1,e1)/(PressureReal(1.5)*rhoP),PressureReal(0));
    const bool resolved=theta0>PressureReal(10)*s.thetaMin;
    const PressureReal thermalScale=resolved?sqrt(clampMin(theta1/theta0,PressureReal(0))):PressureReal(0);
    const PressureReal thetaScale=resolved?thermalScale*thermalScale:PressureReal(0);
    q.px1=px1;q.py1=py1;q.pz1=pz1;q.e1=e1;
    q.ux0=ux0;q.uy0=uy0;q.uz0=uz0;q.ux1=ux1;q.uy1=uy1;q.uz1=uz1;
    q.theta1=theta1;q.resolved=resolved;q.thermalScale=thermalScale;q.thetaScale=thetaScale;
    }
}
template<class Operation,PressureDirectory Directory,bool CompactParticles>
__device__ __forceinline__ void runCellPressureProjection(DeviceState* sp,const PressureTime dt)
{
    DeviceState& s=*sp;const int c=blockIdx.x;
    if(c>=s.nCells)return;
    __shared__ typename Operation::Shared shared;
    PressureParameters q;
    Operation::template prepare<Directory>(s,c,dt,shared,q);
    if constexpr(!Operation::closeInactiveCells) { if(!q.active)return; }
    typename Operation::Accumulator accumulator{};
    if constexpr(Directory==PressureDirectory::Split || Directory==PressureDirectory::Base)
    {
        const int first=s.preBaseCellOffset[c],end=s.preBaseCellOffset[c+1];
        for(int i=first+threadIdx.x;i<end;i+=blockDim.x)
            Operation::template visit<true,CompactParticles>(s,c,i,q.active,q.ux0,q.uy0,q.uz0,q.ux1,q.uy1,q.uz1,
                q.theta1,q.thermalScale,q.thetaScale,q.resolved,accumulator);
    }
    if constexpr(Directory!=PressureDirectory::Base)
    {
        const int first=s.cellParticleOffset[c],end=s.cellParticleOffset[c+1];
        for(int pos=first+threadIdx.x;pos<end;pos+=blockDim.x)
        {
            const int i=CompactParticles?pos:s.sortedParticleIndex[pos];
            Operation::template visit<Directory!=PressureDirectory::Full,CompactParticles>(s,c,i,q.active,q.ux0,q.uy0,q.uz0,q.ux1,q.uy1,q.uz1,
                q.theta1,q.thermalScale,q.thetaScale,q.resolved,accumulator);
        }
    }
    Operation::template finish<Directory>(s,c,q,accumulator);
}
