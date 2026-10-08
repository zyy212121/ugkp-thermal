#pragma once
#include "GpuPressureUnsortedAlgebra.cuh"
#include "GpuPressureConvexAlgebra.cuh"
#include "GpuPressureFailure.cuh"
// Each face is one fixed-weight convex substate (weight=1/faceCount).
// A neighbouring cell may only shorten that substate's safe segment.
__device__ inline PressureReal pressureLocalConvexScale(DeviceState& s,int c,PressureTime dt)
{
    const PressureReal rho=s.momRhoP[c],px=s.momRhoUPx[c],py=s.momRhoUPy[c],pz=s.momRhoUPz[c],e=s.momRhoEP[c];
    if(!finiteDevice(rho)||rho<0||!finiteDevice(px)||!finiteDevice(py)||!finiteDevice(pz)||!finiteDevice(e)||e<0
       ||!finiteDevice(s.V[c])||s.V[c]<=0||!finiteDevice(s.cellLength[c])||s.cellLength[c]<=0
       ||!finiteDevice(s.thetaMin)||s.thetaMin<0||!finiteDevice(s.pressureKickFraction)||s.pressureKickFraction<0
       ||!finiteDevice(s.epsSMin)||s.epsSMin<0||!finiteDevice(s.rhoSolid)||s.rhoSolid<=0)
    {recordPressureFailure(s,pressureBadInitial,c);return 0;}
    if(rho==0)
    {
        if(px!=0||py!=0||pz!=0||e!=0)recordPressureFailure(s,pressureBadInitial,c);
        return 0;
    }
    const PressureReal configuredFloor=PressureReal(1.5)*rho*s.thetaMin;
    const auto initial=pressure_convex::initialState(rho,px,py,pz,e,configuredFloor);
    if(!initial.valid){recordPressureFailure(s,pressureBadInitial,c);return 0;}
    if(rho<=s.epsSMin*s.rhoSolid)return 0;
    const int count=s.cellPlaneCount[c],start=s.cellPlaneStart[c];
    if(count<0||start<0){recordPressureFailure(s,pressureBadFace,c);return 0;}
    const PressureReal maxDU=s.pressureKickFraction*s.cellLength[c]/dt;
    const PressureReal factor=dt/s.V[c];
    if(!finiteDevice(maxDU)||!finiteDevice(factor)){recordPressureFailure(s,pressureBadInitial,c);return 0;}
    PressureReal lambda=1;
    for(int j=0;j<count;++j)
    {
        const int f=s.cellFaceId[start+j];
        if(f<0||f>=s.nFaces||(s.faceOwner[f]!=c&&s.faceNeighbour[f]!=c))
        {recordPressureFailure(s,pressureBadFace,c);return 0;}
        const PressureReal sign=s.faceOwner[f]==c?PressureReal(1):PressureReal(-1);
        const PressureReal weightInverse=PressureReal(count);
        const PressureReal dx=-factor*sign*s.solidPressurePhiMomX[f]*weightInverse;
        const PressureReal dy=-factor*sign*s.solidPressurePhiMomY[f]*weightInverse;
        const PressureReal dz=-factor*sign*s.solidPressurePhiMomZ[f]*weightInverse;
        const PressureReal de=-factor*sign*s.solidPressurePhiEnergy[f]*weightInverse;
        const auto face=pressure_convex::limitFace(rho,px,py,pz,e,configuredFloor,maxDU,dx,dy,dz,de);
        if(!face.valid){recordPressureFailure(s,pressureBadFace,c);return 0;}
        lambda=fmin(lambda,face.beta);
    }
    return lambda;
}
template<class Scratch>
__device__ __forceinline__ void runPressureKickAccumulation
(DeviceState* sp,const PressureTime kickDt,const int computeScale,const int initialiseMoments)
{
    DeviceState& s=*sp;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=s.nCells)return;
    if(initialiseMoments!=0)Scratch::initialise(s,c);
    if(computeScale!=0)
    {
        s.pressureKickScale[c]=pressureLocalConvexScale(s,c,kickDt);
        return;
    }
    PressureReal dx,dy,dz,de;
    accumulatePressureFaceDelta(s,c,kickDt,dx,dy,dz,de);
    s.pressureDeltaMomX[c]=dx;s.pressureDeltaMomY[c]=dy;
    s.pressureDeltaMomZ[c]=dz;s.pressureDeltaEnergy[c]=de;
}
