#include <initializer_list>
#include <cmath>
#include <cstdio>
#include "GpuPressureConvexAlgebra.cuh"
#define __device__
#include "GpuPrecisionTypes.H"
using PressureReal=GpuReal;using PressureTime=GpuTime;
template<class T>bool finiteDevice(T x){return x==x&&std::abs(x)<(sizeof(T)==8?1e300:1e30);}
struct DeviceState{PressureReal momRhoP[1]={1},momRhoUPx[1]={},momRhoUPy[1]={},momRhoUPz[1]={},momRhoEP[1]={1},V[1]={1},cellLength[1]={1},thetaMin=.1,pressureKickFraction=.2,epsSMin=1e-12,rhoSolid=3;
int nFaces=1,cellPlaneCount[1]={1},cellPlaneStart[1]={0},cellFaceId[1]={0},faceOwner[1]={0},faceNeighbour[1]={-1};PressureReal solidPressurePhiMomX[1]={1},solidPressurePhiMomY[1]={},solidPressurePhiMomZ[1]={},solidPressurePhiEnergy[1]={0};};
