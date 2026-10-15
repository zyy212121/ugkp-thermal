
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#define __forceinline__ inline
#define __device__
#include "GpuPrecisionTypes.H"
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_TIME GpuTime
#define GPU_OPERATOR_R(x) GPU_R(x)
#define GPU_OPERATOR_TINY(x) GPU_TINY(x)
#define GPU_OPERATOR_THERMAL 0
#define GPU_PERIODIC_FACE_KIND 6
#define GPU_PARTICLE_EXIT_MASS_FALLBACK s.injectionParcelMass
#define GPU_PERIODIC_COORDINATE(hit,shift,eps,normal) ((hit)-(shift)+(eps)*(normal))
#define GPU_WALL_COORDINATE(hit,eps,normal) ((hit)-(eps)*(normal))
template<class T>T clampMin(T a,T b){return std::max(a,b);}
template<class T>T clampRange(T a,T b,T c){return std::min(std::max(a,b),c);}
template<class T>T finiteOr(T a,T b){return std::isfinite(a)?a:b;}
struct DeviceState {
 int particleCapacity=1,nCells=1,maxFaceWalkHops=512;
 int pStatus[1]={1},pCellId[1]={0},cellPlaneStart[1]={0},cellPlaneCount[1]={2};
 int cellFaceKind[4]={1,1,1,1},cellFaceNeighbor[4]={-1,-1,-1,-1},cellFaceId[4]={0,1,2,3};
 GpuReal px[1]={},py[1]={},pz[1]={},pux[1]={},puy[1]={},puz[1]={},puxOld[1]={},puyOld[1]={},puzOld[1]={},pm[1]={GPU_R(2)};
 GpuReal planeNx[4]={},planeNy[4]={},planeNz[4]={},planeD[4]={},cellLength[1]={GPU_R(.001)};
 GpuReal cellFaceRestitution[4]={GPU_R(1),GPU_R(1),GPU_R(1),GPU_R(1)},faceCx[4]={},faceCy[4]={},faceCz[4]={};
 GpuReal facePeriodicDx[4]={},facePeriodicDy[4]={},facePeriodicDz[4]={},injectionParcelMass=GPU_R(2);
};
bool isPeriodicFace(const DeviceState& s,int f){return s.cellFaceKind[f]==6;}

#include "operators/pointInsideCell.cuh"
#include "GpuParticleTransport.cuh"
int main(){for(int budget:{32,512})for(double dt:{.125,3.5,300.5}){DeviceState s;s.maxFaceWalkHops=budget;s.cellPlaneCount[0]=2;s.planeNx[0]=1;s.planeNx[1]=-1;s.planeD[0]=1;s.planeD[1]=0;s.faceCx[0]=1;s.faceCx[1]=0;s.px[0]=.25;s.pux[0]=s.puxOld[0]=1;trackOneParticleLocalFaceWalk(s,0,dt);printf("budget=%d dt=%a status=%d cell=%d x=%a y=%a z=%a vx=%a vy=%a vz=%a oldvx=%a oldvy=%a oldvz=%a mass=%a\n",budget,dt,s.pStatus[0],s.pCellId[0],double(s.px[0]),double(s.py[0]),double(s.pz[0]),double(s.pux[0]),double(s.puy[0]),double(s.puz[0]),double(s.puxOld[0]),double(s.puyOld[0]),double(s.puzOld[0]),double(s.pm[0]));}}
