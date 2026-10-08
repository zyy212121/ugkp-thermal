
#include <cmath>
#include <algorithm>
#include <iostream>
#include "GpuSstAlgebra.cuh"
#include "OpenFoamWallFunctions.cuh"
#define __device__
#define __global__
#define GPU_OPERATOR_TIME GpuReal
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_R GPU_R
using R=GpuReal;
struct Index{int x=0;};Index blockIdx,threadIdx;struct Block{int x=1;}blockDim;
const R OfVSmall=R(1e-30),OfSmall=R(1e-15);
R clampMin(R a,R b){return std::max(a,b);}R finiteOr(R a,R b){return std::isfinite(a)?a:b;}
R clampRange(R x,R lo,R hi){return std::max(lo,std::min(x,hi));}
struct DeviceState{
 int faceOwner[2]={0,0},facePeriodicPair[2]={1,0};
 R gasPhiRho[2]={0,0},deltaCoeffs[2]={1000,1000},faceWeight[2]={.5,.5};
 R sstPhiRhoK[2]={.1,.2},sstPhiRhoOmega[2]={3,7},V[1]={1},sstSourceNumber[1]={0},sstF1[1]={1},sstF2[1]={1};
 R gradKX[1]={0},gradKY[1]={0},gradKZ[1]={0},gradOmegaX[1]={0},gradOmegaY[1]={0},gradOmegaZ[1]={0};
 int nCells=1,nFaces=2,nInternalFaces=0,sstConfigured=1,sstWallTreatment=0;
 int riemannBoundaryKind[2]={2,2},riemannBoundaryUFix[2]={1,1},cellPlaneStart[1]={0},cellPlaneCount[1]={2},cellFaceId[2]={0,1};
 int sstBoundaryOmegaMode[2]={0,0},sstBoundaryKMode[2]={0,0};
 R rhoNext[1]={1},rhoUxNext[1]={0},rhoUyNext[1]={0},rhoUzNext[1]={0},rhoENext[1]={1};
 R rhoUx[1]={0},rhoUy[1]={0},rhoUz[1]={0},rhoE[1]={1},rhoKInitial[1]={0},rhoOmegaInitial[1]={0};
 R rho[1]={1},rhoMin=1e-12,gasMu=1.8e-5,wallRho[2]={4,8};
 R Ux[1]={2},Uy[1]={0},Uz[1]={0},riemannBoundaryUx[2]={0,0},riemannBoundaryUy[2]={0,0},riemannBoundaryUz[2]={0,0};
 R Sfx[2]={1,1},Sfy[2]={0,0},Sfz[2]={0,0};
 R k[1]={.1},omega[1]={7},rhoOmega[1]={7},rhoK[1]={.1},sstWallDistance[1]={.001};
 R sstWallCmu=.09,sstWallKappa=.41,sstWallE=9.8,sstOmegaMin=1e-8,sstKMin=1e-10;
 R sstBoundaryOmega[2]={42,42},sstBoundaryK[2]={.2,.2};
 ugkwp::SstCoefficients sstCoefficients=ugkwp::defaultSstCoefficients();
};
struct Prim{R rho;};Prim riemannFacePrimitiveForGradient(const DeviceState&s,int,int f){return {s.wallRho[f]};}
bool isPeriodicFace(const DeviceState&,int){return false;}
int coupledFaceNeighbour(const DeviceState&,int){return -1;}
