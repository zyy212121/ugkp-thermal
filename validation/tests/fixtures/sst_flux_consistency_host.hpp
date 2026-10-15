#include <cmath>
#include <algorithm>
#include <iostream>
#include "GpuSstAlgebra.cuh"
#include "gasTransport/GasStateView.H"
#include "gasTransport/GasCapabilities.H"
#include "gasTransport/MixtureThermo.H"
#include "gasTransport/GasGeometryValidation.H"
#include "OpenFoamWallFunctions.cuh"
#include "OpenFoamViscousFlux.cuh"
#define __host__
#define __device__
#define __global__
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_TIME GpuReal
#define GPU_OPERATOR_R GPU_R
#define GPU_OPERATOR_TINY GPU_TINY
using R = GpuReal;
struct Idx {int x=0;}; Idx blockIdx,threadIdx; struct Block{int x=1;} blockDim;
const R OfSmall=R(1e-15), OfVSmall=R(1e-30), OfGreat=R(1e30);
R clampMin(R x,R lo){return std::max(x,lo);} R clampRange(R x,R lo,R hi){return std::max(lo,std::min(x,hi));}
R finiteOr(R x,R alt){return std::isfinite(x)?x:alt;} bool finiteDevice(R x){return std::isfinite(x);}
R sqr3(R x,R y,R z){return x*x+y*y+z*z;}
struct DeviceState {
 int nCells=2,nFaces=2,nInternalFaces=0,sstConfigured=1,sstWallTreatment=0;
 int turbulenceModel=3,gasFluxScheme=1;
 int faceOwner[2]={0,0},faceNeighbour[2]={-1,-1},facePeriodicPair[2]={1,0}; bool periodic=false;
 int cellPlaneStart[2]={0,2},cellPlaneCount[2]={2,0},cellFaceId[4]={0,1,0,1};
 int riemannBoundaryKind[2]={0,0},riemannBoundaryUFix[2]={1,1};
 int sstBoundaryKMode[2]={0,0},sstBoundaryOmegaMode[2]={0,0};
 R sstBoundaryK[2]={2,2},sstBoundaryOmega[2]={20,20};
 R rho[2]={2,8},Ux[2]={3,3},Uy[2]={0,0},Uz[2]={0,0},p[2]={20,20},Tgas[2]={1,1};
 R rhoUx[2]={6,24},rhoUy[2]={0,0},rhoUz[2]={0,0},rhoE[2]={100,100};
 R rhoMin=1e-8,gasMu=R(.002),Rgas=1,gammaGas=R(1.4),TgasMin=R(.001);
 R k[2]={R(.1),R(.2)},omega[2]={2,4},rhoK[2]={R(.2),R(1.6)},rhoOmega[2]={4,32};
 R nut[2]={R(.1),R(.2)},sstF1[2]={1,1},sstF2[2]={1,1},sstWallDistance[2]={R(.1),R(.1)};
 R sstKMin=1e-10,sstOmegaMin=1e-8,sstWallKappa=R(.41),sstWallE=R(9.8),sstWallCmu=R(.09);
 R sstPhiRhoK[2]={0,0},sstPhiRhoOmega[2]={0,0},sstSourceNumber[2]={0,0};
 R gasPhiRho[2]={0,0},gasPhiRhoE[2]={0,0},gasFluxPositivityScale[2]={0,0};
 R maxDiffusionNumber=1,sstMaxSourceNumber=1;
 R V[2]={2,2},Sfx[2]={1,-1},Sfy[2]={0,0},Sfz[2]={0,0},magSf[2]={1,1},deltaCoeffs[2]={10,10},faceWeight[2]={R(.25),R(.75)};
 R Cx[2]={0,1},Cy[2]={0,0},Cz[2]={0,0};
 R riemannBoundaryUx[2]={0,0},riemannBoundaryUy[2]={0,0},riemannBoundaryUz[2]={0,0},boundaryRho[2]={4,4};
 R gradKX[2]={0,0},gradKY[2]={0,0},gradKZ[2]={0,0},gradOmegaX[2]={0,0},gradOmegaY[2]={0,0},gradOmegaZ[2]={0,0};
 R gradUxX[2]={0,0},gradUxY[2]={0,0},gradUxZ[2]={0,0},gradUyX[2]={0,0},gradUyY[2]={0,0},gradUyZ[2]={0,0},gradUzX[2]={0,0},gradUzY[2]={0,0},gradUzZ[2]={0,0};
 R epsGPrev[2]={R(.8),R(.8)},epsSolid[2]={R(.1),R(.1)};
 ugkwp::SstCoefficients sstCoefficients=ugkwp::defaultSstCoefficients();
};
struct Prim{R rho;};
Prim riemannFacePrimitiveForGradient(const DeviceState&s,int,int f){return {s.boundaryRho[f]};}
bool isPeriodicFace(const DeviceState&s,int){return s.periodic;}
int coupledFaceNeighbour(const DeviceState&s,int f){return f<s.nInternalFaces||s.periodic?s.faceNeighbour[f]:-1;}
void periodicMappedCellCentre(const DeviceState&s,int,int c,R&x,R&y,R&z){x=s.Cx[c];y=s.Cy[c];z=s.Cz[c];}
R solidEpsFromMomentDevice(const DeviceState&s,int c){return s.epsSolid[c];}
