#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "GpuPrecisionTypes.H"
#include "GpuAluminaProperties.H"
#define __device__
#define __global__
#define __forceinline__ inline
#define GPU_RECOVERY_INLINE inline
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_TIME GpuTime
#define GPU_OPERATOR_R GPU_R
#define GPU_OPERATOR_TINY GPU_TINY
#define GPU_OPERATOR_THERMAL 1
constexpr GpuReal OfSmall=GPU_R(2.22044604925031308085e-16);
template<class T>bool finiteDevice(T x){return x==x&&std::abs(x)<(sizeof(T)==8?1e300:1e30);}
template<class T>T finiteOr(T x,T y){return finiteDevice(x)?x:y;}
template<class T>T clampMin(T x,T y){return x<y?y:x;}
template<class T>T clampRange(T x,T lo,T hi){return x<lo?lo:(x>hi?hi:x);}
template<class T>T sqr3(T x,T y,T z){return x*x+y*y+z*z;}
struct Dim{int x=0;};Dim blockIdx,threadIdx;struct Block{int x=1;}blockDim;
struct DeviceState{
 int particleGasHeatTransferModelId=1;
 GpuReal couplingRhoOld[1]{1},couplingTgasOld[1]{300},couplingUxOld[1]{},couplingUyOld[1]{},couplingUzOld[1]{};
 GpuReal rho[1]{1},rhoUx[1]{},rhoUy[1]{},rhoUz[1]{};
 GpuReal rhoMin=GPU_R(1e-8),TgasMin=1,Rgas=287,gammaGas=GPU_R(1.4),gasMu=GPU_R(1e-5),gasCp=1000,gasPrClamped=GPU_R(.7);
 double rhoE[1]{1e6};
 int nCells=1,solveParticleTemperature=1;
 GpuReal epsSMin=0,rhoSolid=1,particleDiameterFallback=.001,TpMin=-5,TpMax=4000;
 GpuReal momRhoP[1]{GPU_R(.001)},momRhoUPx[1]{},momRhoUPy[1]{},momRhoUPz[1]{},momRhoEP[1]{1},momRhoPD[1]{GPU_R(.001)},momRhoHpP[1]{GPU_REAL_MAX/GpuReal(1e9)};
 GpuReal epsS[1]{},rhoUsx[1]{},rhoUsy[1]{},rhoUsz[1]{},rhoDs[1]{},rhoHp[1]{},Usx[1]{},Usy[1]{},Usz[1]{},rhoEs[1]{},theta[1]{},dMeanCell[1]{},Tp[1]{};
};
void clearSolidCell(DeviceState&,int){std::abort();}
void recoverGasPrimitiveCell(DeviceState&,int){}
#include "operators/particleSpecificHeatDevice.cuh"
#include "operators/particleTemperatureFromSpecificEnthalpyDevice.cuh"
#include "GpuMaterialEnthalpyMoment.cuh"
#include "GpuMechanicalRecovery.cuh"
#include "gasTransport/GasStateView.H"
#include "gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#include "gpu/GpuCouplingMath.H"
CONDUCTIVITY
#include "operators/solidEpsFromMomentDevice.cuh"
#include "operators/applyEulerianParticleMaterialHeatKernel.cuh"
#include "operators/recoverPrimitivesKernel.cuh"
int main(){
 for(GpuReal minimum:{GpuReal(-5),GpuReal(1)}){
  DeviceState s;s.TpMin=minimum;solidRecoveryFromParticleMomentsCell(s,0);
  printf("mechanical %a %a %a %a %a\n",double(minimum),double(s.Tp[0]),double(s.rhoHp[0]),double(s.epsS[0]),double(s.theta[0]));
  applyEulerianParticleMaterialHeatKernel(&s,1.);printf("eulerian %a %a\n",double(minimum),s.rhoE[0]);
  s.epsS[0]=1;s.rhoHp[0]=-1;s.rhoEs[0]=1;s.rhoDs[0]=.001;recoverPrimitivesKernel(&s);
  printf("recover-solid %a %a %a\n",double(minimum),double(s.Tp[0]),double(s.rhoHp[0]));
  s.Tp[0]=-2;initialiseParticleMaterialEnthalpyKernel(&s);printf("initialise %a %a %a moment=%a\n",double(minimum),double(s.Tp[0]),double(s.rhoHp[0]),double(materialEnthalpyMoment(GpuReal(2),s.Tp[0])));
 }
}
