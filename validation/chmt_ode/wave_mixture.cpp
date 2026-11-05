// Serial execution of actual shared kernels, checked against the independently
// rearranged implicit wave-pressure equation. Full CUDA coverage is the CHMT run.
// Runs unchanged production device functions serially on a host. Not CUDA execution.
#include <cmath>
using std::isfinite;
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "GpuPrecisionTypes.H"
#include "CharacteristicMuscl.cuh"
#include "OpenFoamLimitedLinear.cuh"
#include "OpenFoamViscousFlux.cuh"
#include "OpenFoamWallFunctions.cuh"
#include "RiemannBoundaryState.cuh"
#include "RiemannGasFlux.cuh"
#include "GpuSstAlgebra.cuh"
#include "GpuLesAlgebra.cuh"
using Real=GpuReal;
using Time=double;
#define GPU_OPERATOR_REAL Real
#define GPU_OPERATOR_TIME Time
#define GPU_OPERATOR_R GPU_R
#define GPU_OPERATOR_TINY(x) GPU_R(1e-30)
#define __host__
#define __global__
#define __device__
#define __forceinline__ inline
// PTX traps are represented by process failure in this serial host emulator.
#define asm(...) std::abort()
struct ThreadIndex { int x=0; } blockIdx, threadIdx, blockDim;
constexpr Real OfSmall=GPU_R(2.22044604925031308085e-16);
constexpr Real OfVSmall=GPU_R(1e-30);
constexpr Real OfGreat=GPU_R(1.0)/OfSmall;
struct GasPrimDevice { Real rho,ux,uy,uz,p,T; };
#include "gasTransport/GasStateView.H"
#include "operators/clampMin.cuh"
#include "operators/clampRange.cuh"
#include "operators/linearScheduledValueDevice.cuh"
#include "operators/makeGasPrimDevice.cuh"
#include "operators/updateWaveTransmissivePressureBoundaryKernel.cuh"
#include "operators/riemannFacePrimitiveForGradient.cuh"
#include "operators/computeGasPrimitiveGradientsKernel.cuh"
struct TestState:ugkwp::GasStateView<Real,Time,decltype(ugkwp::defaultSstCoefficients())>{ugkwp::GasSpeciesState<Real,2> gasSpecies;ugkwp::GasBoundaryLayerState<Real> gasBoundaryLayer;};
#include <iostream>
#include <stdexcept>
int main(){
 TestState s{};s.nCells=1;s.nFaces=1;s.nInternalFaces=0;s.Rgas=287;s.rhoMin=1e-12;s.TgasMin=100;
 ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;
#define REAL_FIELD(name,value) Real name[]={value};s.name=name
#define INT_FIELD(name,value) int name[]={value};s.name=name
 INT_FIELD(faceOwner,0);REAL_FIELD(magSf,1);REAL_FIELD(Sfx,1);REAL_FIELD(Sfy,0);REAL_FIELD(Sfz,0);
 REAL_FIELD(deltaCoeffs,20);REAL_FIELD(rho,2);REAL_FIELD(Ux,0);REAL_FIELD(Uy,0);REAL_FIELD(Uz,0);REAL_FIELD(p,300000);
 REAL_FIELD(gasBoundaryUx,100);REAL_FIELD(gasBoundaryUy,0);REAL_FIELD(gasBoundaryUz,0);INT_FIELD(gasBoundaryUFix,2);
 REAL_FIELD(gasBoundaryRho,2);REAL_FIELD(gasBoundaryP,150000);REAL_FIELD(gasBoundaryT,600);
 INT_FIELD(gasBoundaryPWave,1);REAL_FIELD(gasBoundaryPWaveGamma,1.2);REAL_FIELD(gasBoundaryPWaveLInf,0);REAL_FIELD(gasBoundaryPWaveFieldInf,101325);
 REAL_FIELD(riemannBoundaryP,150000);REAL_FIELD(riemannBoundaryT,600);REAL_FIELD(riemannBoundaryRho,1);
 REAL_FIELD(riemannBoundaryUx,100);REAL_FIELD(riemannBoundaryUy,0);REAL_FIELD(riemannBoundaryUz,0);
 INT_FIELD(riemannBoundaryPWave,1);INT_FIELD(riemannBoundaryPFix,0);INT_FIELD(riemannBoundaryTFix,1);INT_FIELD(riemannBoundaryRhoFix,0);INT_FIELD(riemannBoundaryUFix,2);
 INT_FIELD(riemannBoundaryKind,2);
 ugkwp::SpeciesThermoData<Real> species[2]{};Real coeff[]={1200,0,0,1500,0,0};
 for(int k=0;k<2;++k){species[k].molarMass=k?.044:.028;species[k].coefficientOffset=3*k;species[k].minTemperature=100;species[k].maxTemperature=5000;}
 s.gasSpecies.thermo.species=species;s.gasSpecies.thermo.coefficients=coeff;s.gasSpecies.thermo.coefficientCount=6;
 Real density[]={1.5,.5},boundaryY[]={.2,.8};int fixed[]={0};s.gasSpecies.rho=density;s.gasSpecies.boundaryMassFraction=boundaryY;s.gasSpecies.compositionBoundaryFixed=fixed;
 blockIdx.x=threadIdx.x=0;blockDim.x=1;int count=0;
 for(bool mixture:{false,true})for(int composition:{0,1})for(double velocity:{-1000.,0.,100.})for(double farLength:{0.,.2})for(int thermoPolicy:{0,1,2}){
  ugkp::diagnostics::reset(diagnostics,1,1,0,0);s.gasSpecies.mode=mixture?ugkwp::GasMode::MixtureFrozen:ugkwp::GasMode::SingleLegacy;
  fixed[0]=composition;Ux[0]=velocity;gasBoundaryUx[0]=velocity;gasBoundaryPWaveLInf[0]=farLength;gasBoundaryP[0]=riemannBoundaryP[0]=150000;
  const long double y0=composition?.2L:.75L,y1=1-y0;
  const long double R=8.31446261815324L*(y0/.028L+y1/.044L);
  riemannBoundaryRhoFix[0]=thermoPolicy==1;riemannBoundaryTFix[0]=thermoPolicy==0;
  riemannBoundaryRho[0]=gasBoundaryRho[0]=2;
  const long double boundaryT=thermoPolicy==1?150000/(2*R):600;
  const long double psi=mixture?1/(R*boundaryT):2.L/150000;
  const long double speed=std::max(0.L,velocity+std::sqrt(1.2L/psi));
  const long double alpha=speed*1e-5L*20,k=farLength>0?speed*1e-5L/farLength:0;
  const long double expected=(150000+alpha*300000+k*101325)/(1+alpha+k);
  updateWaveTransmissivePressureBoundaryKernel(&s,1e-5);
  if(std::abs(gasBoundaryP[0]-expected)>1e-9L)throw std::runtime_error("wave pressure equation mismatch");
  if(mixture)for(bool refined:{false,true}){
   auto q=riemannBoundaryState(s,0,GasPrimDevice{2,velocity,0,0,300000,600},nullptr,refined);
   if(ugkp::diagnostics::hasFatal(&diagnostics)||std::abs(q.p-expected)>1e-9L||std::abs(q.rho-(thermoPolicy==1?2:expected/(R*600)))>1e-12L)
    throw std::runtime_error("mixture boundary EOS/wave-pressure consumption mismatch");
  }
  ++count;
 }

 // The same primitive wall-projection operator used in the native stage must
 // retain macro SST at a quiescent ODE wall, even with nonzero initial k.
 s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s.sstConfigured=1;s.sstWallTreatment=2;
 s.sstOmegaMin=1e-6;s.sstKMin=1e-12;s.gasMu=9.29152148664344e-5;
 s.sstCoefficients=ugkwp::defaultSstCoefficients();s.sstWallCmu=.09;s.sstWallKappa=.41;s.sstWallE=9.8;
 REAL_FIELD(Tgas,300);REAL_FIELD(k,1.52223);REAL_FIELD(omega,247.536);REAL_FIELD(rhoOmega,495.072);REAL_FIELD(sstWallDistance,7e-5);
 INT_FIELD(cellPlaneStart,0);INT_FIELD(cellPlaneCount,1);INT_FIELD(cellFaceId,0);INT_FIELD(faceNeighbour,-1);INT_FIELD(facePeriodicPair,-1);
 Ux[0]=0;riemannBoundaryUx[0]=0;riemannBoundaryUFix[0]=1;riemannBoundaryTFix[0]=0;riemannBoundaryPWave[0]=0;
 int slots[]={0},status[]={0};ugkwp::GasBoundaryLayerExchange<Real> exchange[1]{};exchange[0].ready=true;
 ugkwp::GasBoundaryLayerSstClosure<Real> closure[1]{};
 auto& wall=s.gasBoundaryLayer;wall.enabled=true;wall.count=1;wall.faceSlot=wall.ownerSlot=slots;wall.status=status;wall.exchange=exchange;wall.sst=closure;
 applySstWallFunctionStateCell(s,0);const Real before=omega[0];
 struct Model{ugkwp::gaswall::WallModelConfig<Real> config;bool volumeSstClosure=true;} model;
 model.config.model=ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde;model.config.enableSst=true;
 ugkwp::configureEquilibriumOdeMacroSst(wall,model);
 applySstWallFunctionStateCell(s,0);
 if(before!=s.sstOmegaMin||omega[0]<=1000||model.config.enableSst||model.volumeSstClosure||!wall.useMacroSstInventory)
  throw std::runtime_error("ODE macro SST projection was not retained");
 std::cout<<"PASS quiescent ODE wall omega before="<<before<<" after="<<omega[0]<<"\n";
 std::cout<<"PASS "<<count<<" legacy/mixture wave-pressure configurations, fixed/extrapolated composition, inflow/outflow, relaxation, both refinement paths\n";
}
