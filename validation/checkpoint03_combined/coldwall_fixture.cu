// Combined checkpoint only. No production source or runtime gate is changed.
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include "GpuPrecisionTypes.H"
#include "wall/GpuColdWallSolidification.H"
#include "GpuParticleThermalCarry.H"
#include "GpuParticleCaloric.H"
struct DeviceState {
 int coldWallSolidificationEnabled=1;
 GpuReal *pColdNodeSpecificEnthalpy,*pColdRingSolidMass,*pColdFrozenArea;
 GpuTime* pColdContactAge;
 Foam::gpuThermal::ColdWallSolidificationParameters coldWallSolidificationParameters;
};
#include "wall/GpuColdWallThermalCarry.cuh"
// Count calls to the original PCR function without replacing its arithmetic.
#define coldWall1DPcrSolve cp3OriginalPcrSolve
#include "wall/GpuColdWall1DAlgebra.cuh"
#undef coldWall1DPcrSolve
__device__ unsigned cp3PcrCalls;
__device__ inline ColdWallThermalReal coldWall1DPcrSolve(ColdWallThermalReal a,
 ColdWallThermalReal b,ColdWallThermalReal c,ColdWallThermalReal d,int lane,unsigned mask){
 if(lane==0)atomicAdd(&cp3PcrCalls,1u);
 return cp3OriginalPcrSolve(a,b,c,d,lane,mask);
}
#define GPU_COLD_WALL_1D_ALGEBRA_ONLY
#include "wall/GpuColdWall1DDevice.cuh"
using namespace Foam::gpuThermal;
void ck(bool x,const char* text){if(!x)throw std::runtime_error(text);}
void cu(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void advance(DeviceState*s,GpuReal wall,GpuReal gas,GpuReal conductance,GpuTime dt,GpuReal*out){
 GpuReal t=0,f=0,q=0;
 bool ok=advanceColdWall1DThermalGroup(*s,0,threadIdx.x,0xffu,
  GPU_R(9.047786842338605e-13),GPU_R(2.8952917895483536e-9),
  GPU_R(2.2e-8),GPU_R(1.7e-8),8e-6,GPU_R(.42),dt,wall,GPU_R(13000),
  GPU_R(.1),gas,conductance,t,f,q);
 if(threadIdx.x==0){out[0]=ok;out[1]=t;out[2]=f;out[3]=q;}
}
int main()try{
 std::cout<<std::setprecision(17);
 GpuReal *h,*rings,*frozen,*out;GpuTime*age;DeviceState*s;
 cu(cudaMallocManaged(&h,8*sizeof(*h)));cu(cudaMallocManaged(&rings,8*sizeof(*rings)));
 cu(cudaMallocManaged(&frozen,sizeof(*frozen)));cu(cudaMallocManaged(&out,4*sizeof(*out)));
 cu(cudaMallocManaged(&age,sizeof(*age)));cu(cudaMallocManaged(&s,sizeof(*s)));
 const ColdWallSolidificationParameters pars{GPU_R(2327),GPU_R(20),GPU_R(1.16e6),
  GPU_R(3990),GPU_R(1273),GPU_R(5.9),GPU_R(.25),GPU_R(0),1,12};
 const double eps=std::numeric_limits<GpuReal>::epsilon(),mass=double(GPU_R(2.8952917895483536e-9));
 // Frozen input cases: solid, post-gas solid, crossing, liquid, two sign reversals.
 const std::array<const char*,6> names={"allsolid","postgas_solid","crossphase","liquid",
  "bottom_cold_to_hot","bottom_hot_to_cold"};
 const double walls[6]={486,486,4000,486,500,2100};
 const double initial[6]={1500,1500,2316,3500,2200,400};
 const double dt[6]={7.5e-8,1e-6,1e-4,7.5e-8,1e-4,1e-4};
 for(int k=0;k<6;++k){
  for(int j=0;j<8;++j){double t=initial[k];if(j==0&&k==4)t=400;if(j==0&&k==5)t=2200;
   h[j]=coldWallSpecificEnthalpyJkg(GpuReal(t),pars);rings[j]=0;}
  *age=8e-6;*frozen=0;*s={1,h,rings,frozen,age,pars};
  double before=0;for(int j=0;j<8;++j)before+=double(h[j])*mass/8;
  const GpuReal gas=k==1?GPU_R(1200):GPU_R(1500),conductance=k==1?GPU_R(3e-4):GPU_R(0);
  GpuReal gasIncrement=0;GpuReal meanH=0;for(int j=0;j<8;++j)meanH+=h[j]/GPU_R(8);
  coldWallGasSpecificEnthalpyIncrement(meanH,GpuReal(mass),gas,conductance,dt[k],pars,gasIncrement);
  unsigned zero=0;cu(cudaMemcpyToSymbol(cp3PcrCalls,&zero,sizeof(zero)));
  advance<<<1,8>>>(s,GpuReal(walls[k]),gas,conductance,dt[k],out);cu(cudaDeviceSynchronize());
  unsigned calls=0;cu(cudaMemcpyFromSymbol(&calls,cp3PcrCalls,sizeof(calls)));
  ck(out[0]==1&&std::isfinite(double(out[1]))&&std::isfinite(double(out[3])),"nonfinite coldWall result");
  double after=0,ringMass=0;for(int j=0;j<8;++j){after+=double(h[j])*mass/8;ringMass+=rings[j];ck(std::isfinite(double(h[j]))&&rings[j]>=0,"invalid cold profile");}
  const double gasEnergy=double(gasIncrement)*mass;
  const double closure=after-before+out[3]-gasEnergy;
  const double gross=std::abs(before)+std::abs(after)+std::abs(double(out[3]))+std::abs(gasEnergy);
  const double bound=(sizeof(GpuReal)==4?512*eps:2e-10)*gross+1e-18;
  const double bottom=coldWallTemperatureFromSpecificEnthalpyK(h[0],pars);
  ck(std::abs(closure)<=bound,"coldWall dimensional energy closure");
  ck(ringMass<=mass*(1+512*eps),"ring mass exceeds particle mass");
  if(k==0||k==1||k==3)ck(after<before&&out[3]>0&&out[1]<initial[k],"resolvable cooling exchange was not executed");
  if(k==2)ck(after>before&&out[3]<0&&out[1]>initial[k],"resolvable heating/crossphase exchange was not executed");
  if(k==0||k==1||k==4||k==5)ck(calls==1,"all-solid first solve not reused");
  if(k==2||k==3)ck(calls>1,"crossphase/liquid old iterations not exercised");
  if(k==4)ck(bottom>walls[k],"cold-to-hot bottom sign reversal not exercised");
  if(k==5)ck(bottom<walls[k],"hot-to-cold bottom sign reversal not exercised");
  if(k==4||k==5){
   // Independent allocation observation: final bottom cooling uses the ring
   // conductance; final heating uses wet area. It never changes solver input.
   const double maximum=double(GPU_R(2.2e-8)),contact=double(GPU_R(1.7e-8)),volume=double(GPU_R(9.047786842338605e-13));
   const double ringArea=maximum/8,thickness=volume/contact,nodeThickness=thickness/8;
   double weights[8]{},capacity[8]{},expected[8]{};
   for(int j=0;j<8;++j){
    const double wet=std::max(0.,std::min(ringArea,contact-j*ringArea));
    const double fraction=(j+.5)/8,coordinate=1-std::sqrt(1-fraction),firstWet=8e-6*.42*coordinate;
    const double age0=std::max(0.,8e-6-firstWet),age1=std::max(0.,8e-6+dt[k]-firstWet);
    const double conductanceIntegral=wallInterfaceConductanceTimeIntegral(GpuReal(wet),GPU_R(.1),age0,age1,
      pars.interfaceResistanceM2KW,GpuReal(.5*nodeThickness/pars.solidThermalConductivityWmK),GPU_R(13000),true);
    weights[j]=bottom>walls[k]?conductanceIntegral/dt[k]*(bottom-walls[k]):wet;
    capacity[j]=pars.solidDensityKgM3*wet*thickness;
   }
   double remaining=mass;
   for(int pass=0;pass<8&&remaining>1e-18*mass;++pass){double total=0;for(int j=0;j<8;++j)if(capacity[j]>expected[j])total+=weights[j];
    if(total==0)break;double allocated=0;for(int j=0;j<8;++j)if(capacity[j]>expected[j]){double v=std::min(capacity[j]-expected[j],remaining*weights[j]/total);expected[j]+=v;allocated+=v;}
    remaining-=allocated;
   }
   double error=0;for(int j=0;j<8;++j)error+=std::abs(double(rings[j])-expected[j]);
   const double allowed=256*eps*mass+1e-18;ck(error<=allowed,"ring weighting does not follow final bottom direction");
   std::cout<<"RING_DIRECTION case="<<names[k]<<" errorKg="<<error<<" allowedKg="<<allowed<<'\n';
  }
  std::cout<<"COLD case="<<names[k]<<" pcrCalls="<<calls<<" bottom="<<bottom
   <<" wall="<<walls[k]<<" wallEnergyJ="<<out[3]<<" gasEnergyJ="<<gasEnergy
   <<" closureJ="<<closure<<" allowedJ="<<bound<<" ringMassKg="<<ringMass<<'\n';
 }
 // Zero proposed heat must preserve a single stored temperature, including
 // values not reproduced exactly by an enthalpy/temperature round trip.
 ugkwp::ParticleCaloric law;law.phaseChange=true;law.solidCp=GPU_R(1273);
 for(double t:{300.125,2317.,2327.,4000.,4999.875}){
  ugkwp::ParticleThermalCarry<GpuReal> q{};q.temperature=GpuReal(t);
  const GpuReal old=q.temperature;
  for(int repeat=0;repeat<1000;++repeat){auto pub=ugkwp::publishParticleThermalHeat(q,GPU_R(0),GPU_R(100),GPU_R(5000),law);
   ck(pub.valid&&pub.acceptedSpecificHeat==0&&q.temperature==old,"zero heat drifted particle temperature");}
  std::cout<<"ZERO_HEAT temperature="<<q.temperature<<" repetitions=1000\n";
 }
 cudaFree(h);cudaFree(rings);cudaFree(frozen);cudaFree(out);cudaFree(age);cudaFree(s);
 std::cout<<"PASS coldwall first solve, crossphase, ring direction, energy closure, no-heat state\n";
 return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
