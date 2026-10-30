// Exercise the production resident donor/stoichiometry kernels, including the
// boundary-layer inputs which otherwise bypass generic flux scaling.
#include "gpu/SharedGasDeviceStorage.cuh"
#include "gpu/CoupledSurfaceTransfer.cuh"
#include "gpu/CoupledSurfacePipeline.cuh"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace chmt;
std::vector<void*> allocations;
template<class T>T* array(int n){T* p=nullptr;assert(cudaMallocManaged(&p,n*sizeof(T))==cudaSuccess);allocations.push_back(p);for(int i=0;i<n;++i)p[i]=T{};return p;}
int main(){
 auto* v=array<SharedGasDeviceView>(1);v->nCells=1;v->nFaces=3;
 v->cellPlaneStart=array<int>(1);v->cellPlaneCount=array<int>(1);v->cellPlaneCount[0]=3;
 v->cellFaceId=array<int>(3);v->faceOwner=array<int>(3);v->faceNeighbour=array<int>(3);
 v->V=array<double>(1);v->V[0]=1;v->magSf=array<double>(3);
 v->rho=array<double>(1);v->rho[0]=1;v->gasPhiRho=array<double>(3);
 for(int f=0;f<3;++f){v->cellFaceId[f]=f;v->faceNeighbour[f]=-1;v->magSf[f]=1;}
 v->gasSpecies.rho=array<double>(Ns);v->gasSpecies.flux=array<double>(3*Ns);
 auto* phase=array<double>(1);v->gasVolumeFraction=phase;
 auto* slots=array<int>(3);slots[0]=0;slots[1]=1;slots[2]=-1;
 auto* cells=array<int>(1);auto* faces=array<int>(2);faces[1]=1;
 auto* geometry=array<CoupledSurfaceInput>(2);auto* program=array<WallFaceSample>(2);
 auto* bounded=array<WallFaceSample>(2);auto* extent=array<double>(1);
 auto* wallSlots=array<int>(3);wallSlots[0]=0;wallSlots[1]=1;wallSlots[2]=-1;
 v->gasBoundaryLayer.enabled=true;v->gasBoundaryLayer.faceSlot=wallSlots;
 v->gasBoundaryLayerModel.input=array<ugkwp::gaswall::WallInput<double,Ns>>(2);
 for(int f=0;f<2;++f){geometry[f].area=1;program[f].temperature=700;program[f].normalVelocity=.25;
  program[f].velocity={1,2,3};program[f].speciesRate[0]=-1;program[f].speciesRate[1]=2;program[f].condensedRate[0]=1;
  v->gasBoundaryLayerModel.input[f].normal[0]=1;}
 for(int scenario=0;scenario<5;++scenario){
  phase[0]=scenario==3?.5:1;v->gasSpecies.rho[0]=scenario==1?0:.3;
  v->gasSpecies.rho[1]=.7;v->gasSpecies.flux[2]=.1;
  const bool transportReady=scenario!=2;
  for(int f=0;f<2;++f){program[f].poreRate[0]=scenario==4?-.2:0;program[f].poreSweepRate[0]=scenario==4?-.05:0;}
  boundSurfaceProgressKernel<<<1,1>>>(v,cells,1,slots,geometry,program,2,{},1,transportReady,extent);
  prepareResidentSurfaceKernel<<<1,2>>>(v,faces,2,geometry,program,{},extent,bounded);
  assert(cudaDeviceSynchronize()==cudaSuccess);
  const double reserve=1-64*2.2204460492503131e-16;
  const double other=(transportReady?phase[0]*.1:0)+(scenario==4?.1:0);
  const double expected=fmax(0,phase[0]*v->gasSpecies.rho[0]*reserve-other)/(scenario==4?2.3:2);
  assert(fabs(extent[0]-expected)<2e-16);
  double loss=other;
  for(int f=0;f<2;++f){const auto& w=bounded[f];
   assert(w.speciesRate[0]==-extent[0]&&w.speciesRate[1]==2*extent[0]&&w.condensedRate[0]==extent[0]);
   assert(w.temperature==700&&w.normalVelocity==.25&&w.velocity.y==2);
   assert(w.poreSweepRate[0]==program[f].poreSweepRate[0]);
   assert(v->gasBoundaryLayerModel.input[f].massFlux[0]==w.speciesRate[0]+w.poreRate[0]);
   loss-=w.speciesRate[0]+w.poreRate[0]-w.poreSweepRate[0];
  }
  // Already excessive transport is handled by its existing positivity pass;
  // the reaction consumes zero and cannot add another withdrawal.
  if(scenario!=1)assert(loss<=phase[0]*v->gasSpecies.rho[0]+1e-16);
  std::printf("scenario=%d progress=%.17g gross=%.17g\n",scenario,extent[0],loss);
 }
 // A gas density floor constrains progress too; it must not subsequently
 // scale wall conduction/pressure work together with an already-built packet.
 phase[0]=1;v->rhoMin=.8;v->gasSpecies.rho[0]=.3;v->gasSpecies.rho[1]=.7;v->gasSpecies.flux[2]=0;
 for(int f=0;f<2;++f){program[f].speciesRate[0]=program[f].speciesRate[1]=-1;program[f].poreRate[0]=program[f].poreSweepRate[0]=0;}
 boundSurfaceProgressKernel<<<1,1>>>(v,cells,1,slots,geometry,program,2,{},1,true,extent);
 assert(cudaDeviceSynchronize()==cudaSuccess);assert(fabs(extent[0]-.05)<2e-15);
 // An inactive constraint is bit-preserving, including opposing pore
 // channels. The second identical publication must not repeat the wall BVP.
 extent[0]=1;program[0].poreRate[0]=1e-12;program[0].poreSweepRate[0]=100;
 auto* changed=array<unsigned char>(2);
 prepareResidentSurfaceKernel<<<1,2>>>(v,faces,2,geometry,program,{},extent,bounded,changed);
 assert(cudaDeviceSynchronize()==cudaSuccess);assert(bounded[0].poreRate[0]==1e-12&&changed[0]);
 prepareResidentSurfaceKernel<<<1,2>>>(v,faces,2,geometry,program,{},extent,bounded,changed);
 assert(cudaDeviceSynchronize()==cudaSuccess);assert(!changed[0]&&!changed[1]);
 for(void* p:allocations)assert(cudaFree(p)==cudaSuccess);
}
