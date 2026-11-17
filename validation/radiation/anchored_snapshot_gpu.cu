#include "common/gpu/AnchoredRadiationSnapshot.cuh"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace ugkp::gpu;
void checked(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::abort();}}
template<class T>T* upload(const std::vector<T>& x){T* p=nullptr;checked(cudaMalloc((void**)&p,x.size()*sizeof(T)));checked(cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
__global__ void evaluate(const RadiationEntryParticle<GpuReal>* entries,const int* offsets,GpuReal q,AnchoredRadiationSample<GpuReal>* out){
 int c=threadIdx.x;if(c>=2)return;
 AnchoredParticleRadiationResponse<GpuReal,ugkwp::ParticleCaloric> response{entries+offsets[c],offsets[c+1]-offsets[c],GpuReal(100),GpuReal(5000),{}};
 out[c]=response(q);
}
int main(){
 std::vector<GpuReal> mass={GpuReal(.2),GpuReal(.3),GpuReal(.4),GpuReal(.5)},temp={GpuReal(300),GpuReal(400),GpuReal(600),GpuReal(800)};
 std::vector<int> status={1,1,1,1},cell={0,1,0,1},offset={0,2,4},index={2,0,3,1},count={4};
 std::vector<unsigned char> wall={0,0,0,1};
 auto* dm=upload(mass);auto* dt=upload(temp);auto* ds=upload(status);auto* dc=upload(cell);auto* dw=upload(wall);auto* dn=upload(count);auto* off=upload(offset);auto* idx=upload(index);
 ParticleRadiationResponseView v{};v.enabled=true;v.capacity=4;v.count=dn;v.cell=dc;v.status=ds;v.wallState=dw;v.mass=dm;v.temperature=dt;v.minimumTemperature=100;v.maximumTemperature=5000;
#ifndef TEST_FP32
 auto* carry=upload(std::vector<double>{.03125,-.0625,.125,.25});v.enthalpyCarry=carry;
#endif
 auto* deviceView=upload(std::vector<ParticleRadiationResponseView>{v});auto borrowed=v;borrowed.deviceView=deviceView;
 RadiationEntryParticle<GpuReal>* anchor=nullptr;checked(cudaMalloc((void**)&anchor,4*sizeof(*anchor)));int* mapping=nullptr;checked(cudaMalloc((void**)&mapping,4*sizeof(int)));
 AnchoredRadiationSample<GpuReal>* result=nullptr;checked(cudaMalloc((void**)&result,2*sizeof(*result)));
 for(bool indirect:{false,true}){
  captureAnchoredRadiationEntriesKernel<<<1,128>>>(borrowed,off,indirect?idx:nullptr,2,anchor,mapping);checked(cudaGetLastError());checked(cudaDeviceSynchronize());
  std::vector<RadiationEntryParticle<GpuReal>> frozen(4),after(4);std::vector<int> mapped(4);std::vector<GpuReal> resident(4);
  checked(cudaMemcpy(frozen.data(),anchor,4*sizeof(*anchor),cudaMemcpyDeviceToHost));checked(cudaMemcpy(mapped.data(),mapping,4*sizeof(int),cudaMemcpyDeviceToHost));checked(cudaMemcpy(resident.data(),dt,4*sizeof(GpuReal),cudaMemcpyDeviceToHost));assert(resident==temp);
  for(int s=0;s<4;++s){int i=indirect?index[s]:s;assert(mapped[s]==i&&frozen[s].mass==mass[i]&&frozen[s].thermal.temperature==temp[i]&&frozen[s].eligible==(wall[i]==0));
#ifndef TEST_FP32
   const double expected[]={.03125,-.0625,.125,.25};assert(frozen[s].thermal.specificEnthalpyResidual==expected[i]);
#endif
  }
  AnchoredRadiationSample<GpuReal> first[2],repeat[2];
  evaluate<<<1,2>>>(anchor,off,GpuReal(10),result);checked(cudaMemcpy(first,result,sizeof(first),cudaMemcpyDeviceToHost));
  evaluate<<<1,2>>>(anchor,off,GpuReal(-10),result);checked(cudaDeviceSynchronize());
  evaluate<<<1,2>>>(anchor,off,GpuReal(10),result);checked(cudaMemcpy(repeat,result,sizeof(repeat),cudaMemcpyDeviceToHost));
  for(int c=0;c<2;++c){assert(first[c].withinDomain&&first[c].publicationValid);assert(first[c].actualEnergy==repeat[c].actualEnergy&&first[c].temperature==repeat[c].temperature);}
  checked(cudaMemcpy(after.data(),anchor,4*sizeof(*anchor),cudaMemcpyDeviceToHost));assert(std::memcmp(frozen.data(),after.data(),4*sizeof(*anchor))==0);
  checked(cudaMemcpy(resident.data(),dt,4*sizeof(GpuReal),cudaMemcpyDeviceToHost));assert(resident==temp);
 }
 std::printf("SNAPSHOT_PASS bits=%zu contiguous indexed device_view eligibility immutable_entry unchanged_resident repeated_bidirectional\n",sizeof(GpuReal)*8);
 checked(cudaDeviceReset());
}
