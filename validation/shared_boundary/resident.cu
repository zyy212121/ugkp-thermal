#include "gasBoundary/Resident.cuh"
#include "gasBoundary/Evaluate.H"
#include <cstdio>
#include <cassert>
#include <vector>
using namespace ugkp::boundary;
__global__ void sample(View v,double time,Primitive* output) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=v.count)return;
 const auto& b=v.faces[i];double value[Components];
 for(int c=0;c<Components;++c)value[c]=evaluate(v,b.function[c],time,b.value[c]);
 bool ok=totalState(b,value,{1,70,0,0,101325,300},287.05,-1,0,0,0,output[i]);
 if(!ok)asm("trap;");
}
int main(){
 Definition d;d.faces.resize(257);
 d.functions={{FunctionKind::Table,Bounds::Clamp,0,3,0},{FunctionKind::Polynomial,Bounds::Clamp,3,2,0}};
 d.terms={{0,302.5},{1,310},{2,320},{104200,0},{100,1}};
 for(auto& f:d.faces) {
  f.uMode=4;f.pMode=f.tMode=2;f.pressureGamma=f.temperatureGamma=1.4;f.compressiblePressure=1;
  f.value[Pressure]=104200;f.value[Temperature]=302.5;
 }
 for(int i=1;i<257;++i){d.faces[i].function[Temperature]=0;d.faces[i].function[Pressure]=1;}
 Resident resident;std::string error;cudaStream_t stream;assert(cudaStreamCreate(&stream)==cudaSuccess);
 assert(resident.configure(d,stream,error));Primitive* result;assert(cudaMalloc(&result,257*sizeof(Primitive))==cudaSuccess);
 cudaGraph_t graph;cudaGraphExec_t executable;assert(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)==cudaSuccess);
 sample<<<2,256,0,stream>>>(resident.view(),.5,result);
 assert(cudaStreamEndCapture(stream,&graph)==cudaSuccess);assert(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0)==cudaSuccess);
 for(int k=0;k<100;++k)assert(cudaGraphLaunch(executable,stream)==cudaSuccess);
 assert(cudaStreamSynchronize(stream)==cudaSuccess);
 std::vector<Primitive> h(257);assert(cudaMemcpy(h.data(),result,h.size()*sizeof(Primitive),cudaMemcpyDeviceToHost)==cudaSuccess);
 for(int i=0;i<257;++i) {
  double T0=i?306.25:302.5,p0=i?104250:104200;const auto& q=h[i];
  double K=.5*70*70,psi=1/(287.05*q.T),g=.4/1.4;
  assert(std::abs(q.T-T0/(1+psi*g*K))<1e-11);
  assert(std::abs(q.p-p0/std::pow(1+psi*g*K,1/g))<1e-8);
 }
 printf("DEVICE_REFERENCE %.17g %.17g %.17g %.17g %.17g\n",h[0].ux,h[0].uy,h[0].uz,h[0].p,h[0].T);
 printf("device_faces=257 graph_replays=100 table_polynomial_pass=1\n");
 cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);cudaFree(result);resident.clear();cudaStreamDestroy(stream);
}
