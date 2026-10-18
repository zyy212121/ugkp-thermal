#include <cuda_runtime.h>
#include "ablation/InterfaceMath.H"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>
using namespace chmt;
static void ck(cudaError_t q){if(q!=cudaSuccess){std::cerr<<cudaGetErrorString(q)<<'\n';exit(2);}}
struct Input{PhysicsConfig p;double b[Ns],y[Ns],R[Ns],D[Ns],Js[Ns];};

__global__ void calculate(const Input* inputs,int* flags,Real* states,int count){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;const auto& x=inputs[k];Real out[Ns]{};flags[k]=interfaceComposition(x.Js,x.b,101325,300,.01,x.p,1e-12,out);for(int i=0;i<Ns;++i)states[k*Ns+i]=out[i];}
__global__ void verify(const Input* inputs,int* flags,const Real* states,int count){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;const auto& x=inputs[k];const Real* out=states+k*Ns;Real f[Ns]{},sum=0,norm=0,tol=1e-12;bool valid=flags[k]&&compositionResidual(out,x.b,x.Js,101325,300,.01,x.p,f,nullptr);for(int i=0;i<Ns;++i){sum+=out[i];norm=maxValue(norm,absValue(f[i]));tol+=x.p.tolerances.relativeMass*absValue(x.Js[i]);valid=valid&&out[i]>=0&&out[i]<=1;}flags[k]=valid&&norm<=tol&&absValue(sum-1)<=1e-12;}
int main(int argc,char**argv){int mode=argc>1?atoi(argv[1]):0;const int count=4096;std::vector<Input> input(count);std::mt19937_64 gen(74921);std::uniform_real_distribution<double>u(0,1);for(auto& x:input){double sb=0,sy=0,R=0;for(int i=0;i<Ns;++i){x.b[i]=std::pow(10.,-5*u(gen));x.y[i]=std::pow(10.,-5*u(gen));sb+=x.b[i];sy+=x.y[i];x.R[i]=std::pow(10.,1+3*u(gen));x.D[i]=std::pow(10.,-6+5*u(gen));}for(int i=0;i<Ns;++i){x.b[i]/=sb;x.y[i]/=sy;}if(mode){sy=0;for(int i=0;i<Ns;++i){x.R[i]=250+150*u(gen);x.D[i]=1e-5+1e-4*u(gen);x.y[i]=x.b[i]*(1+(mode==1?1e-4:1e-6)*(u(gen)-.5));sy+=x.y[i];}for(int i=0;i<Ns;++i)x.y[i]/=sy;}for(int i=0;i<Ns;++i)R+=x.y[i]*x.R[i];double rho=101325/(R*300),J=mode?(mode==1?.01:1e-8):std::pow(10.,-5+5*u(gen)),S=0;for(int i=0;i<Ns;++i)S+=rho*x.D[i]/.01*(x.y[i]-x.b[i]);for(int i=0;i<Ns;++i)x.Js[i]=J*x.y[i]+rho*x.D[i]/.01*(x.y[i]-x.b[i])-x.y[i]*S;}

for(auto& x:input)for(int i=0;i<Ns;++i){x.p.species[i].R=x.R[i];x.p.gasDiffusivity[i]=x.D[i];}Input* data;int* flags;Real* states;ck(cudaMalloc(&states,count*Ns*sizeof(Real)));ck(cudaMalloc(&data,count*sizeof(Input)));ck(cudaMalloc(&flags,count*sizeof(int)));ck(cudaMemcpy(data,input.data(),count*sizeof(Input),cudaMemcpyHostToDevice));cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));std::vector<float> times;
for(int it=0;it<10;++it){ck(cudaEventRecord(start));calculate<<<(count+31)/32,32>>>(data,flags,states,count);ck(cudaGetLastError());ck(cudaEventRecord(stop));ck(cudaEventSynchronize(stop));float elapsed=0;ck(cudaEventElapsedTime(&elapsed,start,stop));if(it>=3)times.push_back(elapsed);}
verify<<<(count+31)/32,32>>>(data,flags,states,count);ck(cudaGetLastError());ck(cudaDeviceSynchronize());std::vector<int> host(count);ck(cudaMemcpy(host.data(),flags,count*sizeof(int),cudaMemcpyDeviceToHost));int good=0;for(int v:host)good+=v;std::cout<<std::setprecision(12)<<"ns="<<Ns<<" mode="<<mode<<" count="<<count<<" good="<<good<<" milliseconds=";for(float v:times)std::cout<<v<<',';std::cout<<'\n';std::vector<Real> output(count*Ns);ck(cudaMemcpy(output.data(),states,count*Ns*sizeof(Real),cudaMemcpyDeviceToHost));if(argc>2){FILE* f=std::fopen(argv[2],"wb");std::fwrite(output.data(),sizeof(Real),output.size(),f);std::fclose(f);}ck(cudaFree(states));ck(cudaFree(flags));ck(cudaFree(data));ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));return good==count?0:3;}
