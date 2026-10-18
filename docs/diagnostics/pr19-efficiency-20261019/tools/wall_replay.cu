#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/WallCudaTeam.cuh"
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cassert>
#include <string>
using namespace ugkwp::gaswall;
using Input=WallInput<double,2>;using Work=WallWorkspace<double,2,24>;using Output=WallOutput<double,2>;
void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);}}
template<class T>T* alloc(int n=1){T*p;check(cudaMalloc(&p,sizeof(T)*n));return p;}
__global__ __launch_bounds__(32) void solve(Input* in,WallModelConfig<double>* c,Work* w,Output* out,WallStatus* status,int* ok,double* profile){int i=blockIdx.x;detail::CudaWallTeam team;bool v=ugkwp::evaluateGasBoundaryLayerWorkspace<true>(in[i],*c,w[i],out[i],status[i],profile+i*24*9,team);if(team.leader())ok[i]=v;}
void loadActual(const char* path,Input* input,int count){FILE*f=fopen(path,"r");assert(f);char line[16384];assert(fgets(line,sizeof(line),f));for(int i=0;i<count;++i){assert(fgets(line,sizeof(line),f));char*p=line;double v[26];for(int j=0;j<26;++j){char*end;v[j]=strtod(p,&end);assert(end!=p);p=*end==','?end+1:end;}
input[i].pressure=v[1];
input[i].temperature=v[2];
input[i].matchingDistance=v[3];
input[i].ownerDistance=v[4];
input[i].massFlux[0]=v[5];
input[i].massFlux[1]=v[6];
input[i].velocity[0]=v[7];
input[i].velocity[1]=v[8];
input[i].velocity[2]=v[9];
input[i].normal[0]=v[10];
input[i].normal[1]=v[11];
input[i].normal[2]=v[12];
input[i].matching.temperature=v[13];
input[i].matching.k=v[14];
input[i].matching.omega=v[15];
input[i].matching.velocity[0]=v[16];
input[i].matching.velocity[1]=v[17];
input[i].matching.velocity[2]=v[18];
input[i].matching.massFraction[0]=v[19];
input[i].matching.massFraction[1]=v[20];
input[i].model.viscosity=v[21];
input[i].model.conductivity=v[22];
input[i].model.molecularPrandtl=v[23];
input[i].model.diffusivity[0]=v[24];
input[i].model.diffusivity[1]=v[25];
 }fclose(f);}
int main(int argc,char**argv){assert(argc>=2);constexpr int count=336;CapturedSstWall capture;
if(argc>3)capture.config.maxIterations=atoi(argv[3]);
std::vector<Input> hostIn(count,capture.input);std::vector<Work>hostW(count);std::vector<Output>hostOut(count);std::vector<WallStatus>hostStatus(count);std::vector<int>hostOk(count);
auto in=alloc<Input>(count);auto c=alloc<WallModelConfig<double>>();auto w=alloc<Work>(count);auto out=alloc<Output>(count);auto status=alloc<WallStatus>(count);auto ok=alloc<int>(count);auto profile=alloc<double>(count*24*9);auto species=alloc<ugkwp::SpeciesThermoData<double>>(2);auto coeff=alloc<double>(6);
check(cudaMemcpy(species,capture.species,sizeof(capture.species),cudaMemcpyHostToDevice));check(cudaMemcpy(coeff,capture.coefficients,sizeof(capture.coefficients),cudaMemcpyHostToDevice));check(cudaMemcpy(c,&capture.config,sizeof(capture.config),cudaMemcpyHostToDevice));
loadActual(argv[1],hostIn.data(),count);for(int i=0;i<count;++i){hostIn[i].model.thermo.species=species;hostIn[i].model.thermo.coefficients=coeff;}
check(cudaMemcpy(in,hostIn.data(),sizeof(Input)*count,cudaMemcpyHostToDevice));check(cudaMemset(w,0,sizeof(Work)*count));check(cudaMemset(out,0,sizeof(Output)*count));check(cudaMemset(status,0,sizeof(WallStatus)*count));check(cudaMemset(ok,0,sizeof(int)*count));check(cudaMemset(profile,0,sizeof(double)*count*24*9));
cudaFuncAttributes a;check(cudaFuncGetAttributes(&a,solve));printf("registers=%d localSizeBytes=%zu sharedSizeBytes=%zu memory=cudaMalloc\n",a.numRegs,a.localSizeBytes,a.sharedSizeBytes);
int resident=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,solve,32,0));printf("blocks_per_sm=%d\n",resident);
cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));std::vector<float> samples;
for(int repeat=0;repeat<10;++repeat){check(cudaEventRecord(start));solve<<<count,32>>>(in,c,w,out,status,ok,profile);check(cudaGetLastError());check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));float ms=0;check(cudaEventElapsedTime(&ms,start,stop));if(repeat>=2)samples.push_back(ms);}
std::sort(samples.begin(),samples.end());printf("actual-wall336 median_ms=%.12g samples_ms=",samples[samples.size()/2]);for(float t:samples)printf("%.12g,",t);puts("");
check(cudaMemcpy(hostW.data(),w,sizeof(Work)*count,cudaMemcpyDeviceToHost));check(cudaMemcpy(hostOut.data(),out,sizeof(Output)*count,cudaMemcpyDeviceToHost));check(cudaMemcpy(hostStatus.data(),status,sizeof(WallStatus)*count,cudaMemcpyDeviceToHost));check(cudaMemcpy(hostOk.data(),ok,sizeof(int)*count,cudaMemcpyDeviceToHost));
for(int i=0;i<count;++i){assert(hostOk[i]);if(capture.config.maxIterations)assert(hostOut[i].residual<=capture.config.absoluteTolerance+capture.config.relativeTolerance);assert(hostW[i].state[0][5]==0);for(int j=1;j<capture.config.nodes;++j)assert(hostW[i].state[j][5]>0);}printf("iterations=%d residual=%.17g heat=%.17g\n",hostOut[0].iterations,hostOut[0].residual,hostOut[0].conductiveHeatFlux);
if(argc>=3){std::vector<double> hostProfile(count*24*9);
check(cudaMemcpy(hostProfile.data(),profile,sizeof(double)*hostProfile.size(),cudaMemcpyDeviceToHost));
std::string prefix=argv[2];FILE* f=fopen((prefix+".profile.bin").c_str(),"wb");assert(f);fwrite(hostProfile.data(),sizeof(double),hostProfile.size(),f);fclose(f);
f=fopen((prefix+".state.bin").c_str(),"wb");assert(f);for(auto& a:hostW)fwrite(a.state,sizeof(double),24*6,f);fclose(f);
}
if(argc>=3){FILE*f=fopen(argv[2],"w");assert(f);fprintf(f,"slot,iterations,residual,heat\n");for(int i=0;i<count;++i)fprintf(f,"%d,%d,%.17g,%.17g\n",i,hostOut[i].iterations,hostOut[i].residual,hostOut[i].conductiveHeatFlux);fclose(f);}
}
