
#include "common/gasWall/WallModel.H"
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
using namespace ugkwp;
using namespace ugkwp::gaswall;
struct Model {
 SpeciesThermoData<double> sp[2]; double coef[6]={1000,0,0,1000,0,0};
 double elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<double> rx[1]; GasStoichTerm<double> re[1],pr[1];
 WallInput<double,2> in;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 auto& t=in.model.thermo;t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 rx[0].reactantCount=1;rx[0].productCount=1;rx[0].highRate.preExponential=2;re[0].species=0;re[0].coefficient=1;pr[0].species=1;pr[0].coefficient=1;
 auto& m=in.model.mechanism;m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=1;m.products=pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 in.integrateOwnerKSource=true; // This fixture explicitly tests auxiliary source integrals.
 in.pressure=101325;in.temperature=300;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;
 in.matching.temperature=500;in.matching.velocity[0]=10;in.matching.massFraction[0]=.7;in.matching.massFraction[1]=.3;
 in.model.viscosity=2e-5;in.model.conductivity=.04;in.model.diffusivity[0]=in.model.diffusivity[1]=1e-5;
 }
};
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<1e-10+r*std::abs(b);}

#include "common/gasWall/WallCudaTeam.cuh"
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <cmath>
using namespace ugkwp::gaswall;
using In=WallInput<double,2>;using Out=WallOutput<double,2>;using Diag=ugkp::diagnostics::DeviceContext;
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}
template<class T>T* dev(int n){T* p;ck(cudaMalloc(&p,sizeof(T)*n));return p;}
template<int C> __global__ __launch_bounds__(32) void pool(In* input,WallModelConfig<double>* config,WallWorkspace<double,2,C>* work,Out* output,WallStatus* status,int* ok,double* profile,int count,int workers){
 const int worker=blockIdx.x;detail::CudaWallTeam team;
 if(worker>=workers)return;
 for(int slot=worker;slot<count;slot+=workers){
  bool result=ugkwp::evaluateGasBoundaryLayerWorkspace<true>(input[slot],*config,work[worker],output[slot],status[slot],profile+slot*config->nodes*9,team);
  if(team.leader())ok[slot]=result;
 }
}
template<int C> void run(int nodes,int mode){
 constexpr int count=5,workers=2;Model fixture;WallModelConfig<double> config;config.enableSst=true;
 config.nodes=nodes;config.maxIterations=100;
 fixture.in.temperature=fixture.in.matching.temperature=500;fixture.in.matching.velocity[0]=30;fixture.in.matching.k=.5;fixture.in.matching.omega=200;fixture.in.volumeSstClosure=false;fixture.in.integrateOwnerKSource=true;
 std::vector<double> hy{.002,.006},hw{2e-7,2e-7};
 auto qy=dev<double>(hy.size()),qw=dev<double>(hw.size());
 ck(cudaMemcpy(qy,hy.data(),hy.size()*sizeof(double),cudaMemcpyHostToDevice));ck(cudaMemcpy(qw,hw.data(),hw.size()*sizeof(double),cudaMemcpyHostToDevice));
 fixture.in.quadrature={qy,qw,int(hy.size()),4e-7,1.6e-9};
 std::vector<In> hi(count,fixture.in);std::vector<Out> ref(count);std::vector<Out> ho(count);std::vector<WallStatus> hs(count);std::vector<int> hk(count);std::vector<Diag> hd(count);std::vector<double> hp(count*nodes*9);
 auto in=dev<In>(count);auto c=dev<WallModelConfig<double>>(1);auto w=dev<WallWorkspace<double,2,C>>(workers);auto o=dev<Out>(count);auto s=dev<WallStatus>(count);auto ok=dev<int>(count);auto profile=dev<double>(count*nodes*9);auto d=dev<Diag>(count);auto species=dev<ugkwp::SpeciesThermoData<double>>(2);auto coef=dev<double>(6);
 ck(cudaMemcpy(species,fixture.sp,sizeof(fixture.sp),cudaMemcpyHostToDevice));ck(cudaMemcpy(coef,fixture.coef,sizeof(fixture.coef),cudaMemcpyHostToDevice));
 for(int i=0;i<count;++i){double masses[5]={0,1e-9,.001,.01,.1};hi[i].massFlux[0]=.7*masses[i];hi[i].massFlux[1]=.3*masses[i];
 In cpu=hi[i];cpu.quadrature.distance=hy.data();cpu.quadrature.volumeWeight=hw.data();WallWorkspace<double,2,C> scratch;WallStatus cs;assert(evaluateWallModel(cpu,config,scratch,ref[i],cs));assert(ref[i].residual<=1.01e-8);hi[i].model.thermo.species=species;hi[i].model.thermo.coefficients=coef;hi[i].diagnostics=d+i;hi[i].model.thermo.diagnostics=d+i;hi[i].diagnosticEntity=100+i;if(mode==2&&i==1)hi[i].temperature=-1;}
 ck(cudaMemcpy(in,hi.data(),sizeof(In)*count,cudaMemcpyHostToDevice));ck(cudaMemcpy(c,&config,sizeof(config),cudaMemcpyHostToDevice));ck(cudaMemset(w,0,sizeof(*w)*workers));ck(cudaMemset(o,0,sizeof(Out)*count));ck(cudaMemset(s,0,sizeof(WallStatus)*count));ck(cudaMemset(d,0,sizeof(Diag)*count));ck(cudaMemset(profile,0,sizeof(double)*hp.size()));
 pool<C><<<workers+1,32>>>(in,c,w,o,s,ok,profile,count,workers);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
 ck(cudaMemcpy(ho.data(),o,sizeof(Out)*count,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hs.data(),s,sizeof(WallStatus)*count,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hk.data(),ok,sizeof(int)*count,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hd.data(),d,sizeof(Diag)*count,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hp.data(),profile,sizeof(double)*hp.size(),cudaMemcpyDeviceToHost));
 for(int i=0;i<count;++i){assert(hk[i]==!(mode==2&&i==1));assert(bool(hd[i].fatalClaim)==(mode==2&&i==1));
 printf("C=%d n=%d mode=%d slot=%d ok=%d iter=%d residual=%a heat=%a fatal=%u site=%u",C,nodes,mode,i,hk[i],ho[i].iterations,ho[i].residual,ho[i].conductiveHeatFlux,hd[i].fatalClaim,hd[i].firstFatal.sourceSiteId);
 for(auto& a:hd[i].warnings)printf(" warning=%u",a.count);
 if(mode!=2||i!=1){assert(std::isfinite(ho[i].ownerOmega)&&ho[i].ownerOmega>0);assert(std::abs(ho[i].traction[0]-ref[i].traction[0])<1e-8*(1+std::abs(ref[i].traction[0])));assert(std::abs(ho[i].ownerOmega-ref[i].ownerOmega)<1e-8*(1+std::abs(ref[i].ownerOmega)));assert(std::isfinite(ho[i].ownerKBalanceResidual));if(mode==0)assert(ho[i].residual<1.01e-8);}
 printf(" omega=%a K=%a D=%a defect=%a\n",ho[i].ownerOmega,ho[i].ownerKContent,ho[i].ownerKDestruction,ho[i].ownerKBalanceResidual);}
 for(void* p:{(void*)in,(void*)c,(void*)w,(void*)o,(void*)s,(void*)ok,(void*)profile,(void*)d,(void*)species,(void*)coef,(void*)qy,(void*)qw})ck(cudaFree(p));
}
int main(){run<24>(24,0);run<48>(48,0);run<128>(96,0);run<128>(128,0);puts("PASS blowing continuation across 20 inputs, CPU/CUDA agreement");}
