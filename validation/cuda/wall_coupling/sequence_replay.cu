// Same-input serial wall-operator timing; no FV/I/O/capture work inside the timer.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <cmath>
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/WallCudaTeam.cuh"
#include <cuda_runtime.h>
#include "common/gasWall/WallCoherentSolver.H"
#include "common/gasWall/reconstruction/ConservativeProfile.H"
#include "SequenceRecord.H"
#include "PrismaticOwnerQuadrature.H"
using namespace ugkwp::gaswall;using namespace ugkwp::gaswall::reconstruction;
static_assert(sizeof(SequenceRecord)==1216,"captured input ABI");
struct Stats{int iterations=0,residualEvaluations=0,factorizations=0;double physicalResidual=0,momentResidual=0,seconds=0;};
using W=WallWorkspace<double,2,24>;using Cache=coherent::Cache<24>;
constexpr int sourcePanels=1;
void bind(const SequenceRecord&r,Input&in,const Config&cfg,WallGeometryDescriptor<double>&d){auto thermo=in.model.thermo;auto mechanism=in.model.mechanism;in=r.input;in.model.thermo=thermo;in.model.mechanism=mechanism;in.diagnostics=nullptr;in.model.thermo.diagnostics=nullptr;d=descriptorForPrism(cfg,r.hi[1],r.hi[0],in.ownerDistance,in.quadrature.volume);in.quadrature=d.quadrature();}
UGKWP_GAS_HD void segments(const SequenceRecord&r,Profile&p){for(int z=0;z<2;++z){Segment a;a.lo=r.lo[z];a.hi=r.hi[z];a.normalLo=r.normalLo[z];a.normalHi=r.normalHi[z];a.algebraicOmega=r.algebraic[z];a.hasMacroGradients=true;for(int q=0;q<9;++q)a.target[q]=r.target[z][q];for(int j=0;j<3;++j){a.macroGradients.T[j]=r.T[z][j];a.macroGradients.k[j]=r.k[z][j];a.macroGradients.omega[j]=r.omega[z][j];for(int k=0;k<3;++k)a.macroGradients.U[k][j]=r.U[z][k][j];for(int k=0;k<2;++k)a.macroGradients.Y[k][j]=r.Y[z][k][j];}p.segments.push_back(a);}}
struct Row{int owner,step,stage,ok,actualOk,fitOk,finite,iterations,F,J;double residual,moment,rawQ,rawTau,fitQ,fitTau,source0,source1,checksum,seconds,bvpSeconds,fitSeconds,sourceFaceSeconds,wallJ0,wallJ1,matchingJ0,matchingJ1,heatBalance,kBalance,profileJ0,profileJ1;};

void ck(cudaError_t e){if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<'\n';std::exit(2);}}
template<class T>T* managed(int n=1){T*p;ck(cudaMallocManaged(&p,sizeof(T)*n));for(int i=0;i<n;++i)new(p+i)T{};return p;}
__global__ __launch_bounds__(32) void replay(Input*in,Config*cfg,W*w,Cache*cache,coherent::Scratch<24>*scratch,
  ProfileScratch*ps,SequenceRecord*record,Row*row,WallOutput<double,2>*output,Stats*stats){
 detail::CudaWallTeam team;
 coherent::Engine<24,detail::CudaWallTeam>engine(*scratch,team);
 auto&out=*output;auto&st=*stats;
 if(team.leader()){out={};st={};}team.sync();
 // The engine's leader publication is consumed during collective refresh;
 // output/stats must be visible to the entire team throughout the solve.
 const bool ok=engine.solve(*in,*cfg,*w,*cache,out,st);
 team.sync();
 Profile p(*in,*cfg,*w,ps->transformSlope,team);p.setWallFlux(out);segments(*record,p);p.reconstructInternalTraces();
 const bool fitted=p.fitTeam(*ps,team);
 double source[2][3];
 for(int s=0;s<2;++s){auto a=p.sourceTeam(s,team,sourcePanels);for(int j=0;j<3;++j)source[s][j]=a[j];}
 if(team.leader()){
  auto&z=*row;z.owner=record->owner;z.step=record->step;z.stage=record->stage;
  z.ok=ok;z.actualOk=ok&&std::isfinite(st.physicalResidual)&&st.physicalResidual<=cfg->absoluteTolerance+cfg->relativeTolerance;
  z.fitOk=fitted;z.moment=p.maxResidual;z.residual=st.physicalResidual;z.iterations=st.iterations;z.F=st.residualEvaluations;z.J=st.factorizations;
  auto interfaceFlux=p.flux(record->hi[0]);auto interfacePrimitive=p.physicalPrimitive(record->hi[0],-1);z.profileJ0=interfaceFlux[7]-interfaceFlux[0]*interfacePrimitive.Y[0];z.profileJ1=interfaceFlux[8]-interfaceFlux[0]*interfacePrimitive.Y[1];z.rawQ=out.conductiveHeatFlux;z.rawTau=out.traction[0];z.wallJ0=out.wallSpeciesFlux[0];z.wallJ1=out.wallSpeciesFlux[1];z.matchingJ0=out.matchingSpeciesFlux[0];z.matchingJ1=out.matchingSpeciesFlux[1];z.heatBalance=out.heatBalanceResidual;z.kBalance=out.ownerKBalanceResidual;
  for(int s=0;s<2;++s){auto b=p.baseAverage(p.segments[s]);for(double x:b)z.checksum+=x;for(double x:source[s])z.checksum+=x;
   if(s==0)z.source0=source[s][0]/(record->hi[s]-record->lo[s]);else z.source1=source[s][0]/(record->hi[s]-record->lo[s]);}
  for(double y:{0.,record->hi[0],record->hi[1]}){auto adv=p.inviscid(p.base(y));auto diff=p.diffusiveFlux(y);
   for(int c=0;c<9;++c)z.checksum+=adv[c]+diff[c];if(y==0){z.fitQ=diff[4];z.fitTau=-diff[1];}}
  z.finite=std::isfinite(z.checksum)&&std::isfinite(z.fitQ)&&std::isfinite(z.source0)&&std::isfinite(z.source1);
  for(int node=0;node<cfg->nodes;++node)for(int c=0;c<6;++c)z.finite=z.finite&&std::isfinite(w->state[node][c])&&std::isfinite(w->residual[node][c]);
 }
}
int main(int argc,char**argv){if(argc<3||argc>4)return 2;const int limit=argc==4?std::atoi(argv[3]):2000;
 auto in=managed<Input>();auto cfg=managed<Config>();auto w=managed<W>();auto cache=managed<Cache>();auto scratch=managed<coherent::Scratch<24>>();
 auto ps=managed<ProfileScratch>();auto record=managed<SequenceRecord>();auto row=managed<Row>();auto output=managed<WallOutput<double,2>>();auto stats=managed<Stats>();
 auto species=managed<ugkwp::SpeciesThermoData<double>>(2);auto coefficients=managed<double>(6);
 auto distance=managed<double>(512);auto weight=managed<double>(512);
 CapturedSstWall mat;std::memcpy(species,mat.species,sizeof(mat.species));std::memcpy(coefficients,mat.coefficients,sizeof(mat.coefficients));
 *cfg=mat.config;cfg->nodes=24;cfg->coherentDiscretization=true;
 std::ofstream f(argv[2]);f<<std::setprecision(17)<<"owner,step,stage,ok,actualOk,fitOk,finite,residual,moment,iterations,F,J,rawQ,rawTau,fitQ,fitTau,source0,source1,checksum,seconds,wallJ0,wallJ1,matchingJ0,matchingJ1,heatBalance,kBalance,profileJ0,profileJ1\n";
 int failures=0,calls=0;double seconds=0,maxResidual=0,maxMoment=0;
 for(int owner:{0,7782}){
  *w={};*cache={};*scratch={};
  std::ifstream file(std::string(argv[1])+"/sequence-owner"+std::to_string(owner)+".bin",std::ios::binary);if(!file)return 3;
  SequenceRecord r;int count=0;
  while(count++<limit&&file.read(reinterpret_cast<char*>(&r),sizeof r)){
   *record=r;WallGeometryDescriptor<double>d;auto host=mat.input;bind(r,host,*cfg,d);
   if(d.distance.size()>512)return 4;*in=host;in->model.thermo.species=species;in->model.thermo.coefficients=coefficients;
   for(size_t j=0;j<d.distance.size();++j){distance[j]=d.distance[j];weight[j]=d.volumeWeight[j];}
   in->quadrature.distance=distance;in->quadrature.volumeWeight=weight;*row={};
   auto begin=std::chrono::steady_clock::now();replay<<<1,32>>>(in,cfg,w,cache,scratch,ps,record,row,output,stats);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
   auto&z=*row;z.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
   f<<z.owner<<','<<z.step<<','<<z.stage<<','<<z.ok<<','<<z.actualOk<<','<<z.fitOk<<','<<z.finite<<','<<z.residual<<','<<z.moment<<','<<z.iterations<<','<<z.F<<','<<z.J<<','<<z.rawQ<<','<<z.rawTau<<','<<z.fitQ<<','<<z.fitTau<<','<<z.source0<<','<<z.source1<<','<<z.checksum<<','<<z.seconds<<','<<z.wallJ0<<','<<z.wallJ1<<','<<z.matchingJ0<<','<<z.matchingJ1<<','<<z.heatBalance<<','<<z.kBalance<<','<<z.profileJ0<<','<<z.profileJ1<<'\n';
   failures+=!z.actualOk||!z.fitOk||!z.finite||z.moment>1e-9||z.wallJ0!=0||z.wallJ1!=0;seconds+=z.seconds;maxResidual=std::max(maxResidual,z.residual);maxMoment=std::max(maxMoment,z.moment);++calls;
   if(count%100==0){f.flush();std::cout<<"owner="<<owner<<" count="<<count<<" failures="<<failures<<std::endl;}
  }
 }
 std::cout<<std::setprecision(17)<<"calls="<<calls<<" failures="<<failures<<" maxResidual="<<maxResidual<<" maxMoment="<<maxMoment<<" seconds="<<seconds<<std::endl;
 ck(cudaFree(in));ck(cudaFree(cfg));ck(cudaFree(w));ck(cudaFree(cache));ck(cudaFree(scratch));ck(cudaFree(ps));ck(cudaFree(record));ck(cudaFree(row));ck(cudaFree(output));ck(cudaFree(stats));ck(cudaFree(species));ck(cudaFree(coefficients));ck(cudaFree(distance));ck(cudaFree(weight));
 return failures?1:0;
}
