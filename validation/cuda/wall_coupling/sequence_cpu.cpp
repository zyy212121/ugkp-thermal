// Same-input serial wall-operator timing; no FV/I/O/capture work inside the timer.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/WallCoherentSolver.H"
#include "common/gasWall/reconstruction/ConservativeProfile.H"
#include "validation/cuda/wall_coupling/SequenceRecord.H"
#include "validation/cuda/wall_coupling/PrismaticOwnerQuadrature.H"
using namespace ugkwp::gaswall;using namespace ugkwp::gaswall::reconstruction;
static_assert(sizeof(SequenceRecord)==1216,"captured input ABI");
struct Stats{int iterations=0,residualEvaluations=0,factorizations=0;double physicalResidual=0,momentResidual=0,seconds=0;};
using W=WallWorkspace<double,2,24>;using Cache=coherent::Cache<24>;
constexpr int sourcePanels=1;
void bind(const SequenceRecord&r,Input&in,const Config&cfg,WallGeometryDescriptor<double>&d){auto thermo=in.model.thermo;auto mechanism=in.model.mechanism;in=r.input;in.model.thermo=thermo;in.model.mechanism=mechanism;in.diagnostics=nullptr;in.model.thermo.diagnostics=nullptr;d=descriptorForPrism(cfg,r.hi[1],r.hi[0],in.ownerDistance,in.quadrature.volume);in.quadrature=d.quadrature();}
void segments(const SequenceRecord&r,Profile&p){for(int z=0;z<2;++z){Segment a;a.lo=r.lo[z];a.hi=r.hi[z];a.normalLo=r.normalLo[z];a.normalHi=r.normalHi[z];a.algebraicOmega=r.algebraic[z];a.hasMacroGradients=true;for(int q=0;q<9;++q)a.target[q]=r.target[z][q];for(int j=0;j<3;++j){a.macroGradients.T[j]=r.T[z][j];a.macroGradients.k[j]=r.k[z][j];a.macroGradients.omega[j]=r.omega[z][j];for(int k=0;k<3;++k)a.macroGradients.U[k][j]=r.U[z][k][j];for(int k=0;k<2;++k)a.macroGradients.Y[k][j]=r.Y[z][k][j];}p.segments.push_back(a);}}
struct Row{int owner,step,stage,ok,actualOk,fitOk,finite,iterations,F,J;double residual,moment,rawQ,rawTau,fitQ,fitTau,source0,source1,checksum,seconds,bvpSeconds,fitSeconds,sourceFaceSeconds,wallJ0,wallJ1,matchingJ0,matchingJ1,heatBalance,kBalance,profileJ0,profileJ1;};
int main(int argc,char**argv){if(argc!=3)return 2;std::string input=argv[1],output=argv[2];std::vector<Row>rows;rows.reserve(4000);std::cout<<std::setprecision(17)<<"recordBytes="<<sizeof(SequenceRecord)<<" inputBytes="<<sizeof(Input)<<" configBytes="<<sizeof(Config)<<" sourcePanels="<<sourcePanels<<"\n";
 for(int owner:{0,7782}){std::ifstream file(input+"/sequence-owner"+std::to_string(owner)+".bin",std::ios::binary);std::vector<SequenceRecord>records;SequenceRecord record;while(file.read(reinterpret_cast<char*>(&record),sizeof record))records.push_back(record);std::cout<<"owner="<<owner<<" records="<<records.size()<<"\n";
  CapturedSstWall mat;auto in=mat.input;auto cfg=mat.config;cfg.nodes=24;cfg.coherentDiscretization=true;W w{};Cache cache{};WallGeometryDescriptor<double>d;static ProfileScratch scratch;std::cout<<"nodes="<<cfg.nodes<<" maxIterations="<<cfg.maxIterations<<" abs="<<cfg.absoluteTolerance<<" rel="<<cfg.relativeTolerance<<"\n";
  for(const auto&r:records){bind(r,in,cfg,d);WallOutput<double,2>out;Stats st;Row z{};z.owner=owner;z.step=r.step;z.stage=r.stage;auto begin=std::chrono::steady_clock::now();
#ifdef BASELINE_HEAD
   bool ok=cache.solve(in,cfg,w,out,st);
#else
   static coherent::Scratch<24> bvpScratch;coherent::Engine<24> engine(bvpScratch);bool ok=engine.solve(in,cfg,w,cache,out,st);
#endif
   auto solved=std::chrono::steady_clock::now();std::chrono::steady_clock::time_point fitted;
   {Profile p(in,cfg,w);p.setWallFlux(out);segments(r,p);p.reconstructInternalTraces();z.fitOk=p.fitTeam(scratch,detail::SerialWallTeam{});z.moment=p.maxResidual;auto interfaceFlux=p.flux(r.hi[0]);auto interfacePrimitive=p.physicalPrimitive(r.hi[0],-1);z.profileJ0=interfaceFlux[7]-interfaceFlux[0]*interfacePrimitive.Y[0];z.profileJ1=interfaceFlux[8]-interfaceFlux[0]*interfacePrimitive.Y[1];fitted=std::chrono::steady_clock::now();
    for(int s=0;s<2;++s){auto base=p.baseAverage(p.segments[s]);for(double x:base)z.checksum+=x;auto source=p.sourceTeam(s,detail::SerialWallTeam{},sourcePanels);for(double x:source)z.checksum+=x;double value=source[0]/(r.hi[s]-r.lo[s]);if(s==0)z.source0=value;else z.source1=value;}
    for(double y:{0.,r.hi[0],r.hi[1]}){auto adv=p.inviscid(p.base(y));auto diff=p.diffusiveFlux(y);for(int c=0;c<9;++c)z.checksum+=adv[c]+diff[c];if(y==0){z.fitQ=diff[4];z.fitTau=-diff[1];}}
   }
#ifdef BASELINE_HEAD
   // Same accepted-step history policy as exact HEAD adapter: not between RK stages.
   if(r.stage==2)cache.commitInterval(cfg.nodes);
#endif
   auto end=std::chrono::steady_clock::now();z.seconds=std::chrono::duration<double>(end-begin).count();z.bvpSeconds=std::chrono::duration<double>(solved-begin).count();z.fitSeconds=std::chrono::duration<double>(fitted-solved).count();z.sourceFaceSeconds=std::chrono::duration<double>(end-fitted).count();z.ok=ok;z.actualOk=ok&&std::isfinite(st.physicalResidual)&&st.physicalResidual<=cfg.absoluteTolerance+cfg.relativeTolerance;z.finite=std::isfinite(z.checksum)&&std::isfinite(z.fitQ)&&std::isfinite(z.source0)&&std::isfinite(z.source1);for(int node=0;node<cfg.nodes;++node)for(int c=0;c<6;++c)z.finite=z.finite&&std::isfinite(w.state[node][c])&&std::isfinite(w.residual[node][c]);z.residual=st.physicalResidual;z.iterations=st.iterations;z.F=st.residualEvaluations;z.J=st.factorizations;z.rawQ=out.conductiveHeatFlux;z.rawTau=out.traction[0];z.wallJ0=out.wallSpeciesFlux[0];z.wallJ1=out.wallSpeciesFlux[1];z.matchingJ0=out.matchingSpeciesFlux[0];z.matchingJ1=out.matchingSpeciesFlux[1];z.heatBalance=out.heatBalanceResidual;z.kBalance=out.ownerKBalanceResidual;rows.push_back(z);
  }
 }
 std::ofstream f(output);f<<std::setprecision(17)<<"owner,step,stage,ok,actualOk,fitOk,finite,residual,moment,iterations,F,J,rawQ,rawTau,fitQ,fitTau,source0,source1,checksum,seconds,bvpSeconds,fitSeconds,sourceFaceSeconds,wallJ0,wallJ1,matchingJ0,matchingJ1,heatBalance,kBalance,profileJ0,profileJ1\n";int failures=0,fitFailures=0,nonfinite=0;double seconds=0,maxResidual=0,maxMoment=0;for(const auto&r:rows){f<<r.owner<<','<<r.step<<','<<r.stage<<','<<r.ok<<','<<r.actualOk<<','<<r.fitOk<<','<<r.finite<<','<<r.residual<<','<<r.moment<<','<<r.iterations<<','<<r.F<<','<<r.J<<','<<r.rawQ<<','<<r.rawTau<<','<<r.fitQ<<','<<r.fitTau<<','<<r.source0<<','<<r.source1<<','<<r.checksum<<','<<r.seconds<<','<<r.bvpSeconds<<','<<r.fitSeconds<<','<<r.sourceFaceSeconds<<','<<r.wallJ0<<','<<r.wallJ1<<','<<r.matchingJ0<<','<<r.matchingJ1<<','<<r.heatBalance<<','<<r.kBalance<<','<<r.profileJ0<<','<<r.profileJ1<<'\n';failures+=!r.actualOk;fitFailures+=!r.fitOk;nonfinite+=!r.finite;seconds+=r.seconds;maxResidual=std::max(maxResidual,r.residual);maxMoment=std::max(maxMoment,r.moment);}std::cout<<"calls="<<rows.size()<<" failures="<<failures<<" fitFailures="<<fitFailures<<" nonfinite="<<nonfinite<<" maxResidual="<<maxResidual<<" maxMoment="<<maxMoment<<" seconds="<<seconds<<"\n";
return (failures||fitFailures||nonfinite)?1:0;
}
