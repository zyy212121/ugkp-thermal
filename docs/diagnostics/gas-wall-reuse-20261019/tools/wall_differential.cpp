#include <initializer_list>
#include "common/gasWall/WallModel.H"
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "validation/tests/gas_wall/fixtures/wall_composition_cases.H"
#include <cstdio>
#include <cstring>
#include <cstdint>
using namespace ugkwp;using namespace ugkwp::gaswall;
uint64_t h=1469598103934665603ULL;
void add(double x){uint64_t u;std::memcpy(&u,&x,8);h=(h^u)*1099511628211ULL;}
void record(const ugkp::diagnostics::Record&r){std::printf(" %u:%u:%u:%u:%lld:%d:%a:%a:%a",unsigned(r.code),r.sourceSiteId,r.operatorId,r.stageId,(long long)r.entityIndex,r.component,r.observed,r.reference,r.corrected);}
template<int N,int M>void summarize(int variant,int nodes,bool good,WallStatus&s,WallOutput<double,N>&o,WallWorkspace<double,N,M>&w,ugkp::diagnostics::DeviceContext&d){
 h=1469598103934665603ULL;
 if(good){add(o.traceTemperature);add(o.traceDensity);add(o.conductiveHeatFlux);add(o.ownerOmega);add(o.integratedKSource);add(o.wallKFlux);add(o.residual);add(o.iterations);
 for(int j=0;j<3;++j){add(o.traction[j]);add(o.traceVelocity[j]);}
 for(int j=0;j<N;++j){add(o.traceMassFraction[j]);add(o.wallSpeciesFlux[j]);add(o.matchingSpeciesFlux[j]);add(o.reactionIntegral[j]);add(o.speciesBalanceResidual[j]);add(o.chemistryMismatch[j]);}
 for(int i=0;i<nodes;++i){add(w.y[i]);for(int j=0;j<N+4;++j){add(w.state[i][j]);add(w.residual[i][j]);}}}
 std::printf("N=%d M=%d nodes=%d case=%d ok=%d status=%d:%d:%d:%a hash=%llx fatal=%u",N,M,nodes,variant,good,int(s.code),s.node,s.iteration,s.residual,(unsigned long long)h,d.fatalClaim);
 if(d.fatalClaim)record(d.firstFatal);
 for(auto &q:d.warnings){std::printf(" warnings=%u/%u",q.count,q.overflow);if(q.count)record(q.sample);}std::puts("");
}
template<int N,int M>void run(int nodes,int variant){
 SpeciesThermoData<double> species[N]{};double coefficients[14*N]{};
 WallInput<double,N> in;WallModelConfig<double> c;WallWorkspace<double,N,M> w{};WallOutput<double,N> out;WallStatus status;ugkp::diagnostics::DeviceContext diagnostics{};
 for(int j=0;j<N;++j){species[j].model=SpeciesThermoModel::LinearCp;species[j].coefficientOffset=3*j;species[j].molarMass=.028+.002*j;species[j].minTemperature=10;species[j].maxTemperature=2000;coefficients[3*j]=1000+3*j;coefficients[3*j+1]=.01;coefficients[3*j+2]=j*100.;in.matching.massFraction[j]=1./N;in.model.diffusivity[j]=1e-5;}
 in.model.thermo.species=species;in.model.thermo.coefficients=coefficients;in.model.thermo.coefficientCount=3*N;in.model.mode=GasMode::MixtureFrozen;in.model.viscosity=1.7e-5;in.model.conductivity=.024;in.model.molecularPrandtl=.72;
 in.diagnostics=&diagnostics;in.model.thermo.diagnostics=&diagnostics;in.diagnosticEntity=71;
 in.pressure=1e5;in.temperature=300;in.matching.temperature=310;in.normal[1]=1;in.matchingDistance=.001;in.ownerDistance=.0002;in.matching.k=1;in.matching.omega=1e4;in.matching.velocity[0]=10;in.matching.velocity[2]=3;
 c.nodes=nodes;c.enableSst=true;c.maxIterations=2;
 switch(variant){case 1:in.matching.massFraction[0]=1.2;break;case 2:for(int j=0;j<N;++j)in.matching.massFraction[j]=.8;break;case 3:coefficients[0]=-100;break;case 4:coefficients[2]=1e309;break;case 5:in.model.thermo.coefficientCount=3*N-1;break;case 6:species[0].molarMass=0;break;case 7:in.model.viscosity=-1e-5;break;case 8:in.matching.k=1e308;break;case 9:c.enableSst=false;break;case 10:c.stretch=0;break;case 11:in.matching.temperature=-1;break;}
 if(variant>=12){
 for(int j=0;j<14*N;++j)coefficients[j]=0;
 for(int j=0;j<N;++j){species[j].model=SpeciesThermoModel::NASA7;species[j].coefficientOffset=14*j;species[j].midTemperature=305;coefficients[14*j]=coefficients[14*j+7]=3.5;}
 in.model.thermo.coefficientCount=14*N-(variant==13?2:variant==14?1:0);
}
 bool ok=evaluateReactingWallLayer(in,c,w,out,status);summarize<N,M>(variant,nodes,ok,status,out,w,diagnostics);
}
void chemistry(int variant,bool sst,bool reversible){
 CompositionWallCase m;m.in.integrateOwnerKSource=false;m.in.matching.k=.5;m.in.matching.omega=200;m.in.model.mode=GasMode::MixtureChemistry;m.rx[0].reversible=reversible;
 WallModelConfig<double>c;c.nodes=12;c.maxIterations=3;c.enableSst=sst;WallWorkspace<double,2,48>w{};WallOutput<double,2>o;WallStatus s;ugkp::diagnostics::DeviceContext d{};m.in.diagnostics=&d;m.in.model.thermo.diagnostics=&d;
 if(variant==1){m.in.massFlux[0]=.0007;m.in.massFlux[1]=.0003;}
 if(variant==2)m.in.matching.massFraction[1]=1.2;
 if(variant==3)m.in.matching.massFraction[1]=-.1;
 if(variant==4)m.rx[0].highRate.preExponential=1e308;
 if(variant==5)m.sp[0].hasEntropyReference=false;
 if(variant==6)m.in.model.mechanism.reactants=nullptr;
 if(variant==7)m.in.integrateOwnerKSource=true;
 bool ok=evaluateReactingWallLayer(m.in,c,w,o,s);summarize<2,48>(100+variant+20*sst+40*reversible,c.nodes,ok,s,o,w,d);
}
int main(){for(int i=0;i<8;++i)for(bool sst:{false,true})for(bool reversible:{false,true})chemistry(i,sst,reversible);
for(int i=0;i<15;++i){run<1,24>(12,i);run<2,24>(12,i);run<3,48>(25,i);run<7,128>(49,i);}
 run<1,24>(2,0);run<3,48>(48,0);run<3,128>(128,0);
 CapturedSstWall m;WallWorkspace<double,2,24>w{};WallOutput<double,2>o;WallStatus s;ugkp::diagnostics::DeviceContext d{};m.input.diagnostics=&d;m.input.model.thermo.diagnostics=&d;
 bool ok=evaluateReactingWallLayer(m.input,m.config,w,o,s);summarize<2,24>(99,24,ok,s,o,w,d);
}