inline int probe_line_calls=0,probe_line_reject=0,probe_line_invalid=0,probe_jac_calls=0,probe_jac_invalid=0,probe_jac_retry=0,probe_seed_failed=0,probe_seed_restored=0;

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

#include <cstdlib>
#include <iomanip>
#include <algorithm>
void run(Model& m,WallModelConfig<double>& c,int index){
 probe_line_calls=probe_line_reject=probe_line_invalid=probe_jac_calls=probe_jac_invalid=probe_jac_retry=probe_seed_failed=probe_seed_restored=0;
#if DIAGNOSTICS
 ugkp::diagnostics::DeviceContext diagnostic{};m.in.diagnostics=&diagnostic;m.in.model.thermo.diagnostics=&diagnostic;
#endif
 WallWorkspace<double,2,128>w;WallOutput<double,2>o;WallStatus st;
 const bool ok=evaluateWallModel(m.in,c,w,o,st);
 double minZ=1e300,minT=1e300,minY=1e300,maxY=-1e300;bool stateFinite=true;
 for(int i=0;i<c.nodes;++i){minZ=std::min(minZ,w.state[i][5]);minT=std::min(minT,m.in.temperature+w.state[i][2]);minY=std::min(minY,w.state[i][3]);maxY=std::max(maxY,w.state[i][3]);for(int j=0;j<6;++j)stateFinite=stateFinite&&std::isfinite(w.state[i][j]);}
 int fatal=0,site=0,node=0,warnClip=0,warnAccuracy=0;double observed=0;
#if DIAGNOSTICS
 fatal=diagnostic.fatalClaim;site=diagnostic.firstFatal.sourceSiteId;node=diagnostic.firstFatal.stageId;observed=diagnostic.firstFatal.observed;warnClip=diagnostic.warnings[2].count;warnAccuracy=diagnostic.warnings[1].count;
 ugkp::diagnostics::DeviceContext recheckDiagnostic{};m.in.diagnostics=&recheckDiagnostic;m.in.model.thermo.diagnostics=&recheckDiagnostic;
#endif
 detail::LayerContext<double,2>ctx;double norm=std::numeric_limits<double>::quiet_NaN();const bool residualOk=detail::makeContext(m.in,c,ctx)&&detail::allResidual(m.in,c,ctx,w.state,w.y,w.trial,norm);
 const bool converged=ok&&!fatal&&residualOk&&std::isfinite(norm)&&norm<=c.absoluteTolerance+c.relativeTolerance;
 std::cout<<std::setprecision(17)<<index<<','<<c.enableSst<<','<<ok<<','<<converged<<','<<int(st.code)<<','<<o.iterations<<','<<o.residual<<','<<residualOk<<','<<norm<<','<<fatal<<','<<site<<','<<node<<','<<observed<<','<<warnClip<<','<<warnAccuracy<<','<<probe_line_calls<<','<<probe_line_reject<<','<<probe_line_invalid<<','<<probe_jac_calls<<','<<probe_jac_invalid<<','<<probe_jac_retry<<','<<probe_seed_failed<<','<<probe_seed_restored<<','<<stateFinite<<','<<minZ<<','<<minT<<','<<minY<<','<<maxY<<','<<o.conductiveHeatFlux<<','<<o.traction[0]<<','<<o.ownerOmega<<','<<o.traceMassFraction[0]<<','<<o.speciesBalanceResidual[0]<<'\n';
}
int main(int argc,char**argv){int count=argc>1?std::atoi(argv[1]):161;unsigned seed=4321;
 std::cout<<"case,sst,returned,converged,status,iterations,published_residual,residual_ok,rechecked_residual,fatal,site,node,observed,warn_clip,warn_accuracy,line_calls,line_reject,line_invalid,jac_calls,jac_invalid,jac_retry,seed_failed,seed_restored,state_finite,min_z,min_T,min_Y,max_Y,heat,shear,omega,wall_Y,balance\n";
 for(int index=0;index<count;++index){auto random=[&](){seed=1664525*seed+1013904223;return double(seed)/4294967296.;};Model m;m.in.integrateOwnerKSource=false;
 m.in.matchingDistance=std::pow(10.,-4+3*random());m.in.ownerDistance=.4*m.in.matchingDistance;
 m.in.temperature=250+800*random();m.in.matching.temperature=300+1500*random();m.in.matching.velocity[0]=std::pow(10.,-1+4*random());
 m.in.matching.k=std::pow(10.,-10+12*random());m.in.matching.omega=std::pow(10.,-2+7*random());
 const double mass=std::pow(10.,-8+8*random());m.in.massFlux[0]=.7*mass;m.in.massFlux[1]=.3*mass;
 WallModelConfig<double> c;c.nodes=16;c.maxIterations=80;c.relativeTolerance=1e-8;c.enableSst=index%2==0;
 if(!c.enableSst){m.in.model.mode=GasMode::MixtureChemistry;m.coef[2]=1e5+3e5*random();m.rx[0].highRate.preExponential=std::pow(10.,3*random());m.in.model.diffusivity[0]=1e-4;m.in.model.diffusivity[1]=3e-4;}
 std::cerr<<std::setprecision(17)<<index<<','<<c.enableSst<<','<<m.in.matchingDistance<<','<<m.in.ownerDistance<<','<<m.in.temperature<<','<<m.in.matching.temperature<<','<<m.in.matching.velocity[0]<<','<<m.in.matching.k<<','<<m.in.matching.omega<<','<<mass<<','<<m.coef[2]<<','<<m.rx[0].highRate.preExponential<<'\n';
 run(m,c,index);
 }
 for(int k=0;k<3;++k){Model m;m.in.integrateOwnerKSource=false;m.in.matching.velocity[0]=30;m.in.temperature=m.in.matching.temperature=500;m.in.matching.k=.5;m.in.matching.omega=200;const double mass=k==0?0:(k==1?1e-9:1e-3);m.in.massFlux[0]=.7*mass;m.in.massFlux[1]=.3*mass;WallModelConfig<double> c;c.enableSst=true;c.nodes=32;c.maxIterations=100;c.relativeTolerance=1e-8;run(m,c,10000+k);}
 {Model m;m.in.integrateOwnerKSource=false;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=0;m.in.matching.k=.5;m.in.matching.omega=200;m.in.massFlux[0]=.0007;m.in.massFlux[1]=.0003;WallModelConfig<double> c;c.enableSst=true;c.nodes=3;run(m,c,10003);}
}
