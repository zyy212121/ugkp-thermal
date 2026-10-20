
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

#include <iomanip>
int main(){std::cout<<"nodes,mass,ok,iterations,residual,tau,seed_sweeps,interval_calls,residual_calls,newton_calls,seed_us,total_us\n"<<std::setprecision(17);
 for(int n:{24,48,96,128})for(double mass:{0.,1e-9,.001,.01,.1}){Model m;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=30;m.in.matching.k=.5;m.in.matching.omega=200;m.in.massFlux[0]=.7*mass;m.in.massFlux[1]=.3*mass;double distances[2]={.002,.006},weights[2]={2e-7,2e-7};m.in.quadrature={distances,weights,2,4e-7,1.6e-9};WallModelConfig<double> c;c.enableSst=true;c.nodes=n;c.maxIterations=100;c.relativeTolerance=1e-8;static WallWorkspace<double,2,128>w;WallOutput<double,2>o;WallStatus st;initprobe::stats={};auto t=std::chrono::steady_clock::now();bool ok=evaluateWallModel(m.in,c,w,o,st);double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();auto&a=initprobe::stats;
 std::cout<<n<<','<<mass<<','<<ok<<','<<o.iterations<<','<<o.residual<<','<<o.traction[0]<<','<<a.seed<<','<<a.interval<<','<<a.residual<<','<<a.newton<<','<<a.seedTime<<','<<us<<'\n';} }
