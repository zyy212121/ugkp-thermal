// Independent physical arithmetic audit. Compile twice with
// -DUGKWP_GPU_REAL_BITS=32 / 64 and -I<repository root>.
// Thresholds below are fixed forward-error budgets, not solver tolerances.
#include "common/gasWall/WallModel.H"
#include "common/chemistry/mechanisms/GeneratedH2O2.H"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>
using namespace ugkwp;
using namespace ugkwp::gaswall;
namespace d=ugkwp::gaswall::detail;
using R=GpuReal;
using L=long double;

// Independent binomial expansion of the polynomial about the starting T.
// This avoids both endpoint subtraction and the production Gauss rule.
L cpIntegral(const R* c,L t,L delta){
 const int binomial[5][5]={{1},{1,1},{1,2,1},{1,3,3,1},{1,4,6,4,1}};
 L out=0;for(int j=0;j<5;++j)for(int k=0;k<=j;++k)
  out+=L(c[j])*binomial[j][k]*std::pow(t,j-k)*std::pow(delta,k+1)/L(k+1);
 return out;
}
L enthalpyPolynomial(const R* c,L t){
 L out=c[5];for(int j=0;j<5;++j)out+=L(c[j])*std::pow(t,j+1)/L(j+1);return out;
}
L branchMagnitude(const R* c,L t){
 L out=std::abs(L(c[5]));for(int j=0;j<5;++j)out+=std::abs(L(c[j])*std::pow(t,j+1)/L(j+1));return out;
}
void nasaAudit(){
 GeneratedH2O2<R> table;auto thermo=table.thermoView();
 L maxRelative=0,maxAbsolute=0,maxBudgetFraction=0;int count=0,naiveZeros=0;
 const std::pair<R,R> cases[]={{R(500),R(1e-8)},{R(500),R(-1e-8)},
  {R(999.99),R(.02)},{R(1000.01),R(-.02)},
  {R(1000),R(.0001)},{R(1000),R(1e-6)},
  {R(900),R(300)},{R(1100),R(-300)}};
 for(int s=0;s<10;++s)for(auto pair:cases){
  const R tw=pair.first,theta=pair.second;const auto& sp=thermo.species[s];
  const R* c=thermo.coefficients+sp.coefficientOffset;const L mid=sp.midTemperature;
  const bool wallHigh=tw>sp.midTemperature;
  // Preserve the branch convention of common speciesH: branch selection is
  // based on the rounded physical T, even for compensated sub-ulp theta.
  const bool targetHigh=R(tw+theta)>sp.midTemperature;
  const L gasR=L(universalGasConstant<R>())/sp.molarMass;
  const L jump=enthalpyPolynomial(c+7,mid)-enthalpyPolynomial(c,mid);
  L integral=0,termMagnitude=0,jumpBudget=0;
  if(wallHigh==targetHigh){integral=cpIntegral(c+(wallHigh?7:0),tw,theta);termMagnitude=std::abs(integral);}
  else{
   const L first=mid-L(tw),second=L(theta)-first;
   const L left=cpIntegral(c+(wallHigh?7:0),tw,first),right=cpIntegral(c+(targetHigh?7:0),mid,second);
   integral=left+right+(targetHigh?jump:-jump);
   termMagnitude=std::abs(left)+std::abs(right)+std::abs(jump);
   // Only immutable branch differences use at least double precision.
   jumpBudget=64*std::numeric_limits<double>::epsilon()*(branchMagnitude(c,mid)+branchMagnitude(c+7,mid));
  }
  const L ref=gasR*integral;
  const R cached=d::speciesEnthalpyBranchJump(s,thermo);
  const R value=d::speciesEnthalpyIncrement(s,tw,theta,thermo,cached);
  assert(value==d::speciesEnthalpyIncrement(s,tw,theta,thermo));
  const L error=std::abs(L(value)-ref);
  const L budget=gasR*(64*std::numeric_limits<R>::epsilon()*termMagnitude+jumpBudget);
  if(!(error<=budget)){std::cerr<<"NASA failure "<<table.speciesNames[s]<<" "<<tw<<" "<<theta<<" error "<<double(error)<<" budget "<<double(budget)<<"\n";std::abort();}
  maxRelative=std::max(maxRelative,error/std::abs(ref));maxAbsolute=std::max(maxAbsolute,error);maxBudgetFraction=std::max(maxBudgetFraction,error/budget);++count;
  if(tw==R(500)&&speciesH(s,R(tw+theta),thermo)-speciesH(s,tw,thermo)==0)++naiveZeros;
 }
 // Deliberately discontinuous, allowed NASA metadata: no continuity assumption.
 R synthetic[140];for(int j=0;j<140;++j)synthetic[j]=table.coefficients[j];
 synthetic[12]+=R(100);thermo.coefficients=synthetic;
 const L expected=L(universalGasConstant<R>())/thermo.species[0].molarMass*(cpIntegral(synthetic,900,100)+cpIntegral(synthetic+7,1000,100)+enthalpyPolynomial(synthetic+7,1000)-enthalpyPolynomial(synthetic,1000));
 const R got=d::speciesEnthalpyIncrement(0,R(900),R(200),thermo);
 assert(std::abs(L(got)-expected)<=64*std::numeric_limits<R>::epsilon()*std::abs(expected));
 std::cout<<"NASA bits="<<sizeof(R)*8<<" cases="<<count<<" max_relative="<<double(maxRelative)<<" max_absolute="<<double(maxAbsolute)<<" budget_fraction="<<double(maxBudgetFraction)<<" naive_tiny_zeros="<<naiveZeros<<"\n";
}

void frameGaugeAudit(){
 SpeciesThermoData<R> species[2];R coefficients[28]={1000,R(.2),100000,1500,R(.1),-200000};
 WallInput<R,2> in;for(int s=0;s<2;++s){species[s].model=SpeciesThermoModel::LinearCp;species[s].coefficientOffset=s*3;species[s].molarMass=s==0?R(.028):R(.032);species[s].minTemperature=200;species[s].maxTemperature=4000;species[s].referencePressure=101325;}
 in.model.thermo.species=species;in.model.thermo.coefficients=coefficients;in.model.thermo.coefficientCount=6;
 in.model.viscosity=2e-5;in.model.conductivity=.04;in.model.diffusivity[0]=1e-4;in.model.diffusivity[1]=3e-4;
 in.pressure=101325;in.temperature=in.matching.temperature=500;in.normal[1]=1;in.matchingDistance=.01;
 in.matching.massFraction[0]=.7;in.matching.massFraction[1]=.3;
 WallModelConfig<R> cfg;L maxGauge=0,maxPhysical=0;int count=0;
 for(R mass:{R(0),R(.001)})for(R shift:{R(0),R(1000)}){
  in.massFlux[0]=R(.7)*mass;in.massFlux[1]=R(.3)*mass;
  in.velocity[0]=shift;in.velocity[2]=-shift*R(.7);in.velocity[1]=R(.2);
  in.matching.velocity[0]=shift+30;in.matching.velocity[2]=in.velocity[2]+10;
  d::LayerContext<R,2> ctx;d::makeContext(in,cfg,ctx);
  R xa[6]={2,-3,R(1e-6),R(.2),0,1},xb[6]={5,6,R(3e-6),R(.6),0,1};
  d::LayerPoint<R,2> a,b;assert(d::pointState(in,cfg,ctx,xa,a));assert(d::pointState(in,cfg,ctx,xb,b));
  d::LayerFlux<R,2> flux;const R length=R(.01);assert(d::layerFlux(in,cfg,ctx,a,b,length,flux));
  const L temperature=L(in.temperature)+(L(xa[2])+xb[2])/2;
  L absolute=0,conditioned=0,sumSpecies=0;
  for(int s=0;s<2;++s){
   const L h=L(coefficients[3*s+2])+temperature*(L(coefficients[3*s])+L(.5)*coefficients[3*s+1]*temperature);
   absolute+=L(flux.species[s])*h;conditioned+=L(flux.species[s])*(h-L(ctx.energyReference));sumSpecies+=flux.species[s];
   const L dh=L((xa[2]+xb[2])/R(2))*(L(coefficients[3*s])+L(coefficients[3*s+1])*(L(in.temperature)+L((xa[2]+xb[2])/R(2))/2));
   const R stable=d::speciesEnthalpyIncrement(s,in.temperature,(xa[2]+xb[2])/R(2),in.model.thermo,ctx.enthalpyJump[s]);
   assert(std::abs(L(stable)-dh)<=8*std::numeric_limits<R>::epsilon()*std::abs(dh));
  }
  const L vn=(L(a.normalVelocity)+b.normalVelocity)/2;
  const L heat=-L(in.model.conductivity)*(L(xb[2])-xa[2])/length;
  absolute+=L(ctx.mass)*vn*vn/2+heat;conditioned+=L(ctx.mass)*vn*vn/2+heat;
  for(int axis=0;axis<2;++axis){
   const L ur=(L(xa[axis])+xb[axis])/2,u=ur+ctx.wallU[axis];
   const L tau=L(in.model.viscosity)*(L(xb[axis])-xa[axis])/length;
   absolute+=L(ctx.mass)*u*u/2-tau*u;conditioned+=L(ctx.mass)*ur*ur/2-tau*ur;
   const L momentum=L(ctx.mass)*ur-tau;
   assert(std::abs(L(flux.momentum[axis])-momentum)<=8*std::numeric_limits<R>::epsilon()*(std::abs(L(ctx.mass)*ur)+std::abs(tau)));
  }
  const L relGauge=std::abs(L(flux.energy)-conditioned)/std::abs(conditioned);
  const L physical=d::physicalLayerEnergyFlux(ctx,flux);
  // Finite arithmetic sum(Fs) need not equal mass bitwise. Compare against
  // the exactly corresponding algebraic reference, then bound that closure.
  const L expectedPhysical=absolute+L(ctx.energyReference)*(L(ctx.mass)-sumSpecies);
  const L relPhysical=std::abs(physical-expectedPhysical)/std::max(L(1),std::abs(expectedPhysical));
  assert(relGauge<(sizeof(R)==4?2e-6L:1e-12L));
  assert(relPhysical<(sizeof(R)==4?2e-6L:1e-12L));
  assert(std::abs(conditioned)>100); // Formation/differential-diffusion energy retained.
  WallWorkspace<R,2,4> work;work.state[1][2]=R(3e-6);assert(reactingWallProfileTemperature(in,work,1)==R(in.temperature+work.state[1][2]));
  maxGauge=std::max(maxGauge,relGauge);maxPhysical=std::max(maxPhysical,relPhysical);++count;
 }
 std::cout<<"frame/gauge bits="<<sizeof(R)*8<<" cases="<<count<<" max_gauge_relative="<<double(maxGauge)<<" max_physical_relative="<<double(maxPhysical)<<"\n";
}
int main(){std::cout<<std::setprecision(17);nasaAudit();frameGaugeAudit();}
