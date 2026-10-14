"""Actual header execution for finite-candidate continuation and honest outputs."""
from pathlib import Path
import subprocess
import pytest
from test_gas_rates import SOURCE as RATE_SOURCE
from test_gas_chemistry_adapter import SOURCE as ADAPTER_SOURCE
ROOT = Path(__file__).resolve().parents[2]
MODEL = RATE_SOURCE.split('int main(){')[0]
SOURCE = MODEL + r'''
#include "common/chemistry/LocalStiffIntegrator.H"
#include <string>
#include <vector>
#include <algorithm>
struct Caloric {
 int kind;
 mutable std::vector<double> evaluated;
 ThermoStatus operator()(double t,double& e,double& c) const {
  evaluated.push_back(t);
  if(kind==3) return ThermoStatus::InvalidModel;
  c=kind==1?-2.:(kind==2?0.:2.);e=c*t+5.;return ThermoStatus::Success;
 }
};
int main(int argc,char** argv){
 assert(argc==2);std::string test=argv[1];Model a;
 ChemistryControls<double> controls;
 ClosedReactorInput<double,2> b;b.speciesMass[0]=.028;b.totalMass=.028;b.volume=.001;b.internalEnergy=mixtureEnergy(b.speciesMass,700.,a.t);
 if(test=="thermo") {
  a.sp[0].hasEntropyReference=false;a.sp[0].minTemperature=1000;a.sp[0].maxTemperature=200;
  assert(speciesCp(0,700.,a.t)==1000);assert(gasFinite(speciesE(0,700.,a.t)));assert(!gasFinite(speciesEntropyStandard(0,700.,a.t)));
  a.coef[0]=100.;assert(speciesCv(0,700.,a.t)<0);
  double mass[2]={-.01,.02},e=-777,cv=-778;assert(evaluateMixtureEnergy(mass,700.,a.t,e,cv)==ThermoStatus::Success);assert(e!=-777&&cv!=-778);
  a.t.coefficients=nullptr;assert(gasFinite(mixtureGasConstant(mass,a.t)));double density=-77;assert(mixtureDensityFromPressureTemperature(101325.,700.,mass,a.t,density)==ThermoStatus::Success);assert(gasFinite(density));
 } else if(test=="property") {
  double mass[2]={1,1};a.coef[0]=a.coef[3]=std::numeric_limits<double>::max();a.coef[2]=a.coef[5]=1;
  assert(mixtureEnergy(mass,0.,a.t)==2);assert(!gasFinite(mixtureHeatCapacity(mass,0.,a.t)));
  a.coef[0]=a.coef[3]=1000;a.coef[2]=a.coef[5]=std::numeric_limits<double>::max();assert(gasFinite(mixtureHeatCapacity(mass,700.,a.t)));assert(!gasFinite(mixtureEnergy(mass,700.,a.t)));
 } else if(test=="segment") {
  SpeciesThermoData<double> sp;sp.model=SpeciesThermoModel::NASA7;sp.molarMass=.028;sp.midTemperature=1000;sp.minTemperature=200;sp.maxTemperature=4000;
  double c[7]={3.5,0,0,0,0,0,1};SpeciesThermoView<double,1> t;t.species=&sp;t.coefficients=c;t.coefficientCount=7;
  assert(gasFinite(speciesE(0,700.,t)));assert(!gasFinite(speciesE(0,1500.,t)));
  double segments[13]={3.5,0,0,0,0,0,1,3.5,0,0,0,0,0};t.coefficients=segments;t.coefficientCount=13;
  assert(gasFinite(speciesE(0,1500.,t)));assert(!gasFinite(speciesEntropyStandard(0,1500.,t)));
  t.coefficientCount=12;assert(gasFinite(speciesCp(0,1500.,t)));assert(!gasFinite(speciesE(0,1500.,t)));
  a.t.coefficientCount=2;assert(speciesCp(0,700.,a.t)==1000);assert(!gasFinite(speciesE(0,700.,a.t)));a.t.coefficientCount=6;
  a.coef[2]=std::numeric_limits<double>::max();a.coef[0]=std::numeric_limits<double>::max()/1000;
  assert(gasFinite(speciesCp(0,700.,a.t)));assert(!gasFinite(speciesH(0,700.,a.t)));
 } else if(test=="inversion") {
  for(int kind=0;kind<3;++kind)for(int budget=0;budget<=1;++budget){Caloric e{kind,{}};ThermoInversionControls<double> c;c.maximumIterations=budget;double T=3500;
   assert(invertCaloricEnergy(e,2005.,200.,4000.,c,T)==ThermoStatus::Success);assert(T==3500);assert(std::find(e.evaluated.begin(),e.evaluated.end(),T)!=e.evaluated.end());}
  Caloric e{0,{}};double T=700;assert(invertCaloricEnergy(e,10005.,200.,4000.,controls.thermo,T)==ThermoStatus::Success);assert(T==5000);
  Caloric decreasing{1,{}};T=700;assert(invertCaloricEnergy(decreasing,-1995.,4000.,200.,controls.thermo,T)==ThermoStatus::Success);assert(T==1000);
  Caloric failure{3,{}};T=700;assert(invertCaloricEnergy(failure,2005.,200.,4000.,controls.thermo,T)!=ThermoStatus::Success);assert(T==700);
 } else if(test=="rates") {
  a.t.speciesOrderHash=9;a.sp[0].hasEntropyReference=false;a.sp[0].minTemperature=900;a.sp[0].maxTemperature=200;a.basis[1]=2;a.pr[0].coefficient=2;
  double c[2]={-3,1},w[2]={-777,-778},j[4]={-777,-777,-777,-777},jt[2]={-777,-777};ChemistryStatus st;
  assert(validateGasMechanismView(a.t,a.m));assert(evaluateGasRates(700.,c,a.t,a.m,w,nullptr,st));assert(w[0]==6&&w[1]==-12);assert(evaluateGasRateDerivatives(700.,c,a.t,a.m,j,jt,st));assert(j[0]==-2&&j[2]==4&&jt[0]==0&&jt[1]==0);
  c[1]=std::numeric_limits<double>::infinity();assert(evaluateGasRates(700.,c,a.t,a.m,w,nullptr,st));assert(w[0]==6&&w[1]==-12);assert(evaluateGasRateDerivatives(700.,c,a.t,a.m,j,jt,st));c[1]=1;
  a.rx[0].highRate.preExponential=0;assert(evaluateGasRates(std::numeric_limits<double>::quiet_NaN(),c,a.t,a.m,w,nullptr,st));assert(w[0]==0&&w[1]==0);a.rx[0].highRate.preExponential=2;
  a.rx[0].reversible=true;w[0]=-777;w[1]=-778;bool available=false;evaluateGasRates(700.,c,a.t,a.m,w,nullptr,st,&available);assert(available&&w[0]!=-777&&w[1]!=-778);
  a.rx[0].reversible=false;a.rx[0].type=static_cast<GasReactionType>(255);assert(evaluateGasRates(700.,c,a.t,a.m,w,nullptr,st));assert(w[0]==0&&w[1]==0);
  a.rx[0].type=GasReactionType::Elementary;a.m.productTermCount=0;assert(evaluateGasRates(700.,c,a.t,a.m,w,nullptr,st));assert(w[0]==6&&w[1]==-12);
 } else if(test=="projection") {
  a.basis[0]=-1e-300;a.basis[1]=1e-300;a.rx[0].highRate.preExponential=1e100;
  double z[2]={},f[2]={-777,-778},T=700;auto st=evaluateConservativeReactor(b,z,a.t,a.m,controls,f,T,nullptr);assert(!st);assert(st.code==ChemistryCode::RateFailure);assert(std::isinf(f[0])&&f[1]==-778);
 } else if(test=="scalar_without_ad") {
  a.m.reactionCount=2;a.re[1].species=1;a.re[1].coefficient=.5;a.pr[1].species=0;a.pr[1].coefficient=.5;
  double z[2]={},f[2]={-777,-778},j[4]={-777,-777,-777,-777},T=700;
  auto st=evaluateConservativeReactor(b,z,a.t,a.m,controls,f,T,j);assert(!st);assert(st.code==ChemistryCode::RateFailure);assert(f[0]==2000);assert(!gasFinite(j[0]));for(int i=1;i<4;++i)assert(j[i]==-777);
 } else if(test=="conservation") {
  a.basis[1]=2;a.pr[0].coefficient=2;controls.absoluteMassTolerance=controls.absoluteElementTolerance=controls.relativeConservationTolerance=0;
  ClosedReactorResult<double,2> r;assert(advanceClosedReactor(b,.1,a.t,a.m,controls,r));assert(r.speciesMass[0]+r.speciesMass[1]>b.totalMass);double beforeElement=0,afterElement=0;for(int i=0;i<2;++i){const double atoms=a.t.elementComposition[i]/a.t.species[i].molarMass;beforeElement+=b.speciesMass[i]*atoms;afterElement+=r.speciesMass[i]*atoms;}assert(afterElement>beforeElement);
  b.totalMass*=2;assert(advanceClosedReactor(b,.1,a.t,a.m,controls,r));assert(r.speciesMass[0]+r.speciesMass[1]<b.totalMass);
 } else if(test=="dense") {
  const double eps=std::numeric_limits<double>::epsilon();double matrix[4]={1,1,1,1+eps},rhs[2]={1,2};assert((gasChemistryDetail::solveDense<double,2>(2,matrix,rhs)));assert(gasFinite(rhs[0])&&gasFinite(rhs[1])&&std::abs(rhs[1])>1e15);
  double singular[4]={1,2,2,4},srhs[2]={1,2};assert((!gasChemistryDetail::solveDense<double,2>(2,singular,srhs)));
  double basis[4]={1,1,0,eps},q[4];a.m.independentRank=2;a.m.stoichiometricBasis=basis;assert(gasChemistryDetail::orthogonalBasis(a.m,q));
 } else if(test=="state") {
  gasChemistryDetail::ReactorContext<double,2> ctx(b,a.t,a.m,controls);double z[2]={2000,0},mass[2]={-777,-778},T=700;assert(!ctx.state(z,mass,T)); // Original internal trial rejection feeds damping/refinement.
  ctx.refinement=false;assert(ctx.state(z,mass,T));assert(mass[0]<0&&mass[1]==.056);assert(gasFinite(T));
 } else if(test=="budget" || test=="lte" || test=="controls") {
  controls.maximumNewtonIterations=0;controls.maximumLineSearchIterations=0;controls.maximumSteps=0;controls.maximumRejectedSteps=0;controls.initialStep=.1;controls.minimumStep=.2;
  if(test=="lte"){controls.relativeTolerance=0;controls.absoluteMassFractionTolerance=0;controls.absoluteTemperatureTolerance=0;}
  if(test=="controls"){controls.thermo.maximumIterations=0;controls.newtonTolerance=0;controls.relativeConservationTolerance=0;controls.absoluteMassTolerance=0;}
  ClosedReactorResult<double,2> r;r.speciesMass[0]=r.speciesMass[1]=r.temperature=-777;
  assert(validateChemistryControls(controls));auto st=advanceClosedReactor(b,1.,a.t,a.m,controls,r);assert(st);assert(r.speciesMass[0]!=b.speciesMass[0]&&r.speciesMass[1]!=-777&&gasFinite(r.temperature));assert(st.attemptedSteps>0&&st.attemptedSteps<=10);assert(r.speciesMass[0]+r.speciesMass[1]==b.totalMass);
 } else if(test=="subnormal") {
  const double h=std::numeric_limits<double>::denorm_min();a.rx[0].highRate.preExponential=1e100;ClosedReactorResult<double,2> r;
  assert(advanceClosedReactor(b,h,a.t,a.m,controls,r));// Original successful half-step arithmetic rounds h/2 to zero. Preserve it.
  assert(r.speciesMass[0]==b.speciesMass[0] && r.speciesMass[1]==0);assert(gasFinite(r.temperature));
 }
}
'''
@pytest.fixture(scope='module')
def continuation_runner(tmp_path_factory):
 p=tmp_path_factory.mktemp('finite-continuation');src=p/'probe.cpp';src.write_text(SOURCE);exe=p/'probe'
 build=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I',str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert build.returncode==0,build.stderr
 return exe
@pytest.mark.parametrize('scenario',['thermo','property','segment','inversion','rates','projection','scalar_without_ad','conservation','dense','state','budget','lte','controls','subnormal'])
def test_complete_finite_candidates(continuation_runner,scenario):
 run=subprocess.run([str(continuation_runner),scenario],capture_output=True,text=True,timeout=20)
 assert run.returncode==0,run.stderr

ADAPTER=ADAPTER_SOURCE.split('int main(){')[0]+r'''
int main(){
 Fixture f;auto& c=f.state.gasSpecies.chemistryControls;c.maximumSteps=0;c.maximumNewtonIterations=0;c.maximumLineSearchIterations=0;c.absoluteMassTolerance=0;c.absoluteElementTolerance=0;c.relativeConservationTolerance=0;c.thermo.relativeEnergyTolerance=0;c.thermo.absoluteEnergyTolerance=0;
 f.partial[2]+=.01;
 ClosedReactorInput<Real,2> input;input.volume=Real(.7);input.totalMass=f.rho[0]*input.volume;for(int i=0;i<2;++i)input.speciesMass[i]=f.partial[2*i]*input.volume;
 const Real ux=f.mx[0]/f.rho[0],uy=f.my[0]/f.rho[0],uz=f.mz[0]/f.rho[0];input.internalEnergy=(f.energy[0]-Real(.5)*f.rho[0]*(ux*ux+uy*uy+uz*uz))*input.volume;
 ClosedReactorResult<Real,2> expected;assert(advanceClosedReactor(input,.1,f.state.gasSpecies.thermo,f.state.gasSpecies.mechanism,c,expected));
 const Real savedEnergy=f.energy[0];auto st=advanceGasChemistryCell(f.state,0,.1,Real(.7));assert(st);assert(f.partial[0]<1.5);assert(f.energy[0]==savedEnergy);assert(f.cellStatus[0]==0);assert(f.partial[0]==expected.speciesMass[0]/input.volume&&f.partial[2]==expected.speciesMass[1]/input.volume);
 Fixture reference;assert(advanceGasChemistryCell(reference.state,0,.01,Real(.5)));
 Fixture stale;stale.cellStatus[0]=int(GasTransportCode::NegativeInventory);stale.chemistryStatus[0].code=ChemistryCode::ConservationFailure;
 // Real trial owners reset aggregate and chemistry statuses before a new trial.
 // Direct adapter success owns chemistryStatus, not other operators' aggregate.
 if(OWNER_RESET){stale.cellStatus[0]=0;stale.chemistryStatus[0]=ChemistryStatus{};}
 const Real staleEnergy=stale.energy[0];assert(advanceGasChemistryCell(stale.state,0,.01,Real(.5)));
 assert(stale.cellStatus[0]==(OWNER_RESET?0:int(GasTransportCode::NegativeInventory))&&stale.chemistryStatus[0]);
 assert(std::memcmp(stale.partial,reference.partial,sizeof stale.partial)==0);
 assert(stale.partial[0]<Real(1.5)&&stale.partial[2]>Real(.5));
 assert(std::abs(stale.partial[0]+stale.partial[2]-stale.rho[0])<Real(1e-12));
 assert(stale.energy[0]==staleEnergy&&stale.rho[0]==reference.rho[0]&&stale.mx[0]==reference.mx[0]&&stale.my[0]==reference.my[0]&&stale.mz[0]==reference.mz[0]);
 Fixture other;other.cellStatus[0]=int(GasTransportCode::InvalidStorage);assert(advanceGasChemistryCell(other.state,0,.01,Real(.5)));assert(other.partial[0]<Real(1.5));assert(other.cellStatus[0]==int(GasTransportCode::InvalidStorage));
 Fixture missing;missing.state.rhoE=nullptr;assert(!advanceGasChemistryCell(missing.state,0,.01,Real(.5)));assert(missing.partial[0]==Real(1.5)&&missing.partial[2]==Real(.5));assert(!missing.chemistryStatus[0]);
 Fixture tiny;tiny.reaction[0].highRate.preExponential=Real(1e30);st=advanceGasChemistryCell(tiny.state,0,1e-46,Real(.5));assert(st);assert(tiny.partial[0]==Real(1.5)&&tiny.partial[2]>=Real(0));assert(tiny.chemistryStatus[0]&&tiny.cellStatus[0]==0);
 Fixture zero;Real before[4];std::memcpy(before,zero.partial,sizeof before);st=advanceGasChemistryCell(zero.state,0,0.,Real(.7));assert(st);assert(std::memcmp(before,zero.partial,sizeof before)==0);
}
'''
@pytest.mark.parametrize('owner_reset',[False,True],ids=['direct_preserves_aggregate','owner_reset_success'])
def test_adapter_publishes_quality_diagnostics(tmp_path,owner_reset):
 src=tmp_path/'adapter.cpp';src.write_text(ADAPTER.replace('TEST_REAL','double').replace('OWNER_RESET','true' if owner_reset else 'false'));exe=tmp_path/'adapter'
 b=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert b.returncode==0,b.stderr
 r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr
