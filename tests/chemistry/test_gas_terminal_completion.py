"""Real terminal producers replace sentinels with their actual IEEE arithmetic."""
from pathlib import Path
import subprocess
import pytest
from test_gas_rates import SOURCE as RATE_SOURCE
from test_gas_chemistry_adapter import SOURCE as ADAPTER_SOURCE
ROOT = Path(__file__).resolve().parents[2]
SOURCE = RATE_SOURCE.split('int main(){')[0] + r'''
#include "common/chemistry/LocalStiffIntegrator.H"
#include <string>
int main(int argc,char** argv){
 assert(argc==2);std::string mode=argv[1];Model a;
 double c[2]={3,1},w[2]={-771,-772},q[2]={-773,-774},j[4]={-775,-776,-777,-778},jt[2]={-779,-780};ChemistryStatus st;
 ChemistryControls<double> controls;ClosedReactorInput<double,2> base;base.speciesMass[0]=.028;base.totalMass=.028;base.volume=.001;base.internalEnergy=mixtureEnergy(base.speciesMass,700.,a.t);
 if(mode=="rates_overflow"){
  a.m.reactionCount=2;a.rx[0].highRate.preExponential=std::numeric_limits<double>::max();
  assert(!evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(std::isinf(w[0])&&w[0]<0&&std::isinf(w[1])&&w[1]>0);assert(std::isinf(q[0])&&q[1]==6);
 }else if(mode=="troe" || mode=="troe_center"){
  a.rx[0].type=GasReactionType::Troe;a.rx[0].lowRate.preExponential=1;a.rx[0].troe.T1=a.rx[0].troe.T3=mode=="troe"?-.001:.001;
  assert(!evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(std::isnan(w[0])&&std::isnan(w[1])&&std::isnan(q[0]));
  w[0]=-771;w[1]=-772;q[0]=-773;assert((!evaluateGasRates<double,2,double,false>(700.,c,a.t,a.m,w,q,st)));assert(w[0]==-771&&w[1]==-772&&q[0]==-773);
 }else if(mode=="ad"){
  a.re[0].coefficient=.5;c[0]=0;assert(evaluateGasRates(700.,c,a.t,a.m,w,nullptr,st));assert(w[0]==0&&w[1]==0);
  assert(!evaluateGasRateDerivatives(700.,c,a.t,a.m,j,jt,st));assert(!std::isfinite(j[0])&&!std::isfinite(j[2]));assert(j[1]==0&&j[3]==0&&jt[0]==0&&jt[1]==0);
 }else if(mode=="basis"){
  a.basis[0]=a.basis[1]=0;double basis[4]={-771,-772,-773,-774};assert(!gasChemistryDetail::orthogonalBasis(a.m,basis));assert(std::isnan(basis[0])&&std::isnan(basis[2]));
  gasChemistryDetail::ReactorContext<double,2> context(base,a.t,a.m,controls,false);for(double& x:context.leftInverse)x=-777;context.prepare();assert(std::isnan(context.leftInverse[0])&&std::isnan(context.leftInverse[1]));
 }else if(mode=="dense"){
  double matrix[4]={0,0,0,0},rhs[2]={1,2};assert((!gasChemistryDetail::solveDense<double,2>(2,matrix,rhs)));assert(std::isnan(rhs[0])&&std::isnan(rhs[1]));
 }else if(mode=="projection"){
  a.basis[0]=-1e-300;a.basis[1]=1e-300;a.rx[0].highRate.preExponential=1e100;
  double z[2]={},f[2]={-777,-778},T=700;st=evaluateConservativeReactor(base,z,a.t,a.m,controls,f,T,j);assert(!st);assert(std::isinf(f[0]));assert(j[0]!=-775);
 }else if(mode=="state"){
  gasChemistryDetail::ReactorContext<double,2> context(base,a.t,a.m,controls,false);double z[2]={std::numeric_limits<double>::infinity(),0},mass[2]={-777,-778},T=std::numeric_limits<double>::quiet_NaN();
  assert(context.state(z,mass,T));assert(std::isinf(mass[0])&&std::isinf(mass[1]));assert(T==2100);
 }else if(mode=="stage" || mode=="interval" || mode=="nonfinite_interval" || mode=="nan_interval" || mode=="negative_interval"){
  a.rx[0].highRate.preExponential=std::numeric_limits<double>::max();controls.maximumSteps=0;controls.maximumNewtonIterations=1;controls.maximumLineSearchIterations=1;
  ClosedReactorResult<double,2> result;result.speciesMass[0]=-771;result.speciesMass[1]=-772;result.temperature=-773;
  if(mode=="stage"){
   gasChemistryDetail::ReactorContext<double,2> context(base,a.t,a.m,controls,false);assert(context.prepare());gasChemistryDetail::sdirkStep(context,.1,result,700.);
  }else{
   const double dt=mode=="interval"?.1:(mode=="negative_interval"?-.1:(mode=="nan_interval"?std::numeric_limits<double>::quiet_NaN():std::numeric_limits<double>::infinity()));st=advanceClosedReactor(base,dt,a.t,a.m,controls,result);
   assert(st.attemptedSteps==1); // Nonfinite-input propagation, not a finite physical solution.
  }
  assert(result.speciesMass[0]!=-771&&result.speciesMass[1]!=-772&&result.temperature!=-773);assert(!std::isfinite(result.speciesMass[0])&&!std::isfinite(result.speciesMass[1]));
 }else if(mode=="wider_underflow" || mode=="wider_overflow"){
  const long double dt=mode=="wider_underflow"?1e-400L:1e309L;a.rx[0].highRate.preExponential=mode=="wider_underflow"?1e300:1e-310;controls.maximumSteps=0;
  ClosedReactorResult<double,2> result;bool available=false;st=advanceClosedReactor(base,dt,a.t,a.m,controls,result,&available);
  assert(st&&available&&st.attemptedSteps==1&&result.speciesMass[1]>0);
  gasChemistryDetail::ReactorContext<double,2> context(base,a.t,a.m,controls,false);assert(context.prepare());double T=0;assert(invertMixtureEnergy(base.speciesMass,base.internalEnergy,a.t,controls.thermo,T)==ThermoStatus::Success);
  ClosedReactorResult<double,2> full,half,end;assert(gasChemistryDetail::sdirkStep(context,dt,full,T));assert(gasChemistryDetail::sdirkStep(context,dt/2,half,T));
  ClosedReactorInput<double,2> middle=base;for(int i=0;i<2;++i)middle.speciesMass[i]=half.speciesMass[i];gasChemistryDetail::ReactorContext<double,2> second(middle,a.t,a.m,controls,false);assert(second.prepare());assert(gasChemistryDetail::sdirkStep(second,dt/2,end,half.temperature));
  for(int i=0;i<2;++i)assert(result.speciesMass[i]==end.speciesMass[i]);
  assert(result.temperature==end.temperature);
 }else if(mode=="metadata"){
  a.m.reactantTermCount=0;a.m.productTermCount=0;assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(w[0]==-6&&w[1]==6&&q[0]==6);
 }else if(mode=="stage_caloric_diagnostic"){
  base.volume=1.;a.rx[0].reactantCount=0;a.rx[0].highRate.preExponential=1e306;a.basis[0]=0;a.basis[1]=1;controls.maximumSteps=0;controls.maximumNewtonIterations=1;controls.maximumLineSearchIterations=1;
  gasChemistryDetail::ReactorContext<double,2> context(base,a.t,a.m,controls,false);assert(context.prepare());
  double z[2]={},rhs[2]={},mass[2],f[2],T=700;bool scalar=false,available=false;
  st=gasChemistryDetail::continueImplicitStage(context,rhs,1.,1.-1./std::sqrt(2.),z,T,mass,f,scalar,false,&available);
  assert(available&&scalar);assert(!st&&st.thermoCode!=ThermoStatus::Success);assert(mass[1]>1e300&&std::isfinite(f[0]));
 }else if(mode=="scalar_ad_split"){
  a.re[0].coefficient=.5;base.speciesMass[0]=0;base.speciesMass[1]=.028;gasChemistryDetail::ReactorContext<double,2> context(base,a.t,a.m,controls,false);assert(context.prepare());
  double f[2]={-771,-772};bool scalar=false,derivative=false;
  st=context.evaluateState(base.speciesMass,f,700.,j,&scalar,&derivative);assert(!st&&scalar&&derivative&&f[0]==0&&!std::isfinite(j[0]));
  a.t.coefficients=nullptr;for(double& x:j)x=-779;scalar=derivative=false;
  st=context.evaluateState(base.speciesMass,f,700.,j,&scalar,&derivative);assert(!st&&scalar&&!derivative&&f[0]==0);for(double x:j)assert(x==-779);
 }else if(mode=="zero_rank_storage"){
  a.m.independentRank=0;a.m.stoichiometricBasis=nullptr;bool available=false;
  assert(gasChemistryDetail::orthogonalBasis(a.m,static_cast<double*>(nullptr),&available)&&available);
  assert((gasChemistryDetail::solveDense<double,2>(0,nullptr,nullptr,&available)&&available));
  gasChemistryDetail::ReactorContext<double,2> context(base,a.t,a.m,controls,false);assert(context.prepare());
  assert(context.evaluateJacobian(base.speciesMass,700.,nullptr,&available)&&available);
 }else if(mode=="unused_backing"){
  a.rx[0].reactantCount=a.rx[0].productCount=0;a.rx[0].reversible=true;a.m.reactants=a.m.products=nullptr;a.t.species=nullptr;
  assert(evaluateGasRates(700.,static_cast<const double*>(nullptr),a.t,a.m,w,q,st));assert(w[0]==0&&w[1]==0&&q[0]==0);
  assert(evaluateGasRateDerivatives(700.,static_cast<const double*>(nullptr),a.t,a.m,j,jt,st));for(double x:j)assert(x==0);for(double x:jt)assert(x==0);
  a.rx[0].reversible=false;a.rx[0].reactantCount=-1;assert(evaluateGasRates(700.,static_cast<const double*>(nullptr),a.t,a.m,w,q,st));assert(w[0]==0&&w[1]==0&&q[0]==2);
 }else if(mode=="gibbs_channels"){
  a.rx[0].reversible=true;a.coef[5]=3e6;double expected[2],expectedJ[4],expectedT[2];
  assert(evaluateGasRates(700.,c,a.t,a.m,expected,nullptr,st));assert(evaluateGasRateDerivatives(700.,c,a.t,a.m,expectedJ,expectedT,st));assert(expectedT[0]!=0&&expectedT[1]!=0);
  a.t.coefficientCount=2;a.sp[0].hasEntropyReference=a.sp[1].hasEntropyReference=false;
  assert(evaluateGasRates(700.,c,a.t,a.m,w,nullptr,st));assert(evaluateGasRateDerivatives(700.,c,a.t,a.m,j,jt,st));
  for(int i=0;i<2;++i){assert(w[i]==expected[i]);assert(jt[i]==expectedT[i]);}for(int i=0;i<4;++i)assert(j[i]==expectedJ[i]);
 }else if(mode=="near_equilibrium"){
  a.rx[0].reversible=true;c[0]=3;c[1]=3*(1+1e-12);assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));
  const double forward=2*c[0],backward=std::exp(std::log(2.)+std::log(c[1]));
  const double expected=-forward*std::expm1(std::log(backward)-std::log(forward));assert(q[0]==expected&&w[0]==-expected&&w[1]==expected);
 }else if(mode=="half_failure" || mode=="second_half_failure"){
  controls.newtonTolerance=0;controls.maximumNewtonIterations=2;controls.maximumLineSearchIterations=1;controls.maximumRejectedSteps=0;
  double initialTemperature=0;assert(invertMixtureEnergy(base.speciesMass,base.internalEnergy,a.t,controls.thermo,initialTemperature)==ThermoStatus::Success);
  const double h=(mode=="half_failure"?12:184)*.0001;
  gasChemistryDetail::ReactorContext<double,2> strict(base,a.t,a.m,controls);assert(strict.prepare());
  ClosedReactorResult<double,2> full,half,end;auto fs=gasChemistryDetail::sdirkStep(strict,h,full,initialTemperature);assert(fs);
  auto hs=gasChemistryDetail::sdirkStep(strict,h/2,half,initialTemperature);
  if(mode=="half_failure")assert(!hs);
  else {assert(hs);ClosedReactorInput<double,2> middle=base;for(int i=0;i<2;++i)middle.speciesMass[i]=half.speciesMass[i];
    gasChemistryDetail::ReactorContext<double,2> second(middle,a.t,a.m,controls);assert(second.prepare());
    auto es=gasChemistryDetail::sdirkStep(second,h/2,end,half.temperature);assert(!es);}
  strict.refinement=false;fs=gasChemistryDetail::sdirkStep(strict,h,full,initialTemperature);hs=gasChemistryDetail::sdirkStep(strict,h/2,half,initialTemperature);assert(fs&&hs);
  ClosedReactorInput<double,2> middle=base;for(int i=0;i<2;++i)middle.speciesMass[i]=half.speciesMass[i];
  gasChemistryDetail::ReactorContext<double,2> second(middle,a.t,a.m,controls,false);assert(second.prepare());
  auto es=gasChemistryDetail::sdirkStep(second,h/2,end,half.temperature);assert(es);
  ClosedReactorResult<double,2> actual;st=advanceClosedReactor(base,h,a.t,a.m,controls,actual);assert(st);
  for(int i=0;i<2;++i)assert(actual.speciesMass[i]==end.speciesMass[i]);
  assert(actual.temperature==end.temperature);
  assert(st.attemptedSteps==2&&st.rejectedSteps==1);
  assert(actual.speciesMass[0]!=half.speciesMass[0]);
 }else if(mode=="prefix_remaining"){
  controls.initialStep=1e-6;controls.maximumSteps=1;
  ClosedReactorResult<double,2> prefix;
  const auto first=advanceClosedReactorTrial(base,1e-6,a.t,a.m,controls,prefix);assert(first&&first.attemptedSteps==1);
  ClosedReactorInput<double,2> current=base;for(int i=0;i<2;++i)current.speciesMass[i]=prefix.speciesMass[i];
  gasChemistryDetail::ReactorContext<double,2> context(current,a.t,a.m,controls,false);assert(context.prepare());
  const double remaining=.1-1e-6;ClosedReactorResult<double,2> full,half,end;
  auto fs=gasChemistryDetail::sdirkStep(context,remaining,full,prefix.temperature);auto hs=gasChemistryDetail::sdirkStep(context,remaining/2,half,prefix.temperature);assert(fs&&hs);
  ClosedReactorInput<double,2> middle=current;for(int i=0;i<2;++i)middle.speciesMass[i]=half.speciesMass[i];
  gasChemistryDetail::ReactorContext<double,2> second(middle,a.t,a.m,controls,false);for(int i=0;i<4;++i)second.leftInverse[i]=context.leftInverse[i];
  auto es=gasChemistryDetail::sdirkStep(second,remaining/2,end,half.temperature);assert(es);
  ClosedReactorResult<double,2> actual;bool available=false;
  st=advanceClosedReactor(base,.1,a.t,a.m,controls,actual,&available);assert(st&&available);
  for(int i=0;i<2;++i)assert(actual.speciesMass[i]==end.speciesMass[i]);
  assert(actual.temperature==end.temperature);
  assert(st.attemptedSteps==2&&st.rejectedSteps==0);
  assert(actual.speciesMass[0]!=half.speciesMass[0]&&actual.speciesMass[0]!=prefix.speciesMass[0]);
 }else if(mode=="missing"){
  a.m.reactants=nullptr;assert(!evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(w[0]==-771&&w[1]==-772&&q[0]==-773);
  a.m.reactants=a.re;a.t.coefficients=nullptr;a.rx[0].reversible=true;assert(!evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(w[0]==-771&&w[1]==-772);
 }else assert(false);
}
'''
@pytest.fixture(scope='module')
def runner(tmp_path_factory):
    folder=tmp_path_factory.mktemp('terminal-completion');src=folder/'probe.cpp';src.write_text(SOURCE);exe=folder/'probe'
    built=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I'+str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
    (folder/'build.log').write_text(built.stdout+built.stderr)
    assert built.returncode==0,built.stderr
    return exe
@pytest.mark.parametrize('scenario',['rates_overflow','troe','troe_center','ad','basis','dense','projection','state','stage','interval','nonfinite_interval','nan_interval','negative_interval','wider_underflow','wider_overflow','metadata','stage_caloric_diagnostic','scalar_ad_split','zero_rank_storage','unused_backing','gibbs_channels','near_equilibrium','half_failure','second_half_failure','prefix_remaining','missing'])
def test_terminal_real_fields(runner,scenario):
    run=subprocess.run([str(runner),scenario],capture_output=True,text=True,timeout=20)
    assert run.returncode==0,run.stderr

ADAPTER=ADAPTER_SOURCE.split('int main(){')[0].replace('TEST_REAL','double')+r'''
#include <string>
int main(int argc,char**argv){
 assert(argc==2);Fixture f;std::string mode=argv[1];auto& c=f.state.gasSpecies.chemistryControls;c.maximumSteps=0;c.maximumNewtonIterations=1;c.maximumLineSearchIterations=1;
 if(mode=="zero_volume"){
  bool available=false;const auto status=advanceGasChemistryCell(f.state,0,.1,0.,false,&available);assert(!status&&available);assert(!std::isfinite(f.partial[0])&&!std::isfinite(f.partial[2]));
 }else if(mode=="nonfinite_energy"){
  f.energy[0]=std::numeric_limits<double>::infinity();advanceGasChemistryCell(f.state,0,.1,.5);assert(f.partial[0]!=1.5&&f.partial[2]!=0);
 }else if(mode=="missing_volume"){
  double old[4];std::memcpy(old,f.partial,sizeof old);bool available=true;assert(!advanceGasChemistryCellWithVolumes(f.state,0,.1,static_cast<const double*>(nullptr),false,&available));assert(!available);assert(std::memcmp(old,f.partial,sizeof old)==0);
  f.state.gasSpecies.mode=GasMode::MixtureFrozen;assert(advanceGasChemistryCellWithVolumes(f.state,0,.1,static_cast<const double*>(nullptr)));
 }else if(mode=="two_halves"){
  const double oldVolume[2]={.5,.7},newVolume[2]={2.,3.};
  const auto predict=[&](double volume){
   ClosedReactorInput<double,2> input;input.volume=volume;input.totalMass=f.rho[0]*volume;
   input.speciesMass[0]=f.partial[0]*volume;input.speciesMass[1]=f.partial[2]*volume;
   input.internalEnergy=(f.energy[0]-(f.mx[0]*f.mx[0]+f.my[0]*f.my[0]+f.mz[0]*f.mz[0])/(2*f.rho[0]))*volume;
   ClosedReactorResult<double,2> result;assert(advanceClosedReactor(input,.05,f.state.gasSpecies.thermo,f.state.gasSpecies.mechanism,c,result));return result;
  };
  const double initialSpecies[2]={f.partial[0],f.partial[2]};const double savedEnergy=f.energy[0];const auto firstExpected=predict(oldVolume[0]);
  assert(advanceGasChemistryCellWithVolumes(f.state,0,.05,oldVolume));const double firstChange[2]={f.partial[0]*.5-initialSpecies[0]*.5,f.partial[2]*.5-initialSpecies[1]*.5};const double middle[2]={f.partial[0],f.partial[2]};
  assert(middle[0]==firstExpected.speciesMass[0]/oldVolume[0]&&middle[1]==firstExpected.speciesMass[1]/oldVolume[0]);
  const auto secondExpected=predict(newVolume[0]);assert(advanceGasChemistryCellWithVolumes(f.state,0,.05,newVolume));const double secondChange[2]={f.partial[0]*2-middle[0]*2,f.partial[2]*2-middle[1]*2};
  assert(firstChange[0]<0&&firstChange[1]>0&&secondChange[0]<0&&secondChange[1]>0);
  assert(std::abs(firstChange[0]+firstChange[1])<1e-12&&std::abs(secondChange[0]+secondChange[1])<1e-12);
  assert(firstChange[0]!=secondChange[0]);assert(f.partial[0]==secondExpected.speciesMass[0]/newVolume[0]&&f.partial[2]==secondExpected.speciesMass[1]/newVolume[0]);assert(f.energy[0]==savedEnergy);
 }else if(mode=="conversion_overflow"){
  f.species[0].molarMass=f.species[1].molarMass=1e100;
  f.basis[0]=0;f.basis[1]=1;f.reaction[0].reactantCount=0;f.reaction[0].highRate.preExponential=1e300;c.thermo.maximumIterations=0;
  advanceGasChemistryCell(f.state,0,1e-6,1e-100);assert(std::isinf(f.partial[2]));
 }else if(mode=="missing"){
  f.state.rhoE=nullptr;assert(!advanceGasChemistryCell(f.state,0,.1,.5));assert(f.partial[0]==1.5);
 }else assert(false);
}
'''
@pytest.mark.parametrize('scenario',['zero_volume','nonfinite_energy','missing_volume','two_halves','conversion_overflow','missing'])
def test_terminal_adapter_fields(tmp_path,scenario):
    src=tmp_path/'adapter.cpp';src.write_text(ADAPTER);exe=tmp_path/'adapter'
    built=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I'+str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    run=subprocess.run([str(exe),scenario],capture_output=True,text=True,timeout=20)
    assert run.returncode==0,run.stderr
