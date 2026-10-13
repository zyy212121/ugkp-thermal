"""Combined actual-channel availability cases from the scoped gas-core review."""
from pathlib import Path
import subprocess
import pytest
from test_gas_rates import SOURCE as RATE_SOURCE

ROOT = Path(__file__).resolve().parents[2]
SOURCE = RATE_SOURCE.split('int main(){')[0] + r'''
#include "common/chemistry/LocalStiffIntegrator.H"
#include <string>
int main(int argc,char** argv){
 assert(argc==2);const std::string test=argv[1];Model a;ChemistryStatus status;
 double c[2]={3,1},w[2]={-777,-778},j[4]={-777,-778,-779,-780},thermal[2]={-781,-782};
 if(test=="undefined_metric_scalar"){
  a.re[0].coefficient=a.pr[0].coefficient=.5;a.rx[0].highRate.preExponential=1000;
  ClosedReactorInput<double,2> base;base.speciesMass[0]=.028;base.totalMass=.028;base.volume=.001;base.internalEnergy=mixtureEnergy(base.speciesMass,700.,a.t);
  ChemistryControls<double> controls;controls.relativeTolerance=controls.absoluteMassFractionTolerance=controls.absoluteTemperatureTolerance=0;controls.maximumNewtonIterations=1;controls.maximumLineSearchIterations=3;
  gasChemistryDetail::ReactorContext<double,2> ctx(base,a.t,a.m,controls);assert(ctx.prepare());
  const double gamma=1-1/std::sqrt(2.);double z[2]={},rhs[2]={},T=700;
  // Undefined convergence metrics must not silently replace an internal
  // rejected trial with a finite candidate before the caller can refine it.
  const auto failed=gasChemistryDetail::implicitStage(ctx,rhs,gamma,z,T);
  assert(!failed && failed.code==ChemistryCode::NewtonFailure);
  ClosedReactorResult<double,2> out;out.temperature=-777;
  assert(!gasChemistryDetail::sdirkStep(base,1.,a.t,a.m,controls,out,700.));
  assert(out.temperature==-777);
 }else if(test=="empty_reactants" || test=="empty_products" || test=="empty_efficiencies"){
  if(test=="empty_reactants"){a.rx[0].reactantCount=0;a.rx[0].reactantOffset=100;a.m.reactantTermCount=0;a.m.reactants=nullptr;}
  if(test=="empty_products"){a.rx[0].productCount=0;a.rx[0].productOffset=100;a.m.productTermCount=0;a.m.products=nullptr;}
  if(test=="empty_efficiencies"){a.rx[0].type=GasReactionType::ThirdBody;a.rx[0].efficiencyCount=0;a.rx[0].efficiencyOffset=100;a.m.efficiencyCount=0;a.m.efficiencies=nullptr;}
  assert(evaluateGasRates(700.,c,a.t,a.m,w,nullptr,status));
  assert(evaluateGasRateDerivatives(700.,c,a.t,a.m,j,thermal,status));
  if(test=="empty_reactants"){assert(w[0]==0 && near(w[1],2));for(double v:j)assert(v==0);}
  if(test=="empty_products"){assert(near(w[0],-6) && w[1]==0);assert(near(j[0],-2) && j[1]==0 && j[2]==0 && j[3]==0);}
  if(test=="empty_efficiencies"){assert(near(w[0],-24) && near(w[1],24));assert(near(j[0],-14) && near(j[1],-6) && near(j[2],14) && near(j[3],6));}
 }else if(test=="zero_order_scalar" || test=="zero_order_ad"){
  a.rx[0].reactantCount=0;a.m.reactantTermCount=0;a.m.reactants=nullptr;const double* absent=nullptr;
  if(test=="zero_order_scalar"){double q=-777;assert(gasChemistryDetail::progressRate(0,700.,absent,a.t,a.m,q));assert(evaluateGasRates(700.,absent,a.t,a.m,w,nullptr,status));assert(w[0]==0 && near(w[1],q) && near(q,2));}
  else {assert(evaluateGasRateDerivatives(700.,absent,a.t,a.m,j,thermal,status));for(double v:j)assert(v==0);for(double v:thermal)assert(v==0);
   a.rx[0].highRate.temperatureExponent=1;assert(evaluateGasRateDerivatives(700.,absent,a.t,a.m,j,thermal,status));for(double v:j)assert(v==0);assert(thermal[0]==0 && near(thermal[1],2));}
 }else if(test=="missing_consumed_data"){
  const double* absent=nullptr;
  // Forward product, reverse product and collider each genuinely read c.
  for(int mode=0;mode<3;++mode){Model b;if(mode==1){b.rx[0].reactantCount=0;b.rx[0].reversible=true;}if(mode==2){b.rx[0].reactantCount=0;b.rx[0].type=GasReactionType::ThirdBody;}
   assert(!evaluateGasRates(700.,absent,b.t,b.m,w,nullptr,status));assert(w[0]==-777 && w[1]==-778);
   assert(!evaluateGasRateDerivatives(700.,absent,b.t,b.m,j,thermal,status));assert(j[0]==-777 && j[3]==-780 && thermal[0]==-781 && thermal[1]==-782);}
  // Truly missing consumed backing remains unavailable; no unsafe OOB probe.
  for(int mode=0;mode<3;++mode){Model b;if(mode==0)b.m.reactants=nullptr;if(mode==1)b.m.products=nullptr;if(mode==2){b.rx[0].type=GasReactionType::ThirdBody;b.rx[0].efficiencyCount=1;b.m.efficiencies=nullptr;b.m.efficiencyCount=1;}
   assert(!evaluateGasRates(700.,c,b.t,b.m,w,nullptr,status));assert(w[0]==-777 && w[1]==-778);
   assert(!evaluateGasRateDerivatives(700.,c,b.t,b.m,j,thermal,status));assert(j[0]==-777 && j[3]==-780 && thermal[0]==-781 && thermal[1]==-782);}
 }else assert(false);
}
'''

@pytest.fixture(scope='module')
def runner(tmp_path_factory):
    folder=tmp_path_factory.mktemp('gas-review-regressions')
    source=folder/'probe.cpp';source.write_text(SOURCE);exe=folder/'probe'
    run=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I'+str(ROOT),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    return exe

@pytest.mark.parametrize('case',['undefined_metric_scalar','empty_reactants','empty_products','empty_efficiencies','zero_order_scalar','zero_order_ad','missing_consumed_data'])
def test_review_actual_channel_availability(runner,case):
    run=subprocess.run([str(runner),case],capture_output=True,text=True,timeout=20)
    assert run.returncode==0,run.stderr
