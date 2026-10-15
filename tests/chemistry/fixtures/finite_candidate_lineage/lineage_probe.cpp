#include "probe_hooks.hpp"
#include "common/chemistry/LocalStiffIntegrator.H"
#include <cstring>
using namespace ugkwp;
struct Model {
 SpeciesThermoData<double> sp[2];double coef[6]={1000,0,0,1000,0,-2e5},elements[2]={1,1},basis[2]={-1,1};
 GasReactionData<double> rx[2];GasStoichTerm<double> re[2]={{0,1},{1,1}},pr[2]={{1,1},{0,1}};
 SpeciesThermoView<double,2> t;GasMechanismView<double,2> m;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 rx[0].reactantOffset=rx[0].productOffset=0;rx[0].reactantCount=rx[0].productCount=1;rx[0].highRate.preExponential=.07;rx[0].highRate.temperatureExponent=.5;
 m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=1;m.products=pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;}
};
int failures=0;
#define CHECK(x) do{if(!(x)){std::cout<<"FAIL "<<#x<<" line="<<__LINE__<<"\n";++failures;}}while(0)
using independent::steps;using independent::stages;using independent::states;
bool same(const std::vector<long double>& a,const std::vector<long double>& b){return a==b;}
template<class R>bool outputIs(const R& r,const independent::Step& s){return independent::values(r.speciesMass)==s.after && (long double)r.temperature==s.Tout;}
void validateStageLineage(){
 for(const auto& step:steps)if(step.success){CHECK(step.lastStage-step.firstStage==2);for(auto value:step.after)CHECK(std::isfinite(value)&&value!=-777);CHECK(std::isfinite(step.Tout)&&step.Tout!=-777);}
 for(const auto& stage:stages)if(stage.success){bool found=false;for(int i=stage.firstState;i<stage.lastState;++i){const auto& s=states[i];if(s.success && s.z==stage.after && s.T==stage.Tout)found=true;}
  if(!found && stage.after==stage.before && stage.rhs==stage.after){
   // Exact equilibrium may reuse the stage-one state. Verify its zero scalar
   // RHS was really evaluated; this does not license any nonzero residual seed.
   for(const auto& s:states)if(s.step==stage.step && s.success && s.z==stage.after && s.T==stage.Tout)
    for(const auto& rate:independent::rates)if(rate.step==stage.step && rate.success && rate.T==stage.Tout && rate.production[0]==0 && rate.production[1]==0)found=true;
  }
  CHECK(found);}
 for(const auto& step:steps)if(step.success && step.lastStage-step.firstStage==2){
  const auto& first=stages[step.firstStage];const auto& second=stages[step.firstStage+1];bool rhsHasSelectedState=false;
  for(const auto& state:states)if(state.step==step.id && state.success && state.z==first.after && state.T==first.Tout){
   for(const auto& rate:independent::rates)if(rate.step==step.id && rate.success && rate.T==first.Tout && rate.c[0]==(double)state.mass[0]/(.028*.001) && rate.c[1]==(double)state.mass[1]/(.028*.001)){
    long double expected=step.h*(1-(double(1)-1/std::sqrt(2.)))*rate.production[1];
    long double scale=std::max(std::abs(expected),std::abs(second.rhs[0]));if(std::abs(expected-second.rhs[0])<=1e-12L*scale)rhsHasSelectedState=true;
   }
  }
  CHECK(rhsHasSelectedState);
 }

}
// Enumerate only genuine successful SDIRK edges whose exact base state and T
// descend from a previous completed edge. Speculative full and half branches
// remain distinct; no declared status or audit value creates an edge.
long double coveredBy(const std::vector<long double>& base,long double T,const std::vector<long double>& target,long double targetT,std::size_t begin=0,long double used=0){
 long double answer=-1;
 for(std::size_t i=begin;i<steps.size();++i){const auto& s=steps[i];if(!s.success || s.h<=0 || !same(s.before,base)||s.Tin!=T)continue;
  long double now=used+s.h;if(s.after==target && s.Tout==targetT)answer=std::max(answer,now);
  answer=std::max(answer,coveredBy(s.after,s.Tout,target,targetT,i+1,now));
 }
 return answer;
}
bool actualPath(const std::vector<long double>& base,long double T,const independent::Consumption& c,std::size_t begin,long double left){
 for(std::size_t i=begin;i<c.endStep;++i){const auto& s=steps[i];if(!s.success || s.h<=0 || s.h>left || s.before!=base || s.Tin!=T)continue;
  long double remaining=left-s.h;if(remaining==0 && s.after==c.state && s.Tout==c.T)return true;
  if(remaining>0 && actualPath(s.after,s.Tout,c,i+1,remaining))return true;
 }return false;
}
int main(int argc,char**argv){
 std::cout<<std::setprecision(21);const std::string mode=argc>1?argv[1]:"ordinary";Model model;
 ClosedReactorInput<double,2> base;base.speciesMass[0]=.028;base.totalMass=.028;base.volume=.001;base.internalEnergy=mixtureEnergy(base.speciesMass,700.,model.t);
 ChemistryControls<double> controls;controls.relativeTolerance=.05;controls.absoluteMassFractionTolerance=1e-10;controls.absoluteTemperatureTolerance=1e-4;
 ClosedReactorResult<double,2> out;out.speciesMass[0]=out.speciesMass[1]=-999;out.temperature=-999;
 double interval=.1;bool direct=false,nontrivial=true;independent::reset();
 if(mode=="full_success_half_failure"){independent::failedSteps={2};controls.maximumRejectedSteps=0;}
 else if(mode=="full_failure_two_half_success"){independent::failedSteps={1};controls.maximumRejectedSteps=0;}
 else if(mode=="first_half_only_then_remainder"){independent::failedSteps={1,3};controls.maximumRejectedSteps=0;}
 else if(mode=="full_success_second_half_failure"){independent::failedSteps={3};controls.maximumRejectedSteps=0;}
 else if(mode=="newton_zero"){controls.maximumNewtonIterations=0;direct=true;}
 else if(mode=="linesearch_zero"){controls.maximumLineSearchIterations=0;direct=true;}
 else if(mode=="both_zero"){controls.maximumNewtonIterations=controls.maximumLineSearchIterations=0;direct=true;}
 else if(mode=="refinement_one"){controls.maximumNewtonIterations=controls.maximumLineSearchIterations=1;direct=true;}
 else if(mode=="ad_only_failure"){direct=true;model.rx[0].highRate.preExponential=2;model.rx[0].highRate.temperatureExponent=0;model.rx[1]=model.rx[0];model.rx[1].reactantOffset=model.rx[1].productOffset=1;model.rx[1].highRate.preExponential=1;model.re[1].coefficient=model.pr[1].coefficient=.5;model.m.reactionCount=model.m.reactantTermCount=model.m.productTermCount=2;controls.maximumNewtonIterations=1;controls.maximumLineSearchIterations=1;}
 else if(mode=="singular_jacobian"||mode=="failed_armijo"){direct=true;model.coef[5]=0;model.rx[0].reactantCount=2;model.m.reactantTermCount=2;model.pr[0].coefficient=2;model.rx[0].highRate.preExponential=1;model.rx[0].highRate.temperatureExponent=0;base.speciesMass[0]=.021;base.speciesMass[1]=.007;base.internalEnergy=mixtureEnergy(base.speciesMass,700.,model.t);controls.maximumNewtonIterations=1;controls.maximumLineSearchIterations=1;double z[2]={},f[2],j[4],T=700;auto ev=evaluateConservativeReactor(base,z,model.t,model.m,controls,f,T,j);CHECK(ev);double gamma=1-1/std::sqrt(2.);interval=mode=="singular_jacobian"?(1/j[0])/gamma:.006;std::cout<<"initial_jacobian="<<j[0]<<" scalar_rhs="<<f[0]<<" stage_matrix="<<1-(gamma*interval)*j[0]<<"\n";if(mode=="singular_jacobian")CHECK(1-(gamma*interval)*j[0]==0);independent::reset();}
 else if(mode=="inversion_local_stop"){model.coef[1]=model.coef[4]=.1;base.internalEnergy=mixtureEnergy(base.speciesMass,700.,model.t);controls.thermo.maximumIterations=2;controls.maximumNewtonIterations=1;controls.maximumLineSearchIterations=1;}
 else if(mode=="tiny_unpaid_tail"){interval=1;controls.initialStep=.3;controls.maximumSteps=1;}
 else if(mode=="absorbed_remaining"){interval=1;controls.initialStep=std::ldexp(1.,-55);controls.maximumSteps=1;}
 else if(mode=="unpaid_tail"){interval=1;controls.initialStep=controls.maximumStep=.1;controls.maximumRejectedSteps=0;}
 else if(mode=="step_budget_zero"){controls.maximumSteps=0;}
 else if(mode=="remainder_budget"){interval=.95;controls.initialStep=.1;controls.maximumStep=.3;controls.maximumSteps=1;}
 else if(mode=="persistent_lte"){interval=1;controls.initialStep=.1;controls.relativeTolerance=1e-15;controls.absoluteMassFractionTolerance=1e-25;controls.absoluteTemperatureTolerance=1e-15;controls.maximumSteps=4;controls.maximumRejectedSteps=0;}
 else if(mode=="undefined_lte"){controls.relativeTolerance=controls.absoluteMassFractionTolerance=controls.absoluteTemperatureTolerance=0;controls.maximumRejectedSteps=0;}
 else if(mode=="below_minimum"){interval=.3;controls.initialStep=.1;controls.minimumStep=.2;controls.maximumRejectedSteps=0;}
 else if(mode.rfind("subnormal",0)==0){interval=std::numeric_limits<double>::denorm_min()*(mode=="subnormal"?1:std::stoi(mode.substr(10)));model.coef[5]=0;model.rx[0].highRate.preExponential=1e100;model.rx[0].highRate.temperatureExponent=0;base.internalEnergy=mixtureEnergy(base.speciesMass,700.,model.t);controls.maximumRejectedSteps=0;}
 else if(mode=="absorbed_update"){interval=1e-20;base.speciesMass[0]=base.speciesMass[1]=.014;model.coef[5]=0;base.internalEnergy=mixtureEnergy(base.speciesMass,700.,model.t);nontrivial=false;}
 else if(mode=="equilibrium"){model.rx[0].highRate.preExponential=0;nontrivial=false;}
 else if(mode=="zero_interval"){interval=0;nontrivial=false;}
 else if(mode!="ordinary"){std::cerr<<"Unknown case\n";return 2;}
 ChemistryStatus st=direct?gasChemistryDetail::sdirkStep(base,interval,model.t,model.m,controls,out,700.):advanceClosedReactor(base,interval,model.t,model.m,controls,out);
 std::cout<<"CASE "<<mode<<" success="<<bool(st)<<" code="<<int(st.code)<<" attempts="<<st.attemptedSteps<<" rejected="<<st.rejectedSteps<<" mass="<<out.speciesMass[0]<<","<<out.speciesMass[1]<<" T="<<out.temperature<<"\n";
 independent::dump();
#ifdef ENDPOINT_ONLY
 return 0;
#endif
 CHECK(st);validateStageLineage();
 if(st){CHECK(out.speciesMass[0]!=-999 && out.speciesMass[1]!=-999 && out.temperature!=-999);if(nontrivial){if(mode=="failed_armijo")CHECK(out.speciesMass[1]!=base.speciesMass[1]);else CHECK(out.speciesMass[1]>base.speciesMass[1]);}
  if(direct){CHECK(stages.size()>=2);CHECK(independent::counters.rates>=3);for(const auto& stage:stages)CHECK(stage.after!=stage.before);}
  else {if(nontrivial){long double covered=0;auto prior=independent::values(base.speciesMass);long double Tprior=steps[0].Tin;std::size_t begin=0;
    for(const auto& consumption:independent::consumed){CHECK(actualPath(prior,Tprior,consumption,begin,consumption.h));covered+=consumption.h;prior=consumption.state;Tprior=consumption.T;begin=consumption.endStep;}
    std::cout<<"computed_lineage_duration="<<covered<<" requested="<<interval<<"\n";CHECK(std::abs(covered-(long double)interval)<4*std::numeric_limits<long double>::epsilon()*interval);CHECK(prior==independent::values(out.speciesMass)&&Tprior==out.temperature);}}
 }
 if(mode=="full_success_half_failure"||mode=="full_success_second_half_failure"){CHECK(steps.size()>=2);if(st&&steps.size()>=2){CHECK(steps[0].success);CHECK(outputIs(out,steps[0]));}}
 if(mode=="full_failure_two_half_success"){CHECK(steps.size()>=3);if(st&&steps.size()>=3){CHECK(!steps[0].success&&steps[1].success&&steps[2].success);CHECK(outputIs(out,steps[2]));CHECK(steps[1].after==steps[2].before);}}
 if(mode=="first_half_only_then_remainder"){CHECK(steps.size()>=4);if(st&&steps.size()>=4){CHECK(!steps[0].success&&steps[1].success&&!steps[2].success);CHECK(steps[3].before==steps[1].after);CHECK(steps[3].h==(long double)interval-steps[1].h);}}
 if(mode.rfind("subnormal",0)==0&&st){double expected=.028*(interval*1e100);std::cout<<"analytic_first_order_product="<<expected<<" actual="<<out.speciesMass[1]<<"\n";CHECK(out.speciesMass[1]>0);CHECK(std::abs(out.speciesMass[1]/expected-1)<1e-12);for(auto& s:steps)if(s.success)CHECK(s.h>0);if(!stages.empty()){double expectedFirst=(interval*((1-1/std::sqrt(2.))*1e100))*1000;CHECK(stages[0].after[0]>0);CHECK(std::abs(stages[0].after[0]/expectedFirst-1)<1e-12);}}
 if(mode=="ad_only_failure"&&st){double expected=(1-1/std::sqrt(2.))*interval*2000;CHECK(std::abs(stages[0].after[0]/expected-1)<1e-12);CHECK(independent::counters.derivatives>=1);}
 if(mode=="equilibrium"&&st){CHECK(out.speciesMass[0]==base.speciesMass[0] && out.speciesMass[1]==base.speciesMass[1]);CHECK(stages.size()>=2);CHECK(independent::counters.rates>0);}
 if(mode=="persistent_lte"||mode=="undefined_lte")CHECK(steps.size()<=100);
 if(mode=="absorbed_update"&&st){CHECK(std::memcmp(out.speciesMass,base.speciesMass,sizeof out.speciesMass)==0);CHECK(stages.size()>=2);CHECK(independent::counters.rates>0);}
 if(mode=="tiny_unpaid_tail"&&st){CHECK(independent::consumed.size()>=3);CHECK(independent::consumed.back().h>0&&independent::consumed.back().h<std::numeric_limits<double>::epsilon());}
 if(mode=="absorbed_remaining"&&st){CHECK(independent::consumed.size()>=3);CHECK(independent::consumed.front().h<std::numeric_limits<double>::epsilon());}
 if(mode=="zero_interval"){CHECK(independent::counters.prepare==0);CHECK(independent::counters.basis==0);CHECK(steps.empty());}
 else if(!direct){CHECK(independent::counters.prepare==1);CHECK(independent::counters.basis==1);}
 std::cout<<"RESULT failures="<<failures<<"\n";return failures?1:0;
}
