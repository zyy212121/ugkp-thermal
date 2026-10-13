#define main lineage_probe_original_main
#include "lineage_probe.cpp"
#undef main
int main(){
 Model model;model.coef[5]=0;model.rx[0].reactantCount=2;model.m.reactantTermCount=2;model.pr[0].coefficient=2;model.rx[0].highRate.preExponential=1;model.rx[0].highRate.temperatureExponent=0;
 ClosedReactorInput<double,2> base;base.totalMass=.028;base.volume=.001;base.speciesMass[0]=.021;base.speciesMass[1]=.007;base.internalEnergy=mixtureEnergy(base.speciesMass,700.,model.t);
 ChemistryControls<double> c;c.relativeTolerance=.05;c.maximumNewtonIterations=1;c.maximumLineSearchIterations=3;
 const double h=.006,gamma=1-1/std::sqrt(2.);double z[2]={},f[2],j[4],T=700;auto evaluated=evaluateConservativeReactor(base,z,model.t,model.m,c,f,T,j);CHECK(evaluated);double expected=.25*((h*gamma*f[0])/(1-h*gamma*j[0]));
 ClosedReactorResult<double,2> result;independent::reset();auto status=gasChemistryDetail::sdirkStep(base,h,model.t,model.m,c,result,700.);CHECK(status);CHECK(stages.size()==2);
 if(stages.size()==2){const auto& first=stages[0];CHECK(first.lastState-first.firstState==4);CHECK(std::abs(first.after[0]/expected-1)<1e-12);CHECK(first.after!=first.before);validateStageLineage();}
 std::cout<<std::setprecision(21)<<"expected_best_first_stage="<<expected<<" actual="<<(stages.empty()?0:stages[0].after[0])<<" output="<<result.speciesMass[0]<<","<<result.speciesMass[1]<<"\n";independent::dump();std::cout<<"RESULT failures="<<failures<<"\n";return failures?1:0;
}
