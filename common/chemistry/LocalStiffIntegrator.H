#ifndef UGKWP_LOCAL_STIFF_INTEGRATOR_H
#define UGKWP_LOCAL_STIFF_INTEGRATOR_H
#include "ConservativeReactor.H"
namespace ugkwp { namespace gasChemistryDetail {
UGKWP_GAS_HD inline bool hasUsableChemistryResult(const ChemistryStatus& status) noexcept {
 return status.code==ChemistryCode::Success || status.code==ChemistryCode::NewtonFailure
     || status.code==ChemistryCode::ConservationFailure
     || (status.code==ChemistryCode::TemperatureFailure && status.thermoCode==ThermoStatus::IterationLimit);
}
UGKWP_GAS_HD inline void retainDiagnostic(ChemistryStatus& output,const ChemistryStatus& input) noexcept {
 if(output && !input){output.linearFailure=input.linearFailure;output.code=input.code;output.thermoCode=input.thermoCode;output.reaction=input.reaction;output.observed=input.observed;output.reference=input.reference;output.nonlinearIteration=input.nonlinearIteration;}
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline R coordinateNorm(const ReactorContext<R,N,TableReal>& context,const R* residual,const R* mass) noexcept {
 R norm=0;int rank=context.mechanism.independentRank;
 for(int s=0;s<N;++s){R value=0;for(int j=0;j<rank;++j)value+=context.mechanism.stoichiometricBasis[s*rank+j]*residual[j];value*=context.thermo.species[s].molarMass*context.base.volume;
 R denominator=context.controls.absoluteMassFractionTolerance*context.base.totalMass+context.controls.relativeTolerance*gasThermoDetail::maxValue(mass[s],context.base.speciesMass[s]);
 norm=gasThermoDetail::maxValue(norm,gasThermoDetail::absValue(value)/denominator);}
 return norm;
}

template<class R,class Time> UGKWP_GAS_HD inline R stageProduct(Time h,R coefficient,R value) noexcept {
 const R hc=R(h*Time(coefficient));
 if(R(h)!=R(0) && gasFinite(R(h)))return hc*value;
 return R(h*(Time(coefficient)*Time(value)));
}
template<class R,int N,class TableReal,class Time> UGKWP_GAS_HD UGKWP_CHEMISTRY_DEVICE_LEAF ChemistryStatus implicitStage
(const ReactorContext<R,N,TableReal>& context,const R* rhs,Time interval,R coefficient,R* z,R& temperature) noexcept {
 ChemistryStatus status;int rank=context.mechanism.independentRank;
 for(int iteration=0;iteration<(context.controls.maximumNewtonIterations>0?context.controls.maximumNewtonIterations:1);++iteration){
   R f[N]={},jac[N*N]={},residual[N]={},masses[N];


   ChemistryStatus local=context.evaluate(z,f,temperature,jac,masses);
   if(!hasUsableChemistryResult(local)){return local;}
   retainDiagnostic(status,local);
   for(int i=0;i<rank;++i)residual[i]=z[i]-rhs[i]-stageProduct(interval,coefficient,f[i]);
   R norm=coordinateNorm(context,residual,masses);
   status.observed=double(norm);status.reference=double(context.controls.newtonTolerance);status.nonlinearIteration=iteration;
   // Do not freeze trace donors by accepting an unchanged initial guess.
   // One actual Newton update retains stiff relaxation below the error floor.
   if(norm<=context.controls.newtonTolerance && (iteration>0 || norm==R(0)))return status;
   R delta[N]={};for(int i=0;i<rank;++i){delta[i]=-residual[i];for(int j=0;j<rank;++j)jac[i*N+j]=(i==j?R(1):R(0))-stageProduct(interval,coefficient,jac[i*N+j]);}
   if(!solveDense<R,N,true>(rank,jac,delta,nullptr,&status.linearFailure)){status.code=ChemistryCode::SingularJacobian;return status;}
   // A Newton update is part of the single implicit solve. Never retry a
   // failed state evaluator with a smaller candidate or another algorithm.
   for(int i=0;i<rank;++i)z[i]+=delta[i];

 }
 status.code=ChemistryCode::NewtonFailure;return status;
}

template<class R,int N,class TableReal> UGKWP_GAS_HD inline ChemistryStatus implicitStage
(const ReactorContext<R,N,TableReal>& context,const R* rhs,R hgamma,R* z,R& temperature) noexcept {
 return implicitStage(context,rhs,hgamma,R(1),z,temperature);
}
template<class R,int N,class TableReal,class Time> UGKWP_GAS_HD inline ChemistryStatus sdirkStep
(const ReactorContext<R,N,TableReal>& context,Time interval,ClosedReactorResult<R,N>& result,R initialTemperature,bool* evaluated=nullptr) noexcept {
 if(evaluated)*evaluated=false;
 ChemistryStatus st;
 const R gamma=R(1)-R(1)/::sqrt(R(2));R z[N]={},rhs[N]={},f[N]={},temperature=initialTemperature;
 st=implicitStage(context,rhs,interval,gamma,z,temperature);if(!hasUsableChemistryResult(st))return st;
 ChemistryStatus eval=context.evaluate(z,f,temperature,nullptr);if(!hasUsableChemistryResult(eval)){return eval;}
 retainDiagnostic(st,eval);
 for(int i=0;i<context.mechanism.independentRank;++i)rhs[i]=stageProduct(interval,R(1)-gamma,f[i]);
 ChemistryStatus second=implicitStage(context,rhs,interval,gamma,z,temperature);if(!hasUsableChemistryResult(second))return second;
 retainDiagnostic(second,st);
 if(!context.state(z,result.speciesMass,temperature)){second.code=ChemistryCode::TemperatureFailure;return second;}
 result.temperature=temperature;if(evaluated)*evaluated=true;return second;
}
template<class R,int N,class TableReal,class Time> UGKWP_GAS_HD inline ChemistryStatus sdirkStep
(const ClosedReactorInput<R,N>& base,Time interval,const SpeciesThermoView<TableReal,N>& t,const GasMechanismView<TableReal,N>& m,const ChemistryControls<R>& controls,ClosedReactorResult<R,N>& result,R initialTemperature) noexcept {
 ReactorContext<R,N,TableReal> context(base,t,m,controls);ChemistryStatus st;
 if(!context.prepare(nullptr,&st.linearFailure)){st.code=ChemistryCode::InvalidModel;return st;}
 return sdirkStep(context,interval,result,initialTemperature);
}
}}
namespace ugkwp {
template<class R,int N,class TableReal,class Time> UGKWP_GAS_HD inline ChemistryStatus advanceClosedReactor
(const ClosedReactorInput<R,N>& base,Time interval,const SpeciesThermoView<TableReal,N>& t,const GasMechanismView<TableReal,N>& m,const ChemistryControls<R>& controls,ClosedReactorResult<R,N>& result,bool* evaluated=nullptr) noexcept {
 if(evaluated)*evaluated=false;
 using namespace gasChemistryDetail;
 ChemistryStatus status;
 if(!gasFinite(interval) || interval<Time(0) || !gasFinite(base.internalEnergy)
    || !gasFinite(base.totalMass) || !gasFinite(base.volume)){
   status.code=ChemistryCode::NonFiniteInput;return status;
 }
 R temperature=R(0);bool temperatureAvailable=false;
 status.thermoCode=invertMixtureEnergy(base.speciesMass,base.internalEnergy,t,controls.thermo,temperature,&temperatureAvailable);
 if(!temperatureAvailable){status.code=ChemistryCode::TemperatureFailure;return status;}
 if(status.thermoCode!=ThermoStatus::Success)status.code=ChemistryCode::TemperatureFailure;
 if(interval==Time(0)){
   for(int s=0;s<N;++s)result.speciesMass[s]=base.speciesMass[s];
   result.temperature=temperature;if(evaluated)*evaluated=true;return status;
 }
 ReactorContext<R,N,TableReal> context(base,t,m,controls,false);
 if(!context.prepare(nullptr,&status.linearFailure)){status.code=ChemistryCode::InvalidModel;return status;}
 bool available=false;
 ChemistryStatus step=sdirkStep(context,interval,result,temperature,&available);
 step.attemptedSteps=1;
 retainDiagnostic(step,status);
 if(evaluated)*evaluated=available;
 return step;
}
}
#endif
