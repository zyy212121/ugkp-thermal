#ifndef UGKWP_LOCAL_STIFF_INTEGRATOR_H
#define UGKWP_LOCAL_STIFF_INTEGRATOR_H
#include "ConservativeReactor.H"
namespace ugkwp { namespace gasChemistryDetail {
UGKWP_GAS_HD inline void retainDiagnostic(ChemistryStatus& output,const ChemistryStatus& input) noexcept {
 if(output && !input){output.code=input.code;output.thermoCode=input.thermoCode;output.reaction=input.reaction;}
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline R coordinateNorm(const ReactorContext<R,N,TableReal>& context,const R* residual,const R* mass) noexcept {
 R norm=0;int rank=context.mechanism.independentRank;
 for(int s=0;s<N;++s){R value=0;for(int j=0;j<rank;++j)value+=context.mechanism.stoichiometricBasis[s*rank+j]*residual[j];value*=context.thermo.species[s].molarMass*context.base.volume;
 R denominator=context.controls.absoluteMassFractionTolerance*context.base.totalMass+context.controls.relativeTolerance*gasThermoDetail::maxValue(mass[s],context.base.speciesMass[s]);
 norm=gasThermoDetail::maxValue(norm,gasThermoDetail::absValue(value)/denominator);}
 return norm;
}
// Only durations refused by the old storage cast need wider multiplication.
template<class R,class Time> UGKWP_GAS_HD inline R stageProduct(Time h,R coefficient,R value) noexcept {
 const R hc=R(h*Time(coefficient));
 if(R(h)!=R(0) && gasFinite(R(h)))return hc*value;
 return R(h*(Time(coefficient)*Time(value)));
}
template<class R,int N,class TableReal,class Time> UGKWP_GAS_HD UGKWP_CHEMISTRY_DEVICE_LEAF ChemistryStatus continueImplicitStage
(const ReactorContext<R,N,TableReal>& context,const R* rhs,Time h,R coefficient,
 R* z,R& temperature,R* masses,R* f,bool& scalarAvailable,bool reuseState=false,bool* evaluated=nullptr) noexcept {
 if(evaluated)*evaluated=false;
 ChemistryStatus status;const int rank=context.mechanism.independentRank;
 bool updated=false;if(!reuseState)scalarAvailable=false;
 const int iterations=context.controls.maximumNewtonIterations>0?context.controls.maximumNewtonIterations:1;
 const int searches=context.controls.maximumLineSearchIterations>0?context.controls.maximumLineSearchIterations:1;
 for(int iteration=0;iteration<iterations;++iteration){
   R jac[N*N]={},residual[N]={};
   ChemistryStatus local;
   if(iteration==0 && !reuseState)local=context.evaluate(z,f,temperature,jac,masses,&scalarAvailable);
   else local=context.evaluateJacobian(masses,temperature,jac);

   retainDiagnostic(status,local);
   if(!scalarAvailable){return local;}
   bool zero=true;
   for(int i=0;i<rank;++i){residual[i]=z[i]-rhs[i]-stageProduct(h,coefficient,f[i]);zero=zero && residual[i]==R(0);}
   const R norm=coordinateNorm(context,residual,masses);
   if(zero || (updated && gasFinite(norm) && norm<=context.controls.newtonTolerance)){if(evaluated)*evaluated=true;return status;}
   R delta[N]={};
   for(int i=0;i<rank;++i){delta[i]=-residual[i];for(int j=0;j<rank;++j)jac[i*N+j]=(i==j?R(1):R(0))-stageProduct(h,coefficient,jac[i*N+j]);}
   if(!local || !solveDense<R,N>(rank,jac,delta)){
     if(updated){if(evaluated)*evaluated=true;return status;}
     // This is an actual fixed-point stage update, only on a failed Newton path.
     for(int i=0;i<rank;++i)delta[i]=-residual[i];
   }
   R origin[N],originT=temperature;for(int i=0;i<rank;++i)origin[i]=z[i];
   bool found=updated,accepted=false;R bestNorm=norm,damping=1;
   ChemistryStatus failed;failed.code=ChemistryCode::NewtonFailure;
   for(int search=0;search<searches;++search){
     R trial[N]={},trialMass[N],trialT=originT,trialF[N]={},trialResidual[N]={};
     for(int i=0;i<rank;++i)trial[i]=origin[i]+damping*delta[i];
     ThermoStatus trialThermo=ThermoStatus::Success;
     if(context.state(trial,trialMass,trialT,&trialThermo)){
       bool trialScalar=false;
       ChemistryStatus attempt=context.evaluateState(trialMass,trialF,trialT,nullptr,&trialScalar);
       if(trialThermo!=ThermoStatus::Success && attempt){attempt.code=ChemistryCode::TemperatureFailure;attempt.thermoCode=trialThermo;}
       R trialNorm=gasThermoDetail::invalidValue<R>();
       if(trialScalar){for(int i=0;i<rank;++i)trialResidual[i]=trial[i]-rhs[i]-stageProduct(h,coefficient,trialF[i]);trialNorm=coordinateNorm(context,trialResidual,trialMass);}
       const bool armijo=trialScalar && gasFinite(trialNorm)
         && ((gasFinite(norm) && trialNorm<norm*(R(1)-R(1e-4)*damping)) || trialNorm<=context.controls.newtonTolerance);
       // Required RHS availability precedes optional residual ranking. A finite
       // final-stage state remains usable when no trial has an available RHS.
       if(armijo || !found || (trialScalar && !scalarAvailable)
          || (gasFinite(trialNorm) && (!gasFinite(bestNorm) || trialNorm<bestNorm))){
         for(int i=0;i<rank;++i){z[i]=trial[i];f[i]=trialF[i];}
         for(int s=0;s<N;++s)masses[s]=trialMass[s];
         temperature=trialT;scalarAvailable=trialScalar;bestNorm=trialNorm;found=true;
       }
       retainDiagnostic(status,attempt);
       if(armijo){accepted=true;break;}
       if(!attempt)failed=attempt;
     } else failed.code=ChemistryCode::TemperatureFailure;
     damping*=R(.5);
   }
   if(!found){return failed;}
   updated=true;
   if(!accepted || !scalarAvailable){if(evaluated)*evaluated=scalarAvailable;return status;}
 }
 if(evaluated)*evaluated=scalarAvailable;
 return status;
}
template<class R,int N,class TableReal,class Time> UGKWP_GAS_HD UGKWP_CHEMISTRY_DEVICE_LEAF ChemistryStatus implicitStage
(const ReactorContext<R,N,TableReal>& context,const R* rhs,Time interval,R coefficient,R* z,R& temperature) noexcept {
 ChemistryStatus status;int rank=context.mechanism.independentRank;
 for(int iteration=0;iteration<(context.controls.maximumNewtonIterations>0?context.controls.maximumNewtonIterations:1);++iteration){
   R f[N]={},jac[N*N]={},residual[N]={},masses[N];
   // Successful strict inversion already returned this exact evaluated T and
   // mass bundle. Re-evaluating the same state accepts the same T immediately.
   ChemistryStatus local=context.evaluate(z,f,temperature,jac,masses);
   if(!local){return local;}
   for(int i=0;i<rank;++i)residual[i]=z[i]-rhs[i]-stageProduct(interval,coefficient,f[i]);
   R norm=coordinateNorm(context,residual,masses);
   // Do not freeze trace donors by accepting an unchanged initial guess.
   // One actual Newton update retains stiff relaxation below the error floor.
   if(norm<=context.controls.newtonTolerance && (iteration>0 || norm==R(0)))return status;
   R delta[N]={};for(int i=0;i<rank;++i){delta[i]=-residual[i];for(int j=0;j<rank;++j)jac[i*N+j]=(i==j?R(1):R(0))-stageProduct(interval,coefficient,jac[i*N+j]);}
   if(!solveDense<R,N,true>(rank,jac,delta)){status.code=ChemistryCode::SingularJacobian;return status;}
   bool accepted=false;R damping=1;
   for(int search=0;search<(context.controls.maximumLineSearchIterations>0?context.controls.maximumLineSearchIterations:1);++search){R trial[N]={},trialMass[N],trialT=temperature,trialF[N]={},trialResidual[N]={};
     for(int i=0;i<rank;++i)trial[i]=z[i]+damping*delta[i];
     if(context.state(trial,trialMass,trialT)){
       ChemistryStatus attempt=context.evaluateState(trialMass,trialF,trialT,nullptr);
       if(attempt){for(int i=0;i<rank;++i)trialResidual[i]=trial[i]-rhs[i]-stageProduct(interval,coefficient,trialF[i]);R trialNorm=coordinateNorm(context,trialResidual,trialMass);
         if(trialNorm<norm*(R(1)-R(1e-4)*damping) || trialNorm<=context.controls.newtonTolerance){for(int i=0;i<rank;++i)z[i]=trial[i];temperature=trialT;accepted=true;break;}}
     }
     damping*=R(.5);
   }
   if(!accepted){status.code=ChemistryCode::NewtonFailure;return status;}
 }
 status.code=ChemistryCode::NewtonFailure;return status;
}
// Direct stage API retains its original hgamma argument.
template<class R,int N,class TableReal> UGKWP_GAS_HD inline ChemistryStatus implicitStage
(const ReactorContext<R,N,TableReal>& context,const R* rhs,R hgamma,R* z,R& temperature) noexcept {
 return implicitStage(context,rhs,hgamma,R(1),z,temperature);
}
template<class R,int N,class TableReal,class Time> UGKWP_GAS_HD inline ChemistryStatus sdirkStep
(const ReactorContext<R,N,TableReal>& context,Time interval,ClosedReactorResult<R,N>& result,R initialTemperature,bool* evaluated=nullptr) noexcept {
 if(evaluated)*evaluated=false;
 ChemistryStatus st;
 if(!context.refinement){
   const R gamma=R(1)-R(1)/::sqrt(R(2));R z[N]={},rhs[N]={},f[N]={},masses[N],temperature=initialTemperature;
   bool scalarAvailable=false;
   bool stageAvailable=false;
   st=continueImplicitStage(context,rhs,interval,gamma,z,temperature,masses,f,scalarAvailable,false,&stageAvailable);if(!stageAvailable)return st;
   if(!scalarAvailable){st.code=ChemistryCode::RateFailure;return st;}
   for(int i=0;i<context.mechanism.independentRank;++i)rhs[i]=stageProduct(interval,R(1)-gamma,f[i]);
   ChemistryStatus second=continueImplicitStage(context,rhs,interval,gamma,z,temperature,masses,f,scalarAvailable,true,&stageAvailable);
   retainDiagnostic(second,st);if(!stageAvailable)return second;
   for(int i=0;i<N;++i)result.speciesMass[i]=masses[i];
   result.temperature=temperature;if(evaluated)*evaluated=true;return second;
 }
 const R gamma=R(1)-R(1)/::sqrt(R(2));R z[N]={},rhs[N]={},f[N]={},temperature=initialTemperature;
 st=implicitStage(context,rhs,interval,gamma,z,temperature);if(!st)return st;
 ChemistryStatus eval=context.evaluate(z,f,temperature,nullptr);if(!eval){return eval;}
 for(int i=0;i<context.mechanism.independentRank;++i)rhs[i]=stageProduct(interval,R(1)-gamma,f[i]);
 ChemistryStatus second=implicitStage(context,rhs,interval,gamma,z,temperature);if(!second)return second;
 if(!context.state(z,result.speciesMass,temperature)){second.code=ChemistryCode::TemperatureFailure;return second;}
 result.temperature=temperature;if(evaluated)*evaluated=true;return second;
}
template<class R,int N,class TableReal,class Time> UGKWP_GAS_HD inline ChemistryStatus sdirkStep
(const ClosedReactorInput<R,N>& base,Time interval,const SpeciesThermoView<TableReal,N>& t,const GasMechanismView<TableReal,N>& m,const ChemistryControls<R>& controls,ClosedReactorResult<R,N>& result,R initialTemperature) noexcept {
 ReactorContext<R,N,TableReal> context(base,t,m,controls);ChemistryStatus st;
 if(!context.prepare()){st.code=ChemistryCode::InvalidModel;return st;}
 return sdirkStep(context,interval,result,initialTemperature);
}
template<class R,int N> UGKWP_GAS_HD inline R stepError
(const ClosedReactorInput<R,N>& base,const ClosedReactorResult<R,N>& a,const ClosedReactorResult<R,N>& b,const ChemistryControls<R>& c) noexcept {
 R error=0;for(int s=0;s<N;++s){R scale=c.absoluteMassFractionTolerance*base.totalMass+c.relativeTolerance*gasThermoDetail::maxValue(a.speciesMass[s],b.speciesMass[s]);error=gasThermoDetail::maxValue(error,gasThermoDetail::absValue(a.speciesMass[s]-b.speciesMass[s])/(R(3)*scale));}
 R scale=c.absoluteTemperatureTolerance+c.relativeTolerance*gasThermoDetail::maxValue(a.temperature,b.temperature);
 return gasThermoDetail::maxValue(error,gasThermoDetail::absValue(a.temperature-b.temperature)/(R(3)*scale));
}
}}
namespace ugkwp {
template<bool RetryTrial=false,class R,int N,class TableReal,class Time> UGKWP_GAS_HD inline ChemistryStatus advanceClosedReactorImpl
(const ClosedReactorInput<R,N>& base,Time interval,const SpeciesThermoView<TableReal,N>& t,const GasMechanismView<TableReal,N>& m,const ChemistryControls<R>& controls,ClosedReactorResult<R,N>& result,bool* evaluated=nullptr) noexcept {
 if(evaluated)*evaluated=false;
 using namespace gasChemistryDetail;using gasThermoDetail::absValue;using gasThermoDetail::maxValue;using gasThermoDetail::minValue;
 ChemistryStatus status;
 if(RetryTrial){
   if(!chemistryTrialControls(controls)){status.code=ChemistryCode::InvalidControls;return status;}
   if(!gasFinite(base.totalMass) || !(base.totalMass>R(0)) || !gasFinite(base.volume) || !(base.volume>R(0))){status.code=ChemistryCode::NonFiniteInput;return status;}
   R sum=0;for(int s=0;s<N;++s){if(!gasFinite(base.speciesMass[s]) || base.speciesMass[s]<R(0)){status.code=ChemistryCode::InvalidComposition;return status;}sum+=base.speciesMass[s];}
   if(absValue(sum-base.totalMass)>controls.absoluteMassTolerance+controls.relativeConservationTolerance*base.totalMass){status.code=ChemistryCode::ConservationFailure;return status;}
 }
 const bool terminalInterval=!gasFinite(interval) || interval<Time(0);
 if(terminalInterval || !gasFinite(base.internalEnergy)){status.code=ChemistryCode::NonFiniteInput;if(RetryTrial)return status;}
 R temperature=!RetryTrial && controls.thermo.maximumIterations<=0?gasThermoDetail::invalidValue<R>():R(0);bool temperatureAvailable=false;
 status.thermoCode=invertMixtureEnergy<RetryTrial>(base.speciesMass,base.internalEnergy,t,controls.thermo,temperature,&temperatureAvailable);
 if(status.thermoCode!=ThermoStatus::Success){
   status.code=RetryTrial && status.thermoCode==ThermoStatus::InvalidModel
     ?ChemistryCode::InvalidModel:ChemistryCode::TemperatureFailure;
   if(RetryTrial || !temperatureAvailable)return status;
 }
 ClosedReactorInput<R,N> current=base;ClosedReactorResult<R,N> accepted;for(int s=0;s<N;++s)accepted.speciesMass[s]=base.speciesMass[s];accepted.temperature=temperature;
 Time covered=0;
 ReactorContext<R,N,TableReal> context(current,t,m,controls);
 if(!RetryTrial && (terminalInterval || !status))context.refinement=false;
 if(interval!=Time(0)){
   bool prepared=false;
   if(!context.prepare(&prepared)){
     if(!RetryTrial && context.refinement){context.refinement=false;context.prepare(&prepared);}
     if(!prepared){status.code=ChemistryCode::InvalidModel;return status;}
     if(status)status.code=ChemistryCode::InvalidModel;
   }
 }
 Time h=controls.initialStep>R(0)?minValue(interval,Time(controls.initialStep)):interval;if(controls.maximumStep>R(0))h=minValue(h,Time(controls.maximumStep));
 bool terminalPending=terminalInterval;
 // Preserve the failed producer when a later effort boundary ends this trial.
 // These diagnostics do not alter the step arithmetic or publication decision.
 ChemistryStatus rejectedDiagnostic;
 if(terminalInterval)h=interval;
 while(covered<interval || terminalPending){
   if(status.attemptedSteps>=controls.maximumSteps){
     if(RetryTrial){status.code=ChemistryCode::StepLimit;status.reaction=rejectedDiagnostic.reaction;status.thermoCode=rejectedDiagnostic.thermoCode;return status;}
     if(context.refinement)h=interval-covered;
     context.refinement=false;
   }
   if(!terminalPending)h=minValue(h,interval-covered);
   if(!terminalPending && (!(h>Time(0)) || covered+h==covered)){
     if(RetryTrial){status.code=ChemistryCode::StepUnderflow;status.reaction=rejectedDiagnostic.reaction;status.thermoCode=rejectedDiagnostic.thermoCode;return status;}
     // Execute the actual remaining interval. Do not move the clock to invent
     // an endpoint for a step whose duration rounded away.
     context.refinement=false;h=interval-covered;
   }
   if(h<controls.minimumStep && h<interval-covered){
     if(RetryTrial){status.code=ChemistryCode::StepUnderflow;status.reaction=rejectedDiagnostic.reaction;status.thermoCode=rejectedDiagnostic.thermoCode;return status;}
     if(context.refinement)h=interval-covered;
     context.refinement=false;
   }
   ++status.attemptedSteps;ClosedReactorResult<R,N> full,half,end;
   bool stepAvailable=false;
   ChemistryStatus trial=sdirkStep(context,h,full,temperature,&stepAvailable);
   ChemistryStatus diagnostic=trial;
   if(stepAvailable){trial=sdirkStep(context,h/Time(2),half,temperature,&stepAvailable);retainDiagnostic(diagnostic,trial);}
   if(stepAvailable){ClosedReactorInput<R,N> middle=current;for(int s=0;s<N;++s)middle.speciesMass[s]=half.speciesMass[s];ReactorContext<R,N,TableReal> middleContext(middle,t,m,controls,context.refinement);for(int k=0;k<N*N;++k)middleContext.leftInverse[k]=context.leftInverse[k];trial=sdirkStep(middleContext,h/Time(2),end,half.temperature,&stepAvailable);retainDiagnostic(diagnostic,trial);}
   if(!context.refinement){retainDiagnostic(status,diagnostic);if(!stepAvailable)return status;}
   R error=stepAvailable?stepError(current,full,end,controls):R(2);
   if(!stepAvailable || (context.refinement && (!gasFinite(error) || error>R(1)))){
     ++status.rejectedSteps;rejectedDiagnostic=trial;
     if(status.rejectedSteps>controls.maximumRejectedSteps){
       if(RetryTrial){status.code=trial?ChemistryCode::StepLimit:trial.code;status.reaction=trial.reaction;status.thermoCode=trial.thermoCode;return status;}
       if(context.refinement){context.refinement=false;h=interval-covered;continue;}
     }
     R factor=trial && gasFinite(error)?maxValue(R(.1),R(.8)*R(::pow(error,R(-1)/R(3)))):R(.25);h*=minValue(R(.5),factor);
     if(h<controls.minimumStep){
       if(RetryTrial){status.code=ChemistryCode::StepUnderflow;status.reaction=trial.reaction;status.thermoCode=trial.thermoCode;return status;}
       if(context.refinement){context.refinement=false;h=interval-covered;continue;}
     }
     if(!(h>Time(0))){
       if(RetryTrial){status.code=trial?ChemistryCode::StepUnderflow:trial.code;status.reaction=trial.reaction;status.thermoCode=trial.thermoCode;return status;}
       context.refinement=false;h=interval-covered;
     }continue;
   }
   rejectedDiagnostic=ChemistryStatus{};
   accepted=end;for(int s=0;s<N;++s)current.speciesMass[s]=end.speciesMass[s];temperature=end.temperature;
   covered+=h;terminalPending=false;
   R factor=error>R(0)?minValue(R(3),maxValue(R(.25),R(.9)*R(::pow(error,R(-1)/R(3))))):R(3);if(context.refinement || (gasFinite(error) && error<=R(1)))h*=factor;if(controls.maximumStep>R(0))h=minValue(h,Time(controls.maximumStep));
 }
 // These residuals select a real smaller-step trial. The direct continuation
 // path has no reporting pass or output-only inventory reconstruction.
 if(RetryTrial){
   R sum=0;for(int s=0;s<N;++s)sum+=accepted.speciesMass[s];
   const R massResidual=sum-base.totalMass;
   if(absValue(massResidual)>controls.absoluteMassTolerance+controls.relativeConservationTolerance*base.totalMass){status.code=ChemistryCode::ConservationFailure;return status;}
   if(!(t.elementCount>0 && !t.elementComposition))
   for(int e=0;e<t.elementCount;++e){R before=0,after=0;for(int s=0;s<N;++s){R atoms=R(t.elementComposition[s*t.elementCount+e])/R(t.species[s].molarMass);before+=base.speciesMass[s]*atoms;after+=accepted.speciesMass[s]*atoms;}
     const R residual=absValue(after-before);
     if(residual>controls.absoluteElementTolerance+controls.relativeConservationTolerance*absValue(before)){status.code=ChemistryCode::ConservationFailure;return status;}
   }
   const R energyResidual=mixtureEnergy(accepted.speciesMass,accepted.temperature,t)-base.internalEnergy;
   if(!gasFinite(energyResidual) || absValue(energyResidual)>controls.thermo.absoluteEnergyTolerance+controls.thermo.relativeEnergyTolerance*absValue(base.internalEnergy)){status.code=ChemistryCode::ConservationFailure;return status;}
 }
 result=accepted;if(evaluated)*evaluated=true;return status;
}
template<class R,int N,class TableReal,class Time> UGKWP_GAS_HD inline ChemistryStatus advanceClosedReactor
(const ClosedReactorInput<R,N>& base,Time interval,const SpeciesThermoView<TableReal,N>& t,const GasMechanismView<TableReal,N>& m,const ChemistryControls<R>& controls,ClosedReactorResult<R,N>& result,bool* evaluated=nullptr) noexcept {
 const R originalInterval=R(interval);
 if(gasFinite(originalInterval) && (interval==Time(0) || originalInterval>R(0)))
   return advanceClosedReactorImpl(base,originalInterval,t,m,controls,result,evaluated);
 return advanceClosedReactorImpl(base,interval,t,m,controls,result,evaluated);
}
// Retry-capable native owners must use this entry: a local budget/conservation
// stop is part of their original successful smaller-step method, not terminal.
template<class R,int N,class TableReal,class Time> UGKWP_GAS_HD inline ChemistryStatus advanceClosedReactorTrial
(const ClosedReactorInput<R,N>& base,Time interval,const SpeciesThermoView<TableReal,N>& t,const GasMechanismView<TableReal,N>& m,const ChemistryControls<R>& controls,ClosedReactorResult<R,N>& result,bool* evaluated=nullptr) noexcept {
 return advanceClosedReactorImpl<true>(base,R(interval),t,m,controls,result,evaluated);
}
}
#endif
