#ifndef UGKWP_LOCAL_STIFF_INTEGRATOR_H
#define UGKWP_LOCAL_STIFF_INTEGRATOR_H
#include "ConservativeReactor.H"
namespace ugkwp { namespace gasChemistryDetail {
template<class R,int N> UGKWP_GAS_HD inline R coordinateNorm(const ReactorContext<R,N>& context,const R* residual,const R* mass) noexcept {
 R norm=0;int rank=context.mechanism.independentRank;
 for(int s=0;s<N;++s){R value=0;for(int j=0;j<rank;++j)value+=context.mechanism.stoichiometricBasis[s*rank+j]*residual[j];value*=context.thermo.species[s].molarMass*context.base.volume;
 R denominator=context.controls.absoluteMassFractionTolerance*context.base.totalMass+context.controls.relativeTolerance*gasThermoDetail::maxValue(mass[s],context.base.speciesMass[s]);
 norm=gasThermoDetail::maxValue(norm,gasThermoDetail::absValue(value)/denominator);}
 return norm;
}
template<class R,int N> UGKWP_GAS_HD inline ChemistryStatus implicitStage
(const ReactorContext<R,N>& context,const R* rhs,R hgamma,R* z,R& temperature) noexcept {
 ChemistryStatus status;int rank=context.mechanism.independentRank;
 for(int iteration=0;iteration<context.controls.maximumNewtonIterations;++iteration){
   R f[N]={},jac[N*N]={},residual[N]={},masses[N];
   ChemistryStatus local=context.evaluate(z,f,temperature,jac);++status.nonlinearIterations;
   if(!local){local.nonlinearIterations=status.nonlinearIterations;return local;}
   if(!context.state(z,masses,temperature)){status.code=ChemistryCode::TemperatureFailure;return status;}
   for(int i=0;i<rank;++i)residual[i]=z[i]-rhs[i]-hgamma*f[i];
   R norm=coordinateNorm(context,residual,masses);
   // Do not freeze trace donors by accepting an unchanged initial guess.
   // One actual Newton update retains stiff relaxation below the error floor.
   if(norm<=context.controls.newtonTolerance && (iteration>0 || norm==R(0)))return status;
   R delta[N]={};for(int i=0;i<rank;++i){delta[i]=-residual[i];for(int j=0;j<rank;++j)jac[i*N+j]=(i==j?R(1):R(0))-hgamma*jac[i*N+j];}
   if(!solveDense<R,N>(rank,jac,delta)){status.code=ChemistryCode::SingularJacobian;return status;}
   bool accepted=false;R damping=1;
   for(int search=0;search<context.controls.maximumLineSearchIterations;++search){R trial[N]={},trialMass[N],trialT=temperature,trialF[N]={},trialResidual[N]={};
     for(int i=0;i<rank;++i)trial[i]=z[i]+damping*delta[i];
     if(context.state(trial,trialMass,trialT)){
       ChemistryStatus attempt=context.evaluate(trial,trialF,trialT,nullptr);
       if(attempt){for(int i=0;i<rank;++i)trialResidual[i]=trial[i]-rhs[i]-hgamma*trialF[i];R trialNorm=coordinateNorm(context,trialResidual,trialMass);
         if(trialNorm<norm*(R(1)-R(1e-4)*damping) || trialNorm<=context.controls.newtonTolerance){for(int i=0;i<rank;++i)z[i]=trial[i];temperature=trialT;accepted=true;break;}}
     }
     damping*=R(.5);
   }
   if(!accepted){status.code=ChemistryCode::NewtonFailure;return status;}
 }
 status.code=ChemistryCode::NewtonFailure;return status;
}
template<class R,int N> UGKWP_GAS_HD inline ChemistryStatus sdirkStep
(const ClosedReactorInput<R,N>& base,R interval,const SpeciesThermoView<R,N>& t,const GasMechanismView<R,N>& m,const ChemistryControls<R>& controls,ClosedReactorResult<R,N>& result,R initialTemperature) noexcept {
 ReactorContext<R,N> context(base,t,m,controls);ChemistryStatus st;
 if(!context.prepare()){st.code=ChemistryCode::InvalidModel;return st;}
 const R gamma=R(1)-R(1)/::sqrt(R(2));R z[N]={},rhs[N]={},f[N]={},temperature=initialTemperature;
 st=implicitStage(context,rhs,gamma*interval,z,temperature);if(!st)return st;
 ChemistryStatus eval=context.evaluate(z,f,temperature,nullptr);if(!eval){eval.nonlinearIterations+=st.nonlinearIterations;return eval;}
 for(int i=0;i<m.independentRank;++i)rhs[i]=interval*(R(1)-gamma)*f[i];
 ChemistryStatus second=implicitStage(context,rhs,gamma*interval,z,temperature);second.nonlinearIterations+=st.nonlinearIterations;if(!second)return second;
 if(!context.state(z,result.speciesMass,temperature)){second.code=ChemistryCode::TemperatureFailure;return second;}
 result.temperature=temperature;return second;
}
template<class R,int N> UGKWP_GAS_HD inline R stepError
(const ClosedReactorInput<R,N>& base,const ClosedReactorResult<R,N>& a,const ClosedReactorResult<R,N>& b,const ChemistryControls<R>& c) noexcept {
 R error=0;for(int s=0;s<N;++s){R scale=c.absoluteMassFractionTolerance*base.totalMass+c.relativeTolerance*gasThermoDetail::maxValue(a.speciesMass[s],b.speciesMass[s]);error=gasThermoDetail::maxValue(error,gasThermoDetail::absValue(a.speciesMass[s]-b.speciesMass[s])/(R(3)*scale));}
 R scale=c.absoluteTemperatureTolerance+c.relativeTolerance*gasThermoDetail::maxValue(a.temperature,b.temperature);
 return gasThermoDetail::maxValue(error,gasThermoDetail::absValue(a.temperature-b.temperature)/(R(3)*scale));
}
}}
namespace ugkwp {
template<class R,int N> UGKWP_GAS_HD inline ChemistryStatus advanceClosedReactor
(const ClosedReactorInput<R,N>& base,R interval,const SpeciesThermoView<R,N>& t,const GasMechanismView<R,N>& m,const ChemistryControls<R>& controls,ClosedReactorResult<R,N>& result,ChemistryAudit<R,N>& audit) noexcept {
 using namespace gasChemistryDetail;using gasThermoDetail::absValue;using gasThermoDetail::maxValue;using gasThermoDetail::minValue;
 ChemistryStatus status=validateGasMechanismView(t,m);if(!status)return status;
 if(!validateChemistryControls(controls)){status.code=ChemistryCode::InvalidControls;return status;}
 if(!gasFinite(interval) || interval<R(0) || !gasFinite(base.totalMass) || !(base.totalMass>R(0)) || !gasFinite(base.volume) || !(base.volume>R(0)) || !gasFinite(base.internalEnergy)){status.code=ChemistryCode::NonFiniteInput;return status;}
 R sum=0;for(int s=0;s<N;++s){if(!gasFinite(base.speciesMass[s]) || base.speciesMass[s]<R(0)){status.code=ChemistryCode::InvalidComposition;return status;}sum+=base.speciesMass[s];}
 if(absValue(sum-base.totalMass)>controls.absoluteMassTolerance+controls.relativeConservationTolerance*base.totalMass){status.code=ChemistryCode::ConservationFailure;return status;}
 R temperature=R(0);status.thermoCode=invertMixtureEnergy(base.speciesMass,base.internalEnergy,t,controls.thermo,temperature);
 if(status.thermoCode!=ThermoStatus::Success){status.code=ChemistryCode::TemperatureFailure;return status;}
 ClosedReactorInput<R,N> current=base;ClosedReactorResult<R,N> accepted;for(int s=0;s<N;++s)accepted.speciesMass[s]=base.speciesMass[s];accepted.temperature=temperature;
 ChemistryAudit<R,N> localAudit;R covered=0;
 R h=controls.initialStep>R(0)?minValue(interval,controls.initialStep):interval;if(controls.maximumStep>R(0))h=minValue(h,controls.maximumStep);
 while(covered<interval){
   if(status.attemptedSteps>=controls.maximumSteps){status.code=ChemistryCode::StepLimit;return status;}
   h=minValue(h,interval-covered);
   if(!(h>R(0)) || covered+h==covered || (h<controls.minimumStep && h<interval-covered)){status.code=ChemistryCode::StepUnderflow;return status;}
   ++status.attemptedSteps;ClosedReactorResult<R,N> full,half,end;
   ChemistryStatus trial=sdirkStep(current,h,t,m,controls,full,temperature);status.nonlinearIterations+=trial.nonlinearIterations;
   if(trial){trial=sdirkStep(current,h/R(2),t,m,controls,half,temperature);status.nonlinearIterations+=trial.nonlinearIterations;}
   if(trial){ClosedReactorInput<R,N> middle=current;for(int s=0;s<N;++s)middle.speciesMass[s]=half.speciesMass[s];trial=sdirkStep(middle,h/R(2),t,m,controls,end,half.temperature);status.nonlinearIterations+=trial.nonlinearIterations;}
   R error=trial?stepError(current,full,end,controls):R(2);
   if(!trial || !gasFinite(error) || error>R(1)){
     ++status.rejectedSteps;
     if(status.rejectedSteps>controls.maximumRejectedSteps){status.code=trial?ChemistryCode::StepLimit:trial.code;status.reaction=trial.reaction;return status;}
     R factor=trial && gasFinite(error)?maxValue(R(.1),R(.8)*R(::pow(error,R(-1)/R(3)))):R(.25);h*=minValue(R(.5),factor);
     if(h<controls.minimumStep){status.code=ChemistryCode::StepUnderflow;return status;}continue;
   }
   accepted=end;for(int s=0;s<N;++s)current.speciesMass[s]=end.speciesMass[s];temperature=end.temperature;
   covered+=h;++localAudit.acceptedSteps;localAudit.maximumLocalError=maxValue(localAudit.maximumLocalError,error);
   R factor=error>R(0)?minValue(R(3),maxValue(R(.25),R(.9)*R(::pow(error,R(-1)/R(3))))):R(3);h*=factor;if(controls.maximumStep>R(0))h=minValue(h,controls.maximumStep);
 }
 sum=0;for(int s=0;s<N;++s){sum+=accepted.speciesMass[s];localAudit.speciesMassChange[s]=accepted.speciesMass[s]-base.speciesMass[s];}
 localAudit.massResidual=sum-base.totalMass;
 if(absValue(localAudit.massResidual)>controls.absoluteMassTolerance+controls.relativeConservationTolerance*base.totalMass){status.code=ChemistryCode::ConservationFailure;return status;}
 for(int e=0;e<t.elementCount;++e){R before=0,after=0;for(int s=0;s<N;++s){R atoms=t.elementComposition[s*t.elementCount+e]/t.species[s].molarMass;before+=base.speciesMass[s]*atoms;after+=accepted.speciesMass[s]*atoms;}
   R residual=absValue(after-before);localAudit.maximumElementResidual=maxValue(localAudit.maximumElementResidual,residual);
   if(residual>controls.absoluteElementTolerance+controls.relativeConservationTolerance*absValue(before)){status.code=ChemistryCode::ConservationFailure;return status;}}
 localAudit.energyResidual=mixtureEnergy(accepted.speciesMass,accepted.temperature,t)-base.internalEnergy;
 if(!gasFinite(localAudit.energyResidual) || absValue(localAudit.energyResidual)>controls.thermo.absoluteEnergyTolerance+controls.thermo.relativeEnergyTolerance*absValue(base.internalEnergy)){status.code=ChemistryCode::ConservationFailure;return status;}
 localAudit.integratedTime=interval;localAudit.rejectedSteps=status.rejectedSteps;localAudit.nonlinearIterations=status.nonlinearIterations;
 result=accepted;audit=localAudit;return status;
}
}
#endif
