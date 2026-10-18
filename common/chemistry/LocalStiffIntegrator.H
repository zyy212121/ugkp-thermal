#ifndef UGKWP_LOCAL_STIFF_INTEGRATOR_H
#define UGKWP_LOCAL_STIFF_INTEGRATOR_H
#include "ConservativeReactor.H"
#include "NewtonFactor.H"
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
 ChemistryStatus status;const int rank=context.mechanism.independentRank;
 if(rank<0||rank>N){status.code=ChemistryCode::InvalidModel;return status;}
 status=validateGasMechanismView(context.thermo,context.mechanism);if(!status)return status;
 // The physical base and static model are evaluated through their real context.
 // All intermediate mathematical iterates use a view without global atomics;
 // CUDA thread-local storage must never be used as an atomic DeviceContext.
 auto trialThermo=context.thermo;trialThermo.diagnostics=nullptr;
 ReactorContext<R,N,TableReal> trialContext(context.base,trialThermo,context.mechanism,context.controls);
 for(int i=0;i<N*N;++i)trialContext.leftInverse[i]=context.leftInverse[i];
 R f[N]={},masses[N],residual[N]={};
 ChemistryStatus local=context.evaluate(z,f,temperature,nullptr,masses);
 if(!hasUsableChemistryResult(local))return local;
 retainDiagnostic(status,local);
 const int iterations=context.controls.maximumNewtonIterations>0?context.controls.maximumNewtonIterations:1;
 for(int iteration=0;iteration<=iterations;++iteration){
   for(int i=0;i<rank;++i)residual[i]=z[i]-rhs[i]-stageProduct(interval,coefficient,f[i]);
   const R norm=coordinateNorm(context,residual,masses);
   status.observed=double(norm);status.reference=double(context.controls.newtonTolerance);status.nonlinearIteration=iteration;
   if((norm<=context.controls.newtonTolerance&&(iteration>0||norm==R(0)))||iteration==iterations){
      ChemistryStatus final=context.evaluate(z,f,temperature,nullptr,masses);if(!hasUsableChemistryResult(final))return final;
      retainDiagnostic(status,final);
      for(int i=0;i<rank;++i)residual[i]=z[i]-rhs[i]-stageProduct(interval,coefficient,f[i]);
      const R finalNorm=coordinateNorm(context,residual,masses);
      status.observed=double(finalNorm);status.reference=double(context.controls.newtonTolerance);
      if(finalNorm>context.controls.newtonTolerance)status.code=ChemistryCode::NewtonFailure;
      return status;
   }
   R jac[N*N]={};local=trialContext.evaluateJacobian(masses,temperature,jac);
   if(!hasUsableChemistryResult(local))return local;
   R delta[N]={},rowScale[N];int pivot[N];
   for(int i=0;i<rank;++i){delta[i]=-residual[i];for(int j=0;j<rank;++j)jac[i*N+j]=(i==j?R(1):R(0))-stageProduct(interval,coefficient,jac[i*N+j]);}
   if(!factorNewton<R,N>(rank,jac,pivot,rowScale,&status.linearFailure)||!solveNewton<R,N>(rank,jac,pivot,rowScale,delta,&status.linearFailure)){status.code=ChemistryCode::SingularJacobian;return status;}
   // Natural merit measures the simplified Newton correction in the existing
   // mass-error norm. It avoids raw residual scaling by stiff reaction modes.
   const R oldNatural=coordinateNorm(context,delta,masses);R alpha=1;bool accepted=false;
   for(int search=0;search<(context.controls.maximumLineSearchIterations>0?context.controls.maximumLineSearchIterations:1);++search){
      R trial[N]={},trialMass[N],trialF[N]={},trialT=temperature,correction[N]={};
      for(int i=0;i<rank;++i)trial[i]=z[i]+alpha*delta[i];
      bool feasible=true;for(int species=0;species<N;++species){R amount=0;for(int j=0;j<rank;++j)amount+=context.mechanism.stoichiometricBasis[species*rank+j]*trial[j];R mass=context.base.speciesMass[species]+context.thermo.species[species].molarMass*context.base.volume*amount;feasible=feasible&&gasFinite(mass)&&mass>=0;}
      if(feasible){ChemistryStatus candidate=trialContext.evaluate(trial,trialF,trialT,nullptr,trialMass);
       if(hasUsableChemistryResult(candidate)){
        for(int i=0;i<rank;++i)correction[i]=trial[i]-rhs[i]-stageProduct(interval,coefficient,trialF[i]);
        const R trialNorm=coordinateNorm(context,correction,trialMass);
        if(solveNewton<R,N>(rank,jac,pivot,rowScale,correction)){
         const R natural=coordinateNorm(context,correction,masses);
         // Below the requested correction-error budget, comparing still
         // smaller corrections is dominated by stiff cancellation. Retain the
         // evaluable full Newton update; only the FINAL residual can establish
         // convergence. This does not release an unconverged state as Success.
         if((search==0&&oldNatural<=context.controls.newtonTolerance)||trialNorm<=context.controls.newtonTolerance||natural<oldNatural){
          for(int i=0;i<rank;++i){z[i]=trial[i];f[i]=trialF[i];}for(int i=0;i<N;++i)masses[i]=trialMass[i];temperature=trialT;accepted=true;break;
         }
        }
       }else if(candidate.code==ChemistryCode::InvalidModel||candidate.code==ChemistryCode::InvalidControls||candidate.thermoCode==ThermoStatus::InvalidModel){
         // Storage/schema errors are not candidate-domain failures. Re-evaluate
         // through the real view to preserve the actual property error record.
         auto reported=context.evaluate(trial,trialF,trialT,nullptr,trialMass);
         return reported;
        }
      }
      alpha*=R(.5);
   }
   if(!accepted){status.code=ChemistryCode::NewtonFailure;return status;}
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
