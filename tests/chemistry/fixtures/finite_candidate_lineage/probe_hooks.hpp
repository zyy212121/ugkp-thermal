#ifndef INDEPENDENT_LINEAGE_PROBE_H
#define INDEPENDENT_LINEAGE_PROBE_H
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <type_traits>
#include <vector>
namespace independent {
struct Counters { int prepare=0,basis=0,inversion=0,rates=0,derivatives=0; };
inline Counters counters;
struct Step {int id=0,code=-1,firstStage=0,lastStage=0;bool success=false,injected=false;long double h=0,Tin=0,Tout=0;std::vector<long double> before,after;};
struct Stage {int step=0,code=-1,firstState=0,lastState=0;bool success=false;long double hg=0,Tin=0,Tout=0;std::vector<long double> rhs,before,after;};
struct State {int step=0,stage=-1;bool success=false;long double T=0;std::vector<long double> z,mass;};
struct Rate {int step=0,stage=-1;bool success=false;long double T=0;std::vector<long double> c,production;};
struct Consumption {long double h=0,T=0;std::vector<long double> state;std::size_t endStep=0;};
inline std::vector<Consumption> consumed;
inline std::vector<Rate> rates;
inline std::vector<Step> steps;inline std::vector<Stage> stages;inline std::vector<State> states;
inline std::set<int> failedSteps;inline int activeStep=0,activeStage=-1;
inline void reset(){counters={};consumed.clear();rates.clear();steps.clear();stages.clear();states.clear();failedSteps.clear();activeStep=0;activeStage=-1;}
template<class A> std::vector<long double> values(const A& a){std::vector<long double> out;for(const auto& v:a)out.push_back(v);return out;}
template<class R> std::vector<long double> values(const R* a,int n){return {a,a+n};}
template<class R>void recordConsumption(long double h,const R& out){Consumption c;c.h=h;c.T=out.temperature;c.state=values(out.speciesMass);c.endStep=steps.size();consumed.push_back(c);}
template<class B,class T,class O> struct StepScope {
 const B& base;const T& thermo;O& output;int index,oldActive,oldCount;
 StepScope(const B& b,long double h,const T& t,O& out,long double initial):base(b),thermo(t),output(out),index(int(steps.size())),oldActive(activeStep),oldCount(t.coefficientCount){
  Step step;step.id=index+1;step.h=h;step.Tin=initial;step.before=values(b.speciesMass);step.firstStage=int(stages.size());step.injected=failedSteps.count(step.id);steps.push_back(step);activeStep=step.id;
  if(step.injected)const_cast<typename std::remove_const<typename std::remove_reference<T>::type>::type&>(thermo).coefficientCount=0;
  for(auto& mass:output.speciesMass)mass=typename std::remove_reference<decltype(mass)>::type(-777);
  output.temperature=-777;
 }
 template<class S>S finish(S s){auto& p=steps[index];p.code=int(s.code);p.success=bool(s);p.after=values(output.speciesMass);p.Tout=output.temperature;p.lastStage=int(stages.size());return s;}
 ~StepScope(){const_cast<typename std::remove_const<typename std::remove_reference<T>::type>::type&>(thermo).coefficientCount=oldCount;activeStep=oldActive;}
};
template<class C,class R> struct StageScope {
 R* z;R& T;int n,index,oldActive;
 StageScope(const C& c,const R* rhs,long double hg,R* zz,R& t):z(zz),T(t),n(c.mechanism.independentRank),index(int(stages.size())),oldActive(activeStage){Stage s;s.step=activeStep;s.hg=hg;s.Tin=t;s.rhs=values(rhs,n);s.before=values(z,n);s.firstState=int(states.size());stages.push_back(s);activeStage=index;}
 template<class S>S finish(S s){auto& p=stages[index];p.code=int(s.code);p.success=bool(s);p.after=values(z,n);p.Tout=T;p.lastState=int(states.size());return s;}
 ~StageScope(){activeStage=oldActive;}
};
template<class R>struct StateScope{
 const R* z;R* masses;R& T;int n;
 StateScope(const R* zz,R* m,R& t,int rank):z(zz),masses(m),T(t),n(rank){}
 bool finish(bool success){State s;s.step=activeStep;s.stage=activeStage;s.success=success;s.z=values(z,n);s.mass=values(masses,2);s.T=T;states.push_back(s);return success;}
};
template<class R>struct RateScope{
 R T;const R* concentrations;R* production;int n;
 RateScope(R t,const R* c,R* p,int size):T(t),concentrations(c),production(p),n(size){++counters.rates;}
 bool finish(bool success){Rate r;r.step=activeStep;r.stage=activeStage;r.success=success;r.T=T;r.c=values(concentrations,n);if(success)r.production=values(production,n);rates.push_back(r);return success;}
};
inline void dump(){
 std::cout<<"counts prepare="<<counters.prepare<<" basis="<<counters.basis<<" state="<<states.size()<<" inversion="<<counters.inversion<<" rates="<<counters.rates<<" derivatives="<<counters.derivatives<<"\n";
 for(const auto& c:consumed)std::cout<<" consumed="<<c.h<<" through_step="<<c.endStep<<" selected="<<c.state[0]<<","<<c.state[1]<<" T="<<c.T<<"\n";
 for(const auto& s:steps)std::cout<<" step="<<s.id<<" injected="<<s.injected<<" ok="<<s.success<<" code="<<s.code<<" h="<<s.h<<" before="<<s.before[0]<<","<<s.before[1]<<" after="<<s.after[0]<<","<<s.after[1]<<" Tin="<<s.Tin<<" Tout="<<s.Tout<<" stages="<<s.lastStage-s.firstStage<<"\n";
 for(const auto& s:stages)std::cout<<" stage step="<<s.step<<" ok="<<s.success<<" hg="<<s.hg<<" rhs="<<s.rhs[0]<<" zin="<<s.before[0]<<" zout="<<s.after[0]<<" Tin="<<s.Tin<<" Tout="<<s.Tout<<" states="<<s.lastState-s.firstState<<"\n";
}
}
#endif
