#include "materials/MaterialTransport.H"
#include "materials/ReactionStepControl.H"
#include "coupling/Coordinator.H"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace chmt;
namespace {
int failures=0;
void check(bool ok,const char* name){std::cout<<(ok?"PASS ":"FAIL ")<<name<<'\n';if(!ok)++failures;}
PhysicsConfig config(){PhysicsConfig p;p.enableReactions=true;p.nReactions=2;p.nElements=1;
 for(int c=0;c<Nc;++c){auto& t=p.condensed[c];t.rho=1000;t.cp0=1000;t.Tmin=100;t.Tmax=3000;t.element[0]=1;}
 for(int s=0;s<Ns;++s){auto& t=p.species[s];t.R=287;t.cp0=1287;t.Tmin=100;t.Tmax=3000;t.element[0]=1;}
 p.reactions[0].A=2;p.reactions[0].order[0]=1;p.reactions[0].condensedNu[0]=-1;p.reactions[0].condensedNu[1]=1;
 p.reactions[1].A=1000;p.reactions[1].order[1]=1;p.reactions[1].condensedNu[1]=-1;p.reactions[1].gasNu[0]=1;return p;}
SolidQ initial(const PhysicsConfig& p,double a=.8,double b=0,double g=.1,double T=700){SolidQ q;q.condensed[0]=a;q.condensed[1]=b;q.pore[0]=g;q.porosity=.1;
 for(int c=0;c<Nc;++c){const auto& t=p.condensed[c];q.energy+=q.condensed[c]*(t.e0+t.cp0*T+t.cp1*T*T/2);}
 const auto& t=p.species[0];q.energy+=g*(t.e0+(t.cp0-t.R)*T+t.cp1*T*T/2);return q;}
struct Result{SolidQ q;std::uint64_t steps=0;double T=0;};
Result solve(const PhysicsConfig& p,const SolidQ& q,double h,double cfl){Result r;std::string error;
 for(int j=0;j<p.nReactions;++j)if(!validateReaction(p.reactions[j],p,j,error))throw std::runtime_error(error);
 if(!advanceMaterialReactions(q,.001,p,h,0,cfl,1000000,r.q,r.steps,error))throw std::runtime_error(error);
 MaterialPrimitive w;if(!recoverMaterial(r.q,.001,p,w))throw std::runtime_error("final EOS invalid");r.T=w.temperature;
 double m=0,m0=0;for(int c=0;c<Nc;++c){m+=r.q.condensed[c];m0+=q.condensed[c];if(r.q.condensed[c]<0)throw std::runtime_error("negative condensed inventory");}
 for(int s=0;s<Ns;++s){m+=r.q.pore[s];m0+=q.pore[s];if(r.q.pore[s]<0)throw std::runtime_error("negative pore inventory");}
 if(std::abs(m-m0)>1e-11||q.energy!=r.q.energy)throw std::runtime_error("reaction conservation failed");
 return r;}
// Independent RK4 reference for a fractional chain, not production stepping/rates.
double fractionalChainReference(double t){double b=0;const int n=131072;const double h=t/n;
 auto rhs=[](double time,double B){return 1.6*std::exp(-2*time)-10*std::sqrt(std::max(0.,B));};
 for(int i=0;i<n;++i){const double time=i*h,k1=rhs(time,b),k2=rhs(time+h/2,b+h*k1/2),k3=rhs(time+h/2,b+h*k2/2),k4=rhs(time+h,b+h*k3);b+=h*(k1+2*k2+2*k3+k4)/6;}return b;}
// Host composition of the actual legacy participant assembly and shared device
// timestep math. This is not a CUDA kernel/runtime test.
SolidQ legacyMidpointChain(const PhysicsConfig& p,double interval,double cfl,double seed=0){SolidQ q=initial(p,.8,seed,.1);double elapsed=0;
 while(elapsed<interval){SolidQ unused,midpoint;double dt=0;bool active=false;
  if(!reactionStepCandidate(q,.001,p,interval-elapsed,cfl,unused,dt,active))throw std::runtime_error("legacy shared bound failed");
  if(!active)break;
  Real rates[Nr]{};if(!advanceLocalReactions(q,.001,p,dt/2,midpoint,rates))throw std::runtime_error("legacy midpoint failed");
  double midpointLimit=0;if(!reactionStepCandidate(midpoint,.001,p,dt,cfl,unused,midpointLimit,active)||midpointLimit<dt*(1-1e-12))throw std::runtime_error("legacy midpoint bound failed");
  if(!boundedReactionRates(midpoint,q,.001,p,dt,rates))throw std::runtime_error("legacy midpoint rates failed");
  SolidQ rhs;for(int r=0;r<p.nReactions;++r){for(int c=0;c<Nc;++c)rhs.condensed[c]+=rates[r]*p.reactions[r].condensedNu[c];
   for(int s=0;s<Ns;++s)rhs.pore[s]+=rates[r]*p.reactions[r].gasNu[s];
   rhs.progress[r]=rates[r];}
  q=assembleSolidCandidate(q,rhs,dt,nullptr,0,0);MaterialPrimitive w;if(!recoverMaterial(q,.001,p,w))throw std::runtime_error("legacy endpoint failed");q.porosity=w.porosity;elapsed+=dt;
  if(interval-elapsed<=8*std::numeric_limits<Real>::epsilon()*interval)elapsed=interval;
 }
 return q;}
}
int main(){try{std::cout<<std::setprecision(17);auto p=config();const double h=.001,exactB=.8*2/998*(std::exp(-2*h)-std::exp(-1000*h));
 ReactionStepControl control;check(reactionStepControl(initial(p),.001,p,.1,control),"shared host/device control evaluates valid zero-start mechanism");
 check(std::abs(control.sensitivity-2000)<1e-10&&std::abs(control.dtLimit-5e-5)<1e-16,"shared extent Jacobian includes generated-intermediate coupling in inverse seconds");
 SolidQ candidate;double dt=0;bool active=false;check(reactionStepCandidate(initial(p),.001,p,.001,.1,candidate,dt,active)&&active&&dt<=5e-5,"legacy timestep helper rejects original one-millisecond chemical step");
 check(candidate.energy==initial(p).energy,"shared device candidate keeps formation-inclusive energy exactly");
 SolidQ failedCandidate=initial(p);double failedStep=123;bool failedActive=true;
 check(!reactionStepCandidate(initial(p),.001,p,0,.1,failedCandidate,failedStep,failedActive)
     &&failedCandidate.energy==initial(p).energy&&failedCandidate.condensed[0]==.8&&failedStep==123&&failedActive,"shared candidate failure leaves all output arguments unchanged");
 double legacyPrevious=1;for(double cfl:{.1,.05,.025}){const auto q=legacyMidpointChain(p,h,cfl),tiny=legacyMidpointChain(p,h,cfl,1e-15);const double error=std::abs(q.condensed[1]-exactB);std::cout<<"legacy_midpoint "<<cfl<<' '<<error<<'\n';check(error<legacyPrevious*.4,"legacy midpoint participant math retains second-order CFL refinement");
 check(std::abs(q.condensed[1]-tiny.condensed[1])<2e-12,"legacy midpoint zero/tiny intermediate continuity");
 check(q.energy==initial(p).energy&&std::abs(q.condensed[0]+q.condensed[1]+q.pore[0]-.9)<1e-12,"legacy midpoint assembly conserves mass and formation energy");legacyPrevious=error;}
 double previous=1;
 for(double cfl:{.1,.05,.025}){auto zero=solve(p,initial(p),h,cfl),tiny=solve(p,initial(p,.8,1e-15),h,cfl);const double error=std::abs(zero.q.condensed[1]-exactB);
 std::cout<<"chain "<<cfl<<' '<<zero.steps<<' '<<error/exactB<<'\n';
 check(zero.steps>=10,"zero intermediate activates automatic stiffness control");check(error/exactB<cfl,"startup chain relative error tracks CFL");
 check(error<previous*.65,"startup chain error reduces with CFL");check(std::abs(zero.q.condensed[1]-tiny.q.condensed[1])<2e-12,"zero/tiny intermediate control continuity");previous=error;}
 p=config();p.reactions[0].A=3;p.reactions[1]={};p.reactions[1].A=7;p.reactions[1].order[0]=1;p.reactions[1].condensedNu[0]=-1;p.reactions[1].gasNu[0]=1;
 Real limitedRates[Nr]{};check(boundedReactionRates(initial(p),initial(p,.1,.1,.1),.001,p,1,limitedRates),"aggregate net-loss limiter accepts shared donor request");
 check(std::abs(limitedRates[0]-.03)<1e-14&&std::abs(limitedRates[1]-.07)<1e-14,"aggregate net-loss limiter preserves branching while bounding total shared loss");
 previous=1;for(double cfl:{.1,.05,.025}){auto r=solve(p,initial(p,.7,.1,.1),.1,cfl);double a=.7*std::exp(-1),error=std::abs(r.q.condensed[0]-a);check(error<previous*.65,"shared-donor concurrent reactions converge");check(std::abs(r.q.condensed[1]-(.1+.3*(.7-r.q.condensed[0])))<1e-13,"shared-donor branch ratio preserved");previous=error;}
 p=config();p.reactions[1].A=10/std::sqrt(.001);p.reactions[1].order[1]=.5;const double fractionalRef=fractionalChainReference(.01);
 previous=1;for(double cfl:{.1,.05,.025}){auto r=solve(p,initial(p),.01,cfl);double error=std::abs(r.q.condensed[1]-fractionalRef);std::cout<<"fractional "<<cfl<<' '<<r.steps<<' '<<error<<'\n';check(error<previous*.75,"zero-start fractional intermediate converges");previous=error;}
 p=config();p.nReactions=1;p.reactions[0]={};auto& rx=p.reactions[0];rx.A=.01;rx.order[0]=.5;rx.order[1]=1.5;rx.condensedNu[0]=-.4;rx.condensedNu[1]=-.6;rx.gasNu[0]=1;
 const double k=.01*std::sqrt(.4)*std::pow(.6,1.5)/.001,exactM=.8/(1+k*.8*.2);previous=1;
 for(double cfl:{.1,.05,.025}){auto r=solve(p,initial(p,.32,.48,.1),.2,cfl);double error=std::abs(r.q.condensed[0]+r.q.condensed[1]-exactM);check(error<previous*.7,"multiple fractional reactant orders converge to analytic law");previous=error;}
 auto absent=solve(p,initial(p,0,.8,.1),.2,.1);check(absent.q.condensed[1]==.8,"inactive singular-order mechanism remains unchanged");
 p=config();p.reactions[0].A=300;p.reactions[1]={};p.reactions[1].A=700;p.reactions[1].order[1]=1;p.reactions[1].condensedNu[1]=-1;p.reactions[1].condensedNu[0]=1;
 auto eq=solve(p,initial(p,.8,.1,.02),.1,.1);check(std::abs(eq.q.condensed[0]-.63)<1e-12,"stiff reversible mechanism reaches correct equilibrium");
 p=config();p.nReactions=1;p.reactions[0].A=1/std::sqrt(.001);p.reactions[0].order[0]=.5;
 auto depleted=solve(p,initial(p),2,.1);
 check(depleted.q.condensed[0]==0&&depleted.steps<10000,"fractional finite-time depletion terminates conservatively without a minimum-step stall");
 p=config();SolidQ rollback=initial(p,.5,.2,.1),unchanged=rollback;std::uint64_t count=777;std::string error;
 const bool limited=advanceMaterialReactions(initial(p),.001,p,.001,0,.1,1,rollback,count,error);
 bool unchangedOutput=rollback.energy==unchanged.energy&&rollback.porosity==unchanged.porosity;
 for(int c=0;c<Nc;++c)unchangedOutput=unchangedOutput&&rollback.condensed[c]==unchanged.condensed[c];
 for(int s=0;s<Ns;++s)unchangedOutput=unchangedOutput&&rollback.pore[s]==unchanged.pore[s];
 for(int r=0;r<Nr;++r)unchangedOutput=unchangedOutput&&rollback.progress[r]==unchanged.progress[r];
 check(!limited&&unchangedOutput&&count==777&&error=="reaction local substep limit reached","maxSubsteps failure is transactional for output and accepted count");
 p=config();p.nReactions=1;p.reactions[0].A=1000;SolidQ inactive;count=999;
 const bool quiet=advanceMaterialReactions(initial(p,0,.8,.1),.001,p,1000,0,.1,3,inactive,count,error);
 check(quiet&&count<=1&&inactive.condensed[0]==0&&inactive.condensed[1]==.8,"entirely inactive stiff mechanism avoids pointless substeps");
 p=config();p.nReactions=1;p.condensed[0].e0=300000;p.condensed[0].cp0=700;p.condensed[0].cp1=.4;
 p.condensed[1].e0=-400000;p.condensed[1].cp0=1200;p.condensed[1].cp1=.15;p.species[0].e0=-200000;p.species[0].cp0=1000;p.species[0].cp1=.25;
 p.reactions[0].condensedNu[1]=.65;p.reactions[0].gasNu[0]=.35;p.reactions[0].temperaturePower=.5;p.reactions[0].activationEnergy=55000;p.reactions[0].A=.6*std::exp(55000/(universalGasConstant*650))/std::sqrt(650.);
 // Independent 60-digit chain-rule reference: C=1012.3 J/K,
 // dT/dextent=455.60728045045935 K/mol at 650 K.
 check(reactionStepControl(initial(p,.72,.18,.1,650),.001,p,.1,control)
     &&std::abs(control.sensitivity-2.6330012920660717)<1e-11,"shared thermal-feedback Jacobian matches independent caloric chain rule");
 previous=1e6;for(double cfl:{.1,.05,.025}){auto r=solve(p,initial(p,.72,.18,.1,650),.5,cfl);double error=std::abs(r.T-862.5149208212807);std::cout<<"adiabatic "<<cfl<<' '<<r.steps<<' '<<error<<'\n';check(error<previous*.7,"Arrhenius variable-cp thermal feedback converges");previous=error;}
 std::cout<<"failures="<<failures<<'\n';return failures?1:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
