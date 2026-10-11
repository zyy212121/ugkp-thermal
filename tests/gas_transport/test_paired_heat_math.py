"""Shared finite-capacity heat acceptance, tested in physical Joules/Kelvin."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
SOURCE=r'''
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include "GpuParticleCaloric.H"
#include "GpuPairedParticleHeat.H"
void ck(bool b,const char*m){if(!b){puts(m);abort();}}
int main(){
 ugkwp::ParticleCaloric h;
 for(double dt:{0.,1e-12,1e-8,.001,1.,1e8}){
  const double C=100*h.specificHeat(3000),Cg=700000,k=3.;
  double a=-expm1(-k*dt);const double tau=ugkwp::pairedHeatBath(C*a,C*a*3000,C,Cg,3200);
  const double actual=C*a*(tau-3000);
  const double exact=Cg*C/(Cg+C)*(-expm1(-k*(1+C/Cg)*dt))*200.;
  ck(std::abs(actual-exact)<1e-8+1e-12*std::abs(exact),"homogeneous finite capacity law changed");
  if(a>0)ck(tau>=3000&&tau<=3200,"bath not convex");
 }
 for(double C:{1e-290,1.,1e290})for(double Cg:{1e-290,1.,1e290})for(double a:{1e-300,1e-12,.5,1.}){
  const double beta=ugkwp::pairedHeatBathFraction(a,C,Cg);
  ck(std::isfinite(beta)&&beta>=0&&beta<=1,"extreme capacity bath invalid");
 }
 // Stable exact branch increments, including intervals smaller than a
 // temperature ULP and phase/ordinary piecewise crossings.
 for(bool cold:{false,true}){
  h.phaseChange=cold;h.solidCp=1200;h.latentHeat=1e6;
  for(double t:{1000.,2317.,2327.,2337.,2800.,4000.,5000.}){
   for(double d:{-100.,-1e-13,1e-13,100.}){
    const double q=h.enthalpyIncrement(t,d);
    ck(q*d>0.,"stable enthalpy increment lost sign or vanished");
    if(std::abs(d)>1e-6)ck(std::abs(q-(h.enthalpy(t+d)-h.enthalpy(t)))<2e-8,"piecewise increment differs from material law");
   }
  }
 }
 ck(std::abs(h.enthalpyIncrement(h.solidus(),-1e-13)/(-1e-13)-h.solidCp)<1e-8,"tiny reverse phase boundary used wrong branch");
 h=ugkwp::ParticleCaloric{};
 // Gas convexity regression: old-Cp weights gave Tg_new=2786.395K.
 for(double ti:{1000.,2310.,2800.,4000.,5000.})for(double tg:{1200.,2327.,4000.,5500.}){
  if(ti==tg)continue;const double dh=h.enthalpyIncrement(ti,tg-ti),secant=dh/(tg-ti);
  const double beta=ugkwp::pairedHeatBathFraction(1.,secant,1.);
  const double nextGas=tg-beta*dh;
  ck(nextGas>=std::min(ti,tg)-1e-9&&nextGas<=std::max(ti,tg)+1e-9,"secant gas temperature left convex interval");
 }
 // Reviewer counterexample: Cp-old cooling would overshoot the cold bath by14K.
 const double ti=4000,tau=2801.8647,old=h.enthalpy(ti);
 const double q=ugkwp::boundedParticleSpecificHeat(h,old,ti,tau,h.specificHeat(ti),1.,1.,6000.);
 ck(h.temperature(old+q)>=tau-1e-10,"variable Cp cooled below bath");
 // Opposite particle exchanges survive a zero cell-net energy change.
 const double bath=ugkwp::pairedHeatBath(2.,2.*3000.,4.,1.,3000.);
 ck(bath==3000&&1.*(bath-2900)>0&&1.*(bath-3100)<0,"cancellation disabled physical relaxation");
 // Joint gas acceptance, including cancellation, uses bounded candidate sum.
 ck(ugkwp::pairedHeatAcceptanceScale(100.,20.)==.2,"gas floor acceptance wrong");
 ck(ugkwp::pairedHeatAcceptanceScale(-100.,20.)==1.,"gas heating was limited");
 ck(ugkwp::pairedHeatAcceptanceScale(0.,0.)==1.,"zero net heat disables exchange");
}
'''
def test_finite_capacity_bath_bounds_cancellation_and_gas_acceptance(tmp_path):
 p=tmp_path/'pair.cpp';p.write_text(SOURCE);exe=tmp_path/'pair'
 r=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),str(p),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stdout+r.stderr
 r=subprocess.run([str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stdout+r.stderr
