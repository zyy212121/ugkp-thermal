"""Executable C++14 rate, mechanism-validation and derivative behavior tests."""
from pathlib import Path
import subprocess
import pytest
ROOT = Path(__file__).resolve().parents[2]
SOURCE = r'''
#include "common/chemistry/GasRates.H"
#include "common/chemistry/GasJacobian.H"
#include "common/chemistry/mechanisms/GeneratedH2O2.H"
#include <cassert>
#include <cmath>
#include <limits>
using namespace ugkwp;
struct Model {
 SpeciesThermoData<double> sp[2]; double coef[6]={1000,0,0,1000,0,0};
 double elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<double> rx[2]; GasStoichTerm<double> re[2],pr[2];
 GasColliderEfficiency<double> eff[1];
 SpeciesThermoView<double,2> t; GasMechanismView<double,2> m;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 for(int r=0;r<2;++r){rx[r].reactantOffset=r;rx[r].reactantCount=1;rx[r].productOffset=r;rx[r].productCount=1;rx[r].sourceIndex=r;rx[r].highRate.preExponential=2;re[r].species=0;re[r].coefficient=1;pr[r].species=1;pr[r].coefficient=1;}
 m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=2;m.products=pr;m.productTermCount=2;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 }
};
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<1e-12+r*std::abs(b);}
int main(){
 {GeneratedH2O2<double> h;auto t=h.thermoView();auto m=h.mechanismView();
 for(int active=0;active<10;++active){double c[10]={},jac[100],dt[10];c[active]=1;ChemistryStatus st;assert(evaluateGasRateDerivatives(1000.,c,t,m,jac,dt,st));for(double v:jac)assert(std::isfinite(v));
 double w0[10],wp[10],wpp[10];assert(evaluateGasRates(1000.,c,t,m,w0,nullptr,st));
 for(int j=0;j<10;++j){double old=c[j],epsilon=1e-5;c[j]=old+epsilon;assert(evaluateGasRates(1000.,c,t,m,wp,nullptr,st));c[j]=old+2*epsilon;assert(evaluateGasRates(1000.,c,t,m,wpp,nullptr,st));c[j]=old;
 for(int i=0;i<10;++i){double numerical=(-3*w0[i]+4*wp[i]-wpp[i])/(2*epsilon);assert(std::abs(jac[i*10+j]-numerical)<1e-3+1e-6*std::abs(numerical));}}
 }}
 Model a; assert(validateGasMechanismView(a.t,a.m));
 double c[2]={3,0},w[2]={99,98},q[2]={97,96}; ChemistryStatus st;
 assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(near(w[0],-6));assert(near(w[1],6));assert(near(q[0],6));
 a.rx[0].reversible=true;c[1]=1;assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(near(w[1],4));
 c[1]=3;assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(w[1]==0);
 a.rx[0].reversible=false;c[1]=0;a.rx[0].highRate.activationEnergy=-1000; assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(near(w[1],6*std::exp(1000/(universalGasConstant<double>()*700))));
 a.rx[0].highRate.activationEnergy=0;a.rx[0].type=GasReactionType::ThirdBody;a.rx[0].defaultEfficiency=1;assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(near(w[1],18));
 a.eff[0].species=0;a.eff[0].efficiency=2;a.m.efficiencies=a.eff;a.m.efficiencyCount=1;a.rx[0].efficiencyCount=1;assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(near(w[1],36));
 a.rx[0].type=GasReactionType::Lindemann;a.rx[0].lowRate.preExponential=4;assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(near(w[1],6*12./13.));
 a.rx[0].type=GasReactionType::Troe;a.rx[0].troe.alpha=.5;a.rx[0].troe.T1=1000;a.rx[0].troe.T3=100;assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(w[1]>0 && w[1]<6*12./13.);
 c[0]=c[1]=0;assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(w[0]==0 && w[1]==0);
 // Derivatives must remain finite at exact zero radicals and include collider terms.
 c[0]=3;c[1]=1;double jac[4],dt[2];assert(evaluateGasRateDerivatives(700.,c,a.t,a.m,jac,dt,st));
 for(int j=0;j<2;++j){double cp[2]={c[0],c[1]},cm[2]={c[0],c[1]},wp[2],wm[2];cp[j]+=1e-5;cm[j]-=1e-5;assert(evaluateGasRates(700.,cp,a.t,a.m,wp,nullptr,st));assert(evaluateGasRates(700.,cm,a.t,a.m,wm,nullptr,st));for(int s=0;s<2;++s)assert(near(jac[s*2+j],(wp[s]-wm[s])/2e-5,1e-7));}
 double wp[2],wm[2];assert(evaluateGasRates(700.001,c,a.t,a.m,wp,nullptr,st));assert(evaluateGasRates(699.999,c,a.t,a.m,wm,nullptr,st));assert(near(dt[1],(wp[1]-wm[1])/.002,1e-7));
 a=Model();a.t.species=a.sp;a.t.coefficients=a.coef;a.t.elementComposition=a.elements;a.m.reactions=a.rx;a.m.reactants=a.re;a.m.products=a.pr;a.m.stoichiometricBasis=a.basis;
 a.m.reactionCount=2;a.rx[0].duplicate=a.rx[1].duplicate=true;c[0]=3;c[1]=0;assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(near(w[1],12));
 // Invalid metadata and rates are diagnosed without partially publishing outputs.
 w[0]=99;w[1]=98;q[0]=97;c[0]=-1;assert(!evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(w[0]==99 && w[1]==98 && q[0]==97);c[0]=3;
 a.pr[0].coefficient=2;assert(!validateGasMechanismView(a.t,a.m));a.pr[0].coefficient=1;
 a.basis[1]=2;assert(!validateGasMechanismView(a.t,a.m));a.basis[1]=1;
 a.m.speciesOrderHash=2;assert(!validateGasMechanismView(a.t,a.m));a.m.speciesOrderHash=1;
 a.rx[0].duplicate=false;assert(!validateGasMechanismView(a.t,a.m));a.rx[0].duplicate=true;
 a.m.reactionCount=1;assert(!validateGasMechanismView(a.t,a.m));a.rx[0].duplicate=false;
 a.pr[0].species=0;assert(!validateGasMechanismView(a.t,a.m));a.pr[0].species=1;
 a.rx[0].reversible=true;a.rx[0].highRate.preExponential=0;c[0]=3;c[1]=1;
 assert(evaluateGasRateDerivatives(700.,c,a.t,a.m,jac,dt,st));for(double v:jac)assert(v==0);for(double v:dt)assert(v==0);
 a.rx[0].highRate.preExponential=std::exp(-300.);a.coef[5]=800*universalGasConstant<double>()*700/.028;c[0]=1;c[1]=0;
 assert(evaluateGasRates(700.,c,a.t,a.m,w,q,st));assert(near(w[1],std::exp(-300.)));
 assert(evaluateGasRateDerivatives(700.,c,a.t,a.m,jac,dt,st));assert(near(jac[3],-std::exp(500.),1e-10));
 a.rx[0].highRate.preExponential=2;a.sp[0].hasEntropyReference=false;assert(!validateGasMechanismView(a.t,a.m));
}
'''
def test_rates_validation_and_derivatives(tmp_path):
    src=tmp_path/'rates.cpp';src.write_text(SOURCE)
    exe=tmp_path/'rates'
    subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-pedantic','-I',str(ROOT),str(src),'-o',str(exe)],check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)
