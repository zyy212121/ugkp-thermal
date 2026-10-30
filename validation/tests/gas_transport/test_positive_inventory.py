"""Depletion limiter must remain conservative at subnormal inventory scales."""
from pathlib import Path
import subprocess
import pytest
from test_mixture_state import compile_probe, ROOT
from test_mixture_transport import fixture

@pytest.mark.parametrize("local_time",[False,True])
def test_real_flux_chain_blocks_zero_donor_subnormal_outflow(tmp_path,local_time):
    layout=fixture().replace("gasSpecies;};", "gasSpecies;double* gasPseudoDt=nullptr;double gasPseudoReferenceDt=1;};")
    compile_probe(tmp_path,layout+r"""
int main(){
 State s;initialise(s);const Real dt=9.6873727665352311e-9;
 double pseudo[2]={dt,dt};s.gasPseudoDt=LOCAL?pseudo:nullptr;
 const double step=LOCAL?1.:dt;
 s.V[0]=s.V[1]=Real(3.6691894773708457e-7);
 s.gasSpecies.rho[0]=1;s.gasSpecies.rho[1]=1;
 s.gasSpecies.rho[2]=0;s.gasSpecies.rho[3]=0;
 // The neighbour is a zero CO2 donor. The request disappears as an integrated
 // mass, but remains representable in the production density update.
 const Real tiny=Real(1.1644331826788377e-317);
 s.gasSpecies.flux[0]=tiny;s.gasSpecies.flux[3]=-tiny;s.gasPhiRho[0]=0;
 s.gasPhiRhoE[0]=123;
 ck(dt*tiny==0 && (dt/s.V[1])*tiny>0,"counterexample not represented");
 for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,step);}
 ck(s.gasSpecies.positivityScale[3]==0,"zero donor considered sufficient after underflow");
 threadIdx.x=0;applyGasFluxPositivityScaleKernel(&s);
 ck(s.gasSpecies.flux[0]==0&&s.gasSpecies.flux[3]==0&&s.gasPhiRhoE[0]==0,"one shared face factor not applied");
 for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,step);}
 ck(s.gasSpecies.rho[2]==0&&s.gasSpecies.rho[3]==0,"negative species produced");
 ck(s.gasSpecies.rho[0]+s.gasSpecies.rho[1]==2,"carrier inventory changed");
}
""".replace("LOCAL","true" if local_time else "false"))

def test_scaled_ratio_extremes_and_ordinary_formula(tmp_path):
    p=tmp_path/'ratio.cpp';p.write_text(r"""
#include "gasTransport/PositiveInventory.H"
#include <cassert>
#include <limits>
#include <algorithm>
int main(){
 using ugkwp::inventoryOutflowScale;
 const double tiny=std::numeric_limits<double>::denorm_min();
 assert(inventoryOutflowScale(0.,1e-7,1e-8,1e-317)==0);
 assert(inventoryOutflowScale(1.,1.,0.,1.)==1);
 assert(inventoryOutflowScale(1.,1.,1.,0.)==1);
 // Both products underflow, yet the mathematical ratio is one half.
 const double a=inventoryOutflowScale(1e-319,1e-7,1e-8,2e-318);
 assert(a>0 && a<.51);
 // Neither an overflowing product nor a lost density*volume may discard
 // an otherwise representable ratio. Long double is the independent oracle.
 const int e[]={-1074,-1060,-1022,-900,-500,-100,0,100,500,900,1023};
 for(int ed:e)for(int ev:e)for(int et:e)for(int ef:e){
  const double d=std::ldexp(1.,ed),v=std::ldexp(1.,ev),t=std::ldexp(1.,et),f=std::ldexp(1.,ef);
  const long double ratio=(static_cast<long double>(d)*v)/(static_cast<long double>(t)*f);
  const double q=inventoryOutflowScale(d,v,t,f);
  assert(std::isfinite(q)&&q>=0&&q<=1);
  if(ratio>=1)assert(q==1);
  else {
   assert(static_cast<long double>(q)<=ratio);
   const long double expected=.999L*ratio;
   assert(std::abs(static_cast<long double>(q)-expected)<=8*std::numeric_limits<double>::epsilon()*expected+2*static_cast<long double>(tiny));
  }
 }
 for(double d:{.001,.1,1.,100.})for(double v:{1e-7,1.,100.})for(double t:{1e-8,.1,10.})for(double f:{.01,1.,1e5}){
  const double old=t*f>d*v?.999*d*v/(t*f):1;
  const double now=inventoryOutflowScale(d,v,t,f);
  assert(std::abs(now-old)<=8*std::numeric_limits<double>::epsilon()*std::max(now,old));
 }
 assert(inventoryOutflowScale(0.f,1e-7f,1e-8f,std::numeric_limits<float>::denorm_min())==0.f);
}
""")
    exe=tmp_path/'ratio'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),str(p),'-o',str(exe)],check=True)
    subprocess.run([exe],check=True)


def test_nonzero_subnormal_inventory_is_transferred_conservatively(tmp_path):
    compile_probe(tmp_path,fixture()+r"""
int main(){
 State s;initialise(s);const Real dt=1e-8;
 s.V[0]=s.V[1]=1e-7;
 s.gasSpecies.rho[0]=s.gasSpecies.rho[1]=1;
 s.gasSpecies.rho[2]=0;s.gasSpecies.rho[3]=1e-319;
 s.gasSpecies.flux[0]=2e-318;s.gasSpecies.flux[3]=-2e-318;
 s.gasPhiRho[0]=0;s.gasPhiRhoE[0]=123;
 const Real inventory=s.gasSpecies.rho[3];
 ck(inventory*s.V[1]==0&&dt*(-s.gasSpecies.flux[3])==0,"both integrated products must underflow");
 for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,dt);}
 const Real factor=s.gasSpecies.positivityScale[3];
 ck(factor>.49&&factor<.5,"representable inventory ratio discarded");
 threadIdx.x=0;applyGasFluxPositivityScaleKernel(&s);
 ck(s.gasSpecies.flux[0]>0&&s.gasSpecies.flux[0]==-s.gasSpecies.flux[3],"trace transfer or mass closure lost");
 ck(s.gasPhiRhoE[0]==123*factor,"energy not scaled with trace transfer");
 for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,dt);}
 ck(s.gasSpecies.rho[2]>0&&s.gasSpecies.rho[3]>0,"transfer missing or donor overdrawn");
 ck(std::abs((s.gasSpecies.rho[2]+s.gasSpecies.rho[3])-inventory)<=2*std::numeric_limits<Real>::denorm_min(),"trace inventory not conserved");
}
""")
