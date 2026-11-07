"""Execute the production Euler/RK host schedule with observable state epochs.
CUDA launches are replaced by synchronous fixture functions; numerical native
application regressions separately cover the real gas kernels.
"""
from validation.tools.paths import SOURCE_ROOT
import re
import subprocess


def test_recovery_epochs_and_direct_substage(tmp_path):
    text=(SOURCE_ROOT/'common/GpuGasAdvance.cuh').read_text()
    a=text.index('template<class GasHostState>')
    b=text.index('template<class GasHostState>\nint finaliseGasBoundaryStage',a)
    schedule=re.sub(r'<<<.*?>>>','',text[a:b],flags=re.S)
    code=r"""
#include <cassert>
#include <cmath>
#include <cstdio>
using cudaError_t=int;constexpr int cudaSuccess=0;
int cudaGetLastError(){return 0;}
void setLastError(const char*,int){assert(false);}void setLastErrorText(const char*){assert(false);}
struct State{
 int fixedCellBlockThreads=128,fixedFaceBlockThreads=128,nCells=1,nFaces=2,nInternalFaces=1;
 int hostTurbulenceModel=3,hostGasFluxScheme=7,hostGasTimeIntegrator=1,hasPeriodicFaces=1;
 void* gasCaptureStream=nullptr;State* deviceState=this;
 double conserved=2,primitive=-999,initial=0,ledger=0;
 int recoveries=0,sstRecoveries=0;bool expectedRefinement=false;
};
namespace ugkwp{
 double gasStageTime(State&,double t){return t;}
 bool gasBoundaryLayerPreparationReusable(State&,double,double){return false;}
 bool gasBoundaryLayerEnabled(State&){return false;}
 bool mixtureGasActive(State&){return false;}
 template<class Policy>int prepareGasBoundaryLayerStage(State*,double,double,bool){return 0;}
}
struct GasHostPolicy{
 using Time=double;using Weight=double;
 static int applyFaceSources(State*s,double,double){assert(s->primitive==s->conserved);return 0;}
 static int applySstFaceSources(State*,double){return 0;}
 static int accumulateWallEnergy(State*s,double dt){s->ledger+=dt;return 0;}
 static double firstLedgerDt(State*s,double dt){return s->hostGasTimeIntegrator==2?dt*.5:dt/6.;}
 static double finalLedgerDt(double dt){return dt*2./3.;}
};
void recoverGasPrimitivesKernel(State*s,bool refined){assert(refined==s->expectedRefinement);s->primitive=s->conserved;++s->recoveries;}
void recoverSstPrimitivesKernel(State*s,bool refined){assert(refined==s->expectedRefinement);assert(s->primitive==s->conserved);++s->sstRecoveries;}
void updateRiemannBoundaryMirrorKernel(State*s,double){assert(s->primitive==s->conserved);}
void updateLegacyGasBoundaryMirrorKernel(State*s,double){assert(s->primitive==s->conserved);}
void computeGasPrimitiveGradientsKernel(State*s,bool){assert(s->primitive==s->conserved);}
void applySstWallFunctionStateKernel(State*,bool){}
void computeGasHllcAdcSensorKernel(State*s){assert(s->primitive==s->conserved);}
void computeGasGradientLimiterKernel(State*,bool){}
void computeGasEddyViscosityKernel(State*s){assert(s->primitive==s->conserved);}
template<bool>void computeGasInternalFaceFluxKernel(State*s,double,bool){assert(s->primitive==s->conserved);}
void enforcePeriodicGasFluxAntisymmetryKernel(State*,bool){}
void computeGasFluxPositivityScaleKernel(State*,double){}
void applyGasFluxPositivityScaleKernel(State*,bool){}
void computeSstGradientsKernel(State*){}
void computeSstFaceFluxKernel(State*){}
void enforcePeriodicSstFluxAntisymmetryKernel(State*){}
void applySstFluxAndSourceKernel(State*,double,bool){}
void applyGasFluxDivergenceByCellKernel(State*s,double dt,bool){s->conserved+=dt*s->primitive;}
void saveGasConservativeStateKernel(State*s){s->initial=s->conserved;}
void blendGasConservativeStateKernel(State*s,double a,double b,bool){s->conserved=a*s->initial+b*s->conserved;}
"""+schedule+r"""
int main(){
 for(int rk:{1,2,3})for(int sst:{0,3})for(bool refined:{false,true}){
  State s;s.hostGasTimeIntegrator=rk;s.hostTurbulenceModel=sst;s.expectedRefinement=refined;
  const double dt=.125;
  assert(advanceGasFluxStage(&s,dt,7.,refined)==0);
  const double expected=2.*(1.+dt+(rk>=2?.5*dt*dt:0.)+(rk==3?dt*dt*dt/6.:0.));
  assert(std::abs(s.conserved-expected)<1e-14);
  assert(s.primitive==s.conserved&&s.recoveries==2*rk);
  assert(s.sstRecoveries==(sst?(rk==1?2:rk==2?5:8):0));
  assert(std::abs(s.ledger-dt)<1e-15);
 }
 State s;s.primitive=-999;
 assert(advanceGasEulerSubstage(&s,.125,.125,7.)==0);
 assert(s.recoveries==2&&s.conserved==2.25&&s.primitive==s.conserved);
 std::puts("Euler/RK2/RK3, SST and refinement epochs, direct entry passed");
}
"""
    cpp=tmp_path/'epochs.cpp';cpp.write_text('#include <initializer_list>\n'+code)
    exe=tmp_path/'epochs'
    built=subprocess.run(['g++','-std=c++17','-O2','-fsanitize=undefined',str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    ran=subprocess.run([str(exe)],capture_output=True,text=True)
    assert ran.returncode==0,ran.stdout+ran.stderr
