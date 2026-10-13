"""T09 actual bodies, with CUDA launch syntax translated only (no GPU claim)."""
from pathlib import Path
import re
import subprocess
import pytest
from test_gas_advance_view import PREAMBLE
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

ROOT=Path(__file__).resolve().parents[2]

def host_probe(tmp_path, extra, main):
    body=re.sub(r'<<<[\s\S]*?>>>','',(ROOT/'common/GpuGasAdvance.cuh').read_text())
    p=tmp_path/'probe.cpp';p.write_text('#include <cmath>\n'+PREAMBLE+extra+body+main)
    exe=tmp_path/'probe'
    b=subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'common'),str(p),'-o',str(exe)],capture_output=True,text=True)
    assert b.returncode==0,b.stderr
    r=subprocess.run([str(exe)],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr


def test_interval_has_no_absolute_time_minimum_or_attempt_gate(tmp_path):
    host_probe(tmp_path,r'''
struct Host:IndependentHost{double value=0,saved=0,last=0;int calls=0,rollbacks=0;};
struct Policy{
 static int begin(Host*h){h->saved=h->value;return 0;}
 static int applySources(Host*h,double dt){h->last=dt;h->value+=dt;++h->calls;return 0;}
 static int validate(Host*){return 0;}static int commit(Host*){return 0;}
 static void rollback(Host*h){h->value=h->saved;++h->rollbacks;}
 static int beginInterval(Host*){return 0;}static int commitInterval(Host*){return 0;}
 static void rollbackInterval(Host*){}static bool retryable(Host*){return false;}
};
''',r'''
int main(){Host h;Payload p;h.deviceState=&p;GasRequestedIntervalControls<double> c;c.minimumSubstep=99;c.maximumAttempts=0;
if(advanceGasRequestedInterval<Policy>(&h,1.,1e20,c)||h.value!=1||h.calls!=1)return 1;
for(int i=0;i<10001;++i){trace.clear();if(advanceGasRequestedInterval<Policy>(&h,1.,1e20,c))return 2;}
if(h.calls!=10002||h.value!=10002||h.rollbacks)return 3;
if(advanceGasRequestedInterval<Policy>(&h,1e-50,1e20,c)||h.last!=1e-50)return 4;
const double before=h.value;failLaunch=launchCount+1;
if(advanceGasRequestedInterval<Policy>(&h,.25,1e20,c)!=1||h.value!=before||h.rollbacks!=1)return 5;
}
''')

@pytest.mark.parametrize('reverse',[False,True])
def test_periodic_average_ignores_unused_frame_and_reciprocity(tmp_path,reverse):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.nFaces=2;s.nInternalFaces=0;s.facePeriodicPair[0]=1;s.facePeriodicPair[1]=REVERSE?0:1;s.gasGeometry.enabled=true;s.gasGeometry.faceSweptVolume=nullptr;s.gasGeometry.interval=0;
Real* fields[]={s.gasPhiRho,s.gasPhiRhoUx,s.gasPhiRhoUy,s.gasPhiRhoUz,s.gasPhiRhoE};for(int k=0;k<5;++k){fields[k][0]=2*(k+1);fields[k][1]=-4*(k+1);}s.gasSpecies.flux[0]=1;s.gasSpecies.flux[1]=-2;s.gasSpecies.flux[2]=1;s.gasSpecies.flux[3]=-2;
threadIdx.x=REVERSE?1:0;enforcePeriodicGasFluxAntisymmetryKernel(&s);threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&s);
for(int k=0;k<5;++k)ck(fields[k][0]==3*(k+1)&&fields[k][1]==-3*(k+1),"complete periodic five-field output");
ck(s.gasSpecies.flux[0]==Real(1.5)&&s.gasSpecies.flux[1]==Real(-1.5)&&s.gasSpecies.flux[2]==Real(1.5)&&s.gasSpecies.flux[3]==Real(-1.5),"complete periodic species output");ck(!s.gasSpecies.faceStatus[0]&&!s.gasSpecies.faceStatus[1],"unused frame certificate blocked average");}
'''.replace('REVERSE','true' if reverse else 'false'))

@pytest.mark.parametrize('operation',['convective','diffusion','sweep_backing'])
def test_courant_required_error_does_not_become_finite_sentinel(tmp_path,operation):
    bodies={
      'convective':'s.gasPhiRhoE[0]=std::numeric_limits<Real>::infinity();computeGasConvectiveCourantByCellKernel(&s,1.);ck(s.gasSpecies.cellStatus[0]!=0,"required overflow hidden as finite Co");',
      'diffusion':'s.gasMu=std::numeric_limits<Real>::infinity();computeGasDiffusionNumberKernel(&s,1.,.5);ck(s.gasSpecies.cellStatus[0]!=0,"required diffusion overflow hidden");',
      'sweep_backing':'s.gasGeometry.enabled=true;s.gasGeometry.faceSweptVolume=nullptr;computeGasCourantFieldKernel(&s,1.);ck(s.gasSpecies.faceStatus[0]==int(ugkwp::GasTransportCode::InvalidStorage),"missing consumed sweep backing hidden");',
    }
    compile_probe(tmp_path,fixture()+'\nint main(){State s;initialise(s);threadIdx.x=0;'+bodies[operation]+'}')

def test_moving_energy_limited_linear_publishes_existing_nonzero_correction(tmp_path):
    from test_legacy_ale_transport import fixture as legacy_fixture
    compile_probe(tmp_path,legacy_fixture()+r'''
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={1,1},sweep[3]={Real(.002),0,0};moving(s,oldV,newV,sweep,Real(.001));
s.Ux[0]=20;s.Ux[1]=30;s.Tgas[0]=400;s.Tgas[1]=700;s.p[0]=s.rho[0]*s.Rgas*400;s.p[1]=s.rho[1]*s.Rgas*700;
s.gradTX[0]=300;s.gradTX[1]=300;s.gradUxX[0]=12;s.gradUxX[1]=12;
Real base[5],out[5];threadIdx.x=0;s.gasReconstruction=0;
ck(computeRiemannGasFaceFluxDevice<false>(s,0,base[0],base[1],base[2],base[3],base[4]),"moving baseline flux unavailable");
const ugkpinterpolation::Vector3 gh0{s.gasCp*s.gradTX[0],0,0},gh1{s.gasCp*s.gradTX[1],0,0},gk0{s.Ux[0]*s.gradUxX[0],0,0},gk1{s.Ux[1]*s.gradUxX[1],0,0},dx{1,0,0};
const Real h0=s.gasCp*s.Tgas[0],h1=s.gasCp*s.Tgas[1],k0=Real(.5)*s.Ux[0]*s.Ux[0],k1=Real(.5)*s.Ux[1]*s.Ux[1];
const Real h=ugkpinterpolation::limitedLinearFaceValue(h0,h1,gh0,gh1,dx,Real(.5),base[0],Real(1));
const Real k=ugkpinterpolation::limitedLinearFaceValue(k0,k1,gk0,gk1,dx,Real(.5),base[0],Real(1));
const Real upwind=base[0]>=0?h0+k0:h1+k1;
const Real expected=ugkpinterpolation::limitedLinearRiemannEnergyFlux(base[4],base[0],upwind,h+k);
ck(expected!=base[4],"fixture did not create nonzero energy correction");s.gasReconstruction=2;
for(auto&v:out)v=-777;
ck(computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]),"moving existing energy reconstruction refused");
for(int k=0;k<4;++k)ck(out[k]==base[k],"moving correction changed another flux component");near(out[4],expected,"moving energy correction differs from existing formula");
}
''')

@pytest.mark.parametrize('rk',[2,3])
def test_moving_rk_executes_real_substages_and_blends(tmp_path,rk):
    from test_legacy_ale_transport import fixture as legacy_fixture
    from test_current_gas_courant import launches
    fake=PREAMBLE.split('#ifdef LEGACY_REFERENCE',1)[0]
    # The native host fixture already provides the common CUDA syntax macros.
    extra=r'''
struct Host:State{
 State*deviceState=this;int fixedCellBlockThreads=4,fixedFaceBlockThreads=4,hostGasTimeIntegrator=RK,hostTurbulenceModel=0,hostGasFluxScheme=1,hasPeriodicFaces=0;
 cudaStream_t gasCaptureStream=nullptr;cudaGraph_t gasGraph=nullptr;cudaGraphExec_t gasGraphExec=nullptr;double gasGraphDt=-1,wallDt=0;
 std::vector<cudaGraphNode_t>gasGraphTimeNodes;std::vector<cudaKernelNodeParams>gasGraphTimeParams;
};
int wall(Host*h,double dt){h->wallDt+=dt;return 0;}
#include "GpuGasHostPolicy.cuh"
using GasHostPolicy=GasHostWithWallEnergy<double,double,wall>;
int kernelLaunches=0;
template<class F>void emulateLaunch(F f,int grid,int block){++kernelLaunches;blockDim.x=block;for(blockIdx.x=0;blockIdx.x<grid;++blockIdx.x)for(threadIdx.x=0;threadIdx.x<block;++threadIdx.x)f();}
template<class S,class T>void advanceGasChemistryKernel(S*s,T dt,const Real*vol,bool nativeRefinement=false){ck(!nativeRefinement,"public stage acquired native owner");int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<s->nCells)ugkwp::advanceGasChemistryCell(*s,c,dt,vol[c],nativeRefinement);}
struct Policy{
 static int begin(Host*){return 0;}static int applySources(Host*,double){return 0;}
 static int validate(Host*h){for(int c=0;c<h->nCells;++c)if(h->gasSpecies.cellStatus[c])return 1;for(int f=0;f<h->nFaces;++f)if(h->gasSpecies.faceStatus[f])return 1;return 0;}
 static int commit(Host*){return 0;}static void rollback(Host*){}
 static const Real*stageVolumes(Host*h,bool after){return after?h->gasGeometry.newVolume:h->gasGeometry.oldVolume;}
 static int captureChemistryAudit(Host*,bool){return 0;}
};
'''.replace('RK',str(rk))
    header=launches((ROOT/'common/GpuGasAdvance.cuh').read_text())
    main=r'''
int main(){Host a,b;initialise(a);initialise(b);Real oldV[2]={1,1},newV[2]={Real(1.001),Real(.999)},sweep[3]={Real(.001),0,0};const double dt=1e-5;
for(Host*h:{&a,&b}){moving(*h,oldV,newV,sweep,Real(dt));h->rhoUx[0]=2;h->rhoUx[1]=4;h->rhoE[0]+=1;h->rhoE[1]+=4;}
ck(advanceGasTrial<Policy>(&a,dt,1e20)==0,"moving RK real stages refused");const int count=kernelLaunches;
// Independent explicit use of the pre-existing Euler and blend body, retaining
// the real stage values and original RK weights/ledger order.
ck(prepareGasTrialTransport<Policy>(&b,dt,1e20)==0,"reference preparation");
for(int c=0;c<b.nCells;++c){blockDim.x=1;blockIdx.x=0;threadIdx.x=c;saveGasConservativeStateKernel(b.deviceState);}
const double first=RK==2?dt*.5:dt*(1./6.);
ck(advanceGasEulerSubstage(&b,dt,first,1e20)==0&&advanceGasEulerSubstage(&b,dt,first,1e20+dt)==0,"reference Euler stages");
if(RK==2)ck(blendGasRungeKuttaStage(&b,.5,.5)==0,"reference RK2 blend");
else{ck(blendGasRungeKuttaStage(&b,.75,.25)==0,"reference RK3 first blend");ck(advanceGasEulerSubstage(&b,dt,dt*(2./3.),1e20+dt*.5)==0,"reference third stage");ck(blendGasRungeKuttaStage(&b,1./3.,2./3.)==0,"reference RK3 final blend");}
ck(finaliseGasTrialSources<Policy>(&b,dt)==0&&finaliseGasBoundaryStage(&b,dt,1e20+dt)==0,"reference finalisation");
Real*left[]={a.rho,a.rhoUx,a.rhoUy,a.rhoUz,a.rhoE,a.Ux,a.Uy,a.Uz,a.p,a.Tgas};Real*right[]={b.rho,b.rhoUx,b.rhoUy,b.rhoUz,b.rhoE,b.Ux,b.Uy,b.Uz,b.p,b.Tgas};
for(int k=0;k<10;++k)for(int c=0;c<2;++c)ck(std::memcmp(&left[k][c],&right[k][c],sizeof(Real))==0,"moving RK complete output differs from existing stage formula");
ck(a.rho[0]!=2&&count>20,"moving stages did not compute");ck(std::abs(a.wallDt-dt)<1e-20&&a.wallDt==b.wallDt,"RK ledger duration/order changed");
}
'''.replace('RK',str(rk))
    compile_probe(tmp_path,legacy_fixture()+fake+'#include "operators/advanceGasChemistryKernel.cuh"\n'+extra+header+main)

@pytest.mark.parametrize('missing',['pointer','index'])
def test_periodic_actual_required_pair_address_stays_error(tmp_path,missing):
    edit='s.facePeriodicPair=nullptr;' if missing=='pointer' else 's.facePeriodicPair[0]=s.nFaces;'
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.nFaces=2;s.nInternalFaces=0;EDIT
const Real before=s.gasPhiRho[0];threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&s);
ck(s.gasSpecies.faceStatus[0]!=0&&s.gasPhiRho[0]==before,"missing required periodic address silently skipped");}
'''.replace('EDIT',edit))
