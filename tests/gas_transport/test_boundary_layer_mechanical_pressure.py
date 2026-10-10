"""The physical wall force retains the existing local reconstructed pressure.

Matching pressure is the thermodynamic/BVP input, not a nonlocal acoustic
boundary condition. Execute the real flux operator, including ALE pressure work.
"""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


@pytest.mark.parametrize('bits', [32, 64])
@pytest.mark.parametrize('moving', [False, True])
@pytest.mark.parametrize('blowing', [False, True])
def test_local_pressure_force_and_sweep_work_leave_wall_transport_unchanged(tmp_path, bits, moving, blowing):
    body = r'''
#include "gasTransport/GasBoundaryLayerEvaluation.H"
void near(Real a,Real b,const char*m){if(std::abs(a-b)>Real(256)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(b))){std::fprintf(stderr,"%s: %.17g != %.17g\n",m,double(a),double(b));std::abort();}}
int main(){State s;initialise(s);s.gasReconstruction=1;s.riemannBoundaryKind[1]=2;
s.gasGradientLimiterRho[0]=s.gasGradientLimiterP[0]=s.gasGradientLimiterT[0]=s.gasSpecies.limiter[0]=1;
s.gradPx[0]=Real(12000);s.gradTX[0]=s.gradPx[0]/s.p[0]*s.Tgas[0];
const Real area=2;s.magSf[1]=area;s.Sfx[1]=-area;
const auto local=reconstructGasCellToFace(s,0,1);ck(local.p!=s.p[0],"fixture must exercise reconstructed rather than cell pressure");
auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;int fs[3]={-1,0,-1},os[2]={0,-1},status[1]={0};w.faceSlot=fs;w.ownerSlot=os;w.status=status;
ugkwp::GasBoundaryLayerExchange<Real> exchange[1];ugkwp::GasBoundaryLayerSstClosure<Real> closure[1];Real speciesFlux[2];w.exchange=exchange;w.sst=closure;w.speciesFlux=speciesFlux;
ugkwp::gaswall::WallInput<Real,2> input;input.pressure=Real(.7)*s.p[0];input.normal[0]=1;input.model.thermo=s.gasSpecies.thermo;input.quadrature.volume=1;
ugkwp::gaswall::WallOutput<Real,2> output;output.traceTemperature=700;output.traceVelocity[0]=Real(2.5);output.traceVelocity[1]=3;output.traceVelocity[2]=-1;
output.wallSpeciesFlux[0]=BLOWING?Real(.4):Real(0);output.wallSpeciesFlux[1]=BLOWING?Real(-.1):Real(0);
output.conductiveHeatFlux=50;output.traction[0]=4;output.traction[1]=5;output.traction[2]=-2;
output.matchingSpeciesFlux[0]=999;output.reactionIntegral[0]=333;
const Real dt=Real(.01),speed=MOVING?Real(.25):Real(0);Real oldV[2]={1,1},newV[2]={1-area*speed*dt,1},sweeps[3]={0,-area*speed*dt,0};
if(MOVING){s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweeps;s.gasGeometry.interval=dt;s.gasGeometry.absoluteGeometryTolerance=s.gasGeometry.relativeGeometryTolerance=Real(64)*std::numeric_limits<Real>::epsilon();}
ck(ugkwp::publishGasBoundaryLayerOutput(s,0,input,output,area,speed),"publication failed");
const auto published=exchange[0];Real m,x,y,z,e;ck(computeRiemannGasFaceFluxDevice<false>(s,1,m,x,y,z,e),"physical wall rejected");
const Real mass=output.wallSpeciesFlux[0]+output.wallSpeciesFlux[1];
near(x,-area*(mass*output.traceVelocity[0]+local.p-output.traction[0]),"normal mechanical force used nonlocal matching pressure");
near(y,published.momentumY,"tangential momentum changed");near(z,published.momentumZ,"second tangent changed");near(m,published.mass,"mass changed");
const Real h=output.wallSpeciesFlux[0]*ugkwp::speciesH(0,output.traceTemperature,input.model.thermo)+output.wallSpeciesFlux[1]*ugkwp::speciesH(1,output.traceTemperature,input.model.thermo);
const Real kinetic=Real(.5)*mass*(Real(2.5)*Real(2.5)+9+1),viscous=4*Real(2.5)+5*3+(-2)*(-1);
near(e,-area*(h+kinetic+50+local.p*speed-viscous),"same-face pressure work, species enthalpy or viscous work changed");
near(e-published.energy,-area*(local.p-input.pressure)*speed,"mechanical correction is not exactly delta-p times sweep");
for(int k=0;k<2;++k)near(s.gasSpecies.flux[k*s.nFaces+1],-area*output.wallSpeciesFlux[k],"species changed");
ck(input.pressure==Real(.7)*s.p[0]&&exchange[0].energy==published.energy,"flux evaluation mutated BVP thermodynamics/publication");
exchange[0].ready=false;ck(computeRiemannGasFaceFluxDevice<false,true>(s,1,m,x,y,z,e),"mass predictor forced profile");near(m,published.mass,"mass-only predictor changed");
}
'''.replace('MOVING', str(int(moving))).replace('BLOWING', str(int(blowing)))
    compile_probe(tmp_path, fixture()+body, bits)


def test_n128_muscl_euler_couette_reaches_original_time_and_accuracy_gate(tmp_path):
    """Same N128/endTime/.4 Co, without hiding the acoustic defect by finer mesh/RK."""
    from pathlib import Path
    import math
    import re
    import subprocess
    from test_mixture_state import ROOT, HERE
    pre=(HERE/'gas_state_probe.cpp').read_text().split('void hashBytes')[0]
    storage=re.sub(r'new (Real|int)\[\d+\]\{\}',r'new \1[2048]{}',fixture())
    source=tmp_path/'couette.cpp'
    source.write_text(pre+storage+(HERE/'boundary_layer_couette_probe.cpp').read_text())
    exe=tmp_path/'couette'
    build=subprocess.run(['g++','-std=c++17','-O2','-DUGKWP_GPU_REAL_BITS=64',
        '-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),
        str(source),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stdout+build.stderr
    run=subprocess.run([str(exe),'1'],capture_output=True,text=True,timeout=120)
    assert run.returncode==0,run.stdout+run.stderr
    (tmp_path/'couette.log').write_text(run.stdout+run.stderr)
    summary=run.stdout.splitlines()[0]
    assert 'PASS rk=1' in summary and 'time=0.050000000000000003' in summary,summary
    values=dict(re.findall(r'(maxCo|maxDiff)=([\d.e+-]+)',summary))
    assert float(values['maxCo'])<=.4*(1+1e-10),summary
    assert float(values['maxDiff'])<=.25,summary
    rows=[[float(x) for x in line.split()[1:]] for line in run.stdout.splitlines() if line.startswith('DATA ')]
    assert len(rows)==128
    # Both centre-value and true finite-volume cell-average references. Neither
    # reference changes the original .02/.05 gates.
    for cell_average in (False,True):
        errors=[]
        for y,u,_,_ in rows:
            reference=y
            for k in range(1,500):
                sinc=math.sin(k*math.pi/256)/(k*math.pi/256) if cell_average else 1
                reference+=2*(-1)**k/(k*math.pi)*math.sin(k*math.pi*y)*math.exp(-.1*(k*math.pi)**2*.05)*sinc
            errors.append(u-reference)
        l2=math.sqrt(sum(e*e for e in errors)/128)
        linf=max(abs(e) for e in errors)
        assert l2<.02 and linf<.05,(cell_average,l2,linf,summary)
    assert max(abs(row[2]) for row in rows)<1e-3


@pytest.mark.parametrize('bits', [32, 64])
def test_unavailable_matching_pressure_cannot_publish_a_ready_full_flux(tmp_path,bits):
    from test_boundary_layer_operators import SETUP
    from test_legacy_ale_transport import fixture as legacy
    compile_probe(tmp_path,legacy()+SETUP+r'''
int main(){State s;initialise(s);wall(s);Real m,x,y,z,e;
s.gasBoundaryLayer.exchange[0].matchingPressure=0;
ck(!computeRiemannGasFaceFluxDevice<false>(s,1,m,x,y,z,e),"missing thermodynamic reference pressure silently accepted");
ck(s.gasSpecies.faceStatus[1]==int(ugkwp::GasTransportCode::InvalidThermodynamics),"missing pressure failure lost");
s.gasSpecies.faceStatus[1]=0;s.gasBoundaryLayer.exchange[0].ready=false;
ck(computeRiemannGasFaceFluxDevice<false,true>(s,1,m,x,y,z,e),"known mass needs no pressure/profile solve");near(m,Real(-.4),"mass predictor changed");}
''',bits)


@pytest.mark.parametrize('bits', [32, 64])
def test_nonfinite_corrected_force_rejects_before_species_publication(tmp_path,bits):
    from test_boundary_layer_operators import SETUP
    from test_legacy_ale_transport import fixture as legacy
    compile_probe(tmp_path,legacy()+SETUP+r'''
int main(){State s;initialise(s);wall(s);s.p[0]=std::pow(Real(10),Real(sizeof(Real)==4?20:290));
s.magSf[1]=Real(1e20);s.Sfx[1]=-s.magSf[1];s.gasBoundaryLayer.exchange[0].matchingPressure=1;
Real m,x,y,z,e;ck(!computeRiemannGasFaceFluxDevice<false>(s,1,m,x,y,z,e),"overflowed mechanical force was accepted");
ck(s.gasSpecies.faceStatus[1]==int(ugkwp::GasTransportCode::NonFiniteState),"overflowed mechanical force lacks failure status");}
''',bits)
