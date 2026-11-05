"""Run actual shared operators with optional steady local steps; no CUDA claim."""
import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture
from test_authoritative_energy_sources import source_fixture


def steady_fixture():
    return fixture().replace(' gasSpecies;};', ' gasSpecies; double* gasPseudoDt=nullptr; double gasPseudoReferenceDt=1;};')


def test_optional_time_capabilities_and_subintervals(tmp_path):
    compile_probe(tmp_path, r'''
#include "gasTransport/GasPseudoTime.H"
struct Empty {};
struct Steady { double* gasPseudoDt=nullptr;double gasPseudoReferenceDt=2;bool gasSteadyMode=false;double gasSteadyBoundaryTime=7;};
int main(){Empty e;Steady s;double steps[]={.25,.75};
if(ugkwp::gasCellTimeStep(e,123,.125)!=.125 || ugkwp::gasStageTime(e,9.)!=9.)return 1;
if(ugkwp::gasCellTimeStep(s,0,.125)!=.125 || ugkwp::gasStageTime(s,9.)!=9.)return 2;
s.gasPseudoDt=steps;s.gasSteadyMode=true;
if(ugkwp::gasCellTimeStep(s,0,2.)!=.25 || ugkwp::gasCellTimeStep(s,1,1.)!=.375)return 3;
if(ugkwp::gasStageTime(s,9.)!=7.)return 4;
}
''')


@pytest.mark.parametrize('operator,fields', [
    ('computeGasFluxPositivityScaleKernel', ['gasFluxPositivityScale']),
    ('applyGasFluxDivergenceByCellKernel', ['rho','rhoUx','rhoUy','rhoUz','rhoE']),
    ('applySstFluxAndSourceKernel', ['rhoK','rhoOmega']),
])
def test_local_cell_operators_match_transient_at_each_local_dt(tmp_path, operator, fields):
    comparisons='\n'.join(f'near(a.{f}[c],b.{f}[c],"{operator} {f}");' for f in fields)
    compile_probe(tmp_path, steady_fixture()+r'''
int main(){State a,b;initialise(a);initialise(b);double steps[]={.001,.002};a.gasPseudoDt=steps;
a.sstConfigured=b.sstConfigured=1;
for(int f=0;f<3;++f){
 a.gasPhiRho[f]=b.gasPhiRho[f]=f==0?3000:0;
 a.gasPhiRhoUx[f]=b.gasPhiRhoUx[f]=f==0?.2:0;
 a.gasPhiRhoE[f]=b.gasPhiRhoE[f]=f==0?.3:0;
 a.sstPhiRhoK[f]=b.sstPhiRhoK[f]=f==0?.4:0;
 a.sstPhiRhoOmega[f]=b.sstPhiRhoOmega[f]=f==0?.5:0;
}
// A fractional reference interval must scale local steps, not replace it.
for(int c=0;c<2;++c){threadIdx.x=c;OP(&a,.5);OP(&b,steps[c]*.5);CHECK}
}
'''.replace('OP',operator).replace('CHECK',comparisons))


def test_wave_boundary_uses_owner_cell_time_step(tmp_path):
    compile_probe(tmp_path, steady_fixture()+r'''
int main(){State a,b;initialise(a);initialise(b);double steps[]={.001,.003};a.gasPseudoDt=steps;
for(int f=1;f<3;++f){a.gasBoundaryPWave[f]=b.gasBoundaryPWave[f]=1;a.gasBoundaryP[f]=b.gasBoundaryP[f]=90000;a.gasBoundaryRho[f]=b.gasBoundaryRho[f]=2;a.gasBoundaryPWaveGamma[f]=b.gasBoundaryPWaveGamma[f]=1.4;a.gasBoundaryPWaveLInf[f]=b.gasBoundaryPWaveLInf[f]=2;a.gasBoundaryPWaveFieldInf[f]=b.gasBoundaryPWaveFieldInf[f]=95000;
threadIdx.x=f-1;updateWaveTransmissivePressureBoundaryKernel(&a,.5);updateWaveTransmissivePressureBoundaryKernel(&b,.5*steps[f-1]);near(a.gasBoundaryP[f],b.gasBoundaryP[f],"wave owner dt");}
}
''')


def test_gravity_matches_reference_fraction_without_changing_energy(tmp_path):
    body=source_fixture().replace('struct DeviceState:State{', 'struct DeviceState:State{ double* gasPseudoDt=nullptr; double gasPseudoReferenceDt=1;')
    compile_probe(tmp_path, body+r'''
int main(){DeviceState a,b;setup(a);setup(b);double steps[]={.125};a.gasPseudoDt=steps;a.gravityX=b.gravityX=.2;
applyGasGravityCell<true>(a,0,.5);applyGasGravityCell<true>(b,0,.0625);
closeEnergy(a.rhoE[0],b.rhoE[0],"local gravity energy");near(a.rhoUx[0],b.rhoUx[0],"local gravity momentum");}
''')


def test_frozen_species_positivity_and_divergence_share_local_dt(tmp_path):
    from test_mixture_transport import fixture as mixture_fixture
    body=mixture_fixture().replace(' gasSpecies;};', ' gasSpecies; double* gasPseudoDt=nullptr; double gasPseudoReferenceDt=1;};')
    compile_probe(tmp_path,body+r'''
int main(){State a,b;initialise(a);initialise(b);double steps[]={.1,.2};a.gasPseudoDt=steps;
a.gasSpecies.flux[0]=b.gasSpecies.flux[0]=.1;a.gasSpecies.flux[3]=b.gasSpecies.flux[3]=-.1;
for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&a,.5);computeGasFluxPositivityScaleKernel(&b,steps[c]*.5);
for(int k=0;k<2;++k)ck(a.gasSpecies.positivityScale[k*2+c]==b.gasSpecies.positivityScale[k*2+c],"species local positivity");
applyGasFluxDivergenceByCellKernel(&a,.5);applyGasFluxDivergenceByCellKernel(&b,steps[c]*.5);
for(int k=0;k<2;++k)ck(a.gasSpecies.rho[k*2+c]==b.gasSpecies.rho[k*2+c],"species local divergence");
ck(a.gasSpecies.cellStatus[c]==0 && b.gasSpecies.cellStatus[c]==0,"species status");
ck(a.rho[c]==b.rho[c] && a.rhoE[c]==b.rhoE[c],"species bulk consistency");}
}
''')


@pytest.mark.parametrize('boundary_flags', [
    'bool gasSteadyMode=true; double gasSteadyBoundaryTime=7;',
    'bool gasSteadyMode=false; bool gasSteadyBoundaryFrozen=true; double gasSteadyBoundaryTime=7;',
])
def test_shared_host_stages_and_graph_freeze_boundary_time(tmp_path, boundary_flags):
    import re
    import subprocess
    from test_gas_advance_view import PREAMBLE, ROOT
    host=re.sub(r'<<<[\s\S]*?>>>', '', (ROOT/'common/GpuGasAdvance.cuh').read_text())
    (tmp_path/'protocol.hpp').write_text(host)
    preamble=PREAMBLE.replace('struct HOST_TYPE {', 'struct HOST_TYPE { '+boundary_flags)
    main=r'''
#include "protocol.hpp"
int main(){IndependentHost h;Payload p;h.deviceState=&p;
for(int rk=1;rk<=3;++rk){h.hostGasTimeIntegrator=rk;trace.clear();
 if(advanceGasFluxStage(&h,.3,2.) || finaliseGasBoundaryStage(&h,.3,2.3))return 1;
 int sampled=0;for(auto&s:trace)if(s.find("updateLegacyGasBoundaryMirrorKernel ")==0){++sampled;if(s!="updateLegacyGasBoundaryMirrorKernel 7")return 2;}
 if(sampled!=2)return 3;
}
h.hostGasTimeIntegrator=1;graphNodes.clear();expectedDevice=&p;expectedTime=7;
if(advancePureGasGraph(&h,.3,2.) || advancePureGasGraph(&h,.3,99.))return 4;
}
'''
    source=tmp_path/'host.cpp';source.write_text(preamble+main)
    exe=tmp_path/'host'
    build=subprocess.run(['g++','-std=c++17','-O0','-I'+str(ROOT/'common'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stdout+build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr


def test_steady_gravity_is_old_state_rhs_with_local_step(tmp_path):
    body=steady_fixture().replace(' gasSpecies;', ' gasSpecies; bool gasSteadyMode=true; double gasSteadyBoundaryTime=0; Real gravityX=.2,gravityY=-.1,gravityZ=.3;')
    compile_probe(tmp_path,body+r'''
int main(){State s;initialise(s);double steps[]={.01,.03};s.gasPseudoDt=steps;
for(int c=0;c<2;++c){s.rhoUx[c]=3;s.rhoUy[c]=4;s.rhoUz[c]=5;}
s.gasPhiRho[0]=.1;s.gasPhiRhoUx[0]=.4;s.gasPhiRhoE[0]=.6;
for(int c=0;c<2;++c){threadIdx.x=c;const Real rho=s.rho[c],energy=s.rhoE[c];const double dt=.5*steps[c],sign=c==0?-1:1;
applyGasFluxDivergenceByCellKernel(&s,.5);
near(s.rhoUx[c],3+dt*(sign*.4+rho*s.gravityX),"unsplit gravity momentum");
near(s.rhoUy[c],4+dt*rho*s.gravityY,"unsplit gravity y");
near(s.rhoUz[c],5+dt*rho*s.gravityZ,"unsplit gravity z");
near(s.rhoE[c],energy+dt*(sign*.6+3*s.gravityX+4*s.gravityY+5*s.gravityZ),"unsplit gravity old-state energy");}
}
''')


def test_explicit_boundary_clock_is_independent_of_steady_sources(tmp_path):
    compile_probe(tmp_path,r'''
#include "gasTransport/GasPseudoTime.H"
struct Sampling { bool gasSteadyMode=false; bool gasSteadyBoundaryFrozen=true; double gasSteadyBoundaryTime=7; double*gasPseudoDt=nullptr; double gasPseudoReferenceDt=1; };
struct ClockOnly { bool gasSteadyBoundaryFrozen=true; double gasSteadyBoundaryTime=3; };
int main(){Sampling s;ClockOnly c;
if(ugkwp::gasStageTime(s,9.)!=7. || ugkwp::gasCellTimeStep(s,0,.125)!=.125)return 1;
if(ugkwp::gasStageTime(c,9.)!=3.)return 2;
s.gasSteadyBoundaryFrozen=false;s.gasSteadyMode=true;
if(ugkwp::gasStageTime(s,9.)!=9.)return 3;
}
''')


def test_freeze_only_sampling_keeps_transport_rhs_free_of_gravity(tmp_path):
    body=steady_fixture().replace(' gasSpecies;', ' gasSpecies; bool gasSteadyMode=false; bool gasSteadyBoundaryFrozen=true; double gasSteadyBoundaryTime=7; Real gravityX=.2,gravityY=-.1,gravityZ=.3;')
    compile_probe(tmp_path,body+r'''
int main(){State s;initialise(s);
for(int c=0;c<2;++c){s.rhoUx[c]=3;s.rhoUy[c]=4;s.rhoUz[c]=5;}
s.gasPhiRhoUx[0]=.4;s.gasPhiRhoE[0]=.6;
for(int c=0;c<2;++c){threadIdx.x=c;const Real energy=s.rhoE[c];const double dt=.01,sign=c==0?-1:1;
applyGasFluxDivergenceByCellKernel(&s,dt);
near(s.rhoUx[c],3+dt*sign*.4,"freeze-only transport injected gravity");
near(s.rhoUy[c],4,"freeze-only transport injected y gravity");
near(s.rhoUz[c],5,"freeze-only transport injected z gravity");
near(s.rhoE[c],energy+dt*sign*.6,"freeze-only transport injected work");}
}
''')
