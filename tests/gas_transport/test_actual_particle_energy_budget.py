"""HOST budgets from actual particle publications, never intended transfers.

The optional baseline root is an exact old source snapshot, not a reimplemented
old exchange model. See docs/development/energy-fp64-20261011/particle-budget.md.
"""
from decimal import Decimal, localcontext
import json
import math
import os
from pathlib import Path
import subprocess

import pytest

from test_authoritative_energy_sources import source_fixture
from test_mixture_state import ROOT, HERE


def particle_fixture():
    source = source_fixture()
    source = source.replace('struct DeviceState:State{', r'''
#include "wall/GpuFiniteWallContact.H"
#include "wall/GpuColdWallSolidification.H"
#include "wall/GpuCapillaryDetachment.H"
#include "wall/GpuParticleWallContactHeat.H"
struct DeviceState:State{
 int particleCapacity=1, particleCount=1, *particleCountDevice=&particleCount;
 int pStatus[1]={1},pCellId[1]={0},pStuckFaceId[1]={-1},cellParticleCount[2]={};
 unsigned char pStuck[1]={0},particleStuckCandidateMask[3]={};
 Real pux[1]={},puy[1]={},puz[1]={},puxOld[1]={},puyOld[1]={},puzOld[1]={};
 Real pTheta[1]={},pT[1]={300},pm[1]={100},pd[1]={Real(.001)};
 Real invRhoSolid=Real(.001),gasPrOneThird=0;
 int dragModelId=1,particleWallHeatTransferEnabled=0;
 double pContactDuration[1]={},pContactMaximumArea[1]={},pContactPeakFraction[1]={},pDepositionArea[1]={},pColdFrozenArea[1]={};
 Time pContactAge[1]={};
 Real *particleWallEffusivityByFace=nullptr,particleWallDensityKgM3=0,particleWallSpecificHeatJkgK=0,particleWallConductivityWmK=0;
 Real particleWallReflectionHeatTransferEfficiency=0,particleWallContactAreaScale[3]={};
 Real particleWallAdhesionEnergyScale=0,particleWallContactAngleCosine=0;
 double particleWallReflectedEnergy[3]={};
 Foam::gpuThermal::ColdWallSolidificationParameters coldWallSolidificationParameters{};
''')
    return source + r'''
#define GPU_THERMAL_RELAX_NATIVE_ORDER 0
#define GPU_CONTACT_AGE(s,i) s.pContactAge[i]
#define GPU_CONTACT_TIME_ZERO Time(0)
#define GPU_RESET_CONTACT_AGE(s,i) s.pContactAge[i]=0;
ThreadIndex gridDim{1};
bool gasDragModelActive(int model){return model!=0;}
template<class Model>Real dragInverseTimeDevice(const DeviceState&,Real,Real,Real,Real,Model){return Real(1);}
void atomicAddParticleWallEnergyByFace(DeviceState&,double*,int,Real){std::abort();}
void clearColdWallParticleState(DeviceState&,int){std::abort();}
void clearColdWall2DParticleState(DeviceState&,int){std::abort();}
Real particleSpecificEnthalpyDevice(Real t){return Real(1000)*t;}
Real particleMomentThetaDevice(const DeviceState&s,int i){return s.pStuck[i]==0?clampMin(finiteOr(s.pTheta[i],Real(0)),Real(0)):Real(0);}
template<class T>void atomicAdd(T* p,T v){*p+=v;}
#include "GpuThermalParticleRelaxation.cuh"
#include "operators/accumulateParticleMomentsAtomicKernel.cuh"
#include "operators/normalizeParticleMomentsAtomicKernel.cuh"
void publish(DeviceState&s){
 s.momRhoP[0]=s.momRhoUPx[0]=s.momRhoUPy[0]=s.momRhoUPz[0]=s.momRhoEP[0]=s.momRhoPD[0]=s.momRhoHpP[0]=0;
 s.cellParticleCount[0]=0;accumulateParticleMomentsAtomicKernel(&s);normalizeParticleMomentsAtomicKernel(&s);
}
void printState(const char* label,const DeviceState&s){
 const double fields[]={double(Real(1)-solidEpsFromMomentDevice(s,0)),double(s.rho[0]),double(s.rhoUx[0]),double(s.rhoUy[0]),double(s.rhoUz[0]),double(s.rhoE[0]),double(s.pm[0]),double(s.pux[0]),double(s.puy[0]),double(s.puz[0]),double(s.pTheta[0]),double(s.pT[0]),double(s.momRhoHpP[0]),double(s.momRhoEP[0]),double(s.momRhoUPx[0]),double(s.momRhoUPy[0]),double(s.momRhoUPz[0]),double(s.Ux[0]),double(s.Uy[0]),double(s.Uz[0])};
 std::printf("\"%s\":[",label);for(int j=0;j<20;++j)std::printf("%s%.17g",j?",":"",fields[j]);std::printf("]");
}
int main(int argc,char**argv){
 const int scenario=std::atoi(argv[1]);DeviceState s;setup(s);threadIdx.x=0;
 s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s.V[0]=1;
 s.gasPrOneThird=std::pow(s.gasPrClamped,Real(1)/Real(3));
 s.rho[0]=1;s.Ux[0]=s.Uy[0]=s.Uz[0]=0;s.Tgas[0]=300;
 s.pux[0]=s.puy[0]=s.puz[0]=0;s.pTheta[0]=0;s.pT[0]=300;
 double dt=.25;
 if(scenario==0){dt=0;s.rho[0]=Real(1.1);s.Ux[0]=Real(19999920)/s.rho[0];s.pux[0]=s.Ux[0];}
 if(scenario==1){s.Ux[0]=s.pux[0]=8;s.Uy[0]=s.puy[0]=-4;s.Uz[0]=s.puz[0]=2;}
 if(scenario==2){s.Ux[0]=4;s.Uy[0]=-3;s.Uz[0]=2;s.pux[0]=-2;s.puy[0]=1;s.puz[0]=-.5;s.pTheta[0]=Real(.0002);}
 if(scenario==3){s.rho[0]=Real(1.1);s.Ux[0]=Real(19999920)/s.rho[0];s.pux[0]=s.Ux[0]-Real(128);s.Uy[0]=-3;s.puy[0]=1;s.Uz[0]=2;s.puz[0]=-.5;s.pTheta[0]=Real(.0002);}
 if(scenario==4){s.rho[0]=5;s.pTheta[0]=Real(.0002);}
 if(scenario>=5){s.Ux[0]=s.pux[0]=40000;s.Tgas[0]=500;s.gasMu=Real(1e-5);s.dragModelId=0;if(scenario==6)dt=1e-8;}
 s.solveParticleTemperature=scenario>=5;
 s.rhoUx[0]=s.rho[0]*s.Ux[0];s.rhoUy[0]=s.rho[0]*s.Uy[0];s.rhoUz[0]=s.rho[0]*s.Uz[0];
 const double kin=(double(s.rhoUx[0])*s.rhoUx[0]+double(s.rhoUy[0])*s.rhoUy[0]+double(s.rhoUz[0])*s.rhoUz[0])/(2*double(s.rho[0]));
 s.rhoE[0]=kin+double(s.rho[0])*double(s.Rgas)*double(s.Tgas[0])/(double(s.gammaGas)-1);publish(s);
 std::printf("{");printState("before",s);
 if(scenario<5)drag<0>(&s,dt,0);
 else {s.couplingRhoOld[0]=s.rho[0];s.couplingUxOld[0]=s.Ux[0];s.couplingUyOld[0]=s.Uy[0];s.couplingUzOld[0]=s.Uz[0];s.couplingTgasOld[0]=s.Tgas[0];applyEulerianParticleMaterialHeatKernel(&s,dt);}
 std::printf(",");printState("gas_published",s);
 relaxOneParticleToResidentGas(s,0,dt,0);publish(s);
 std::printf(",");printState("after",s);
 std::printf(",\"dt\":%.17g,\"gamma\":%.17g,\"Rgas\":%.17g,\"conductivity\":%.17g,\"diameter\":%.17g,\"rhoSolid\":%.17g,\"tg\":%.17g}\n",dt,double(s.gammaGas),double(s.Rgas),double(molecularGasConductivity(s)),double(s.pd[0]),double(s.rhoSolid),double(s.Tgas[0]));
}
'''


FIELDS = ('eps', 'rho', 'gx', 'gy', 'gz', 'ge', 'mass', 'px', 'py', 'pz',
          'theta', 'temperature', 'published_h', 'published_mechanical',
          'published_px', 'published_py', 'published_pz',
          'gas_ux_cache', 'gas_uy_cache', 'gas_uz_cache')


@pytest.fixture()
def probe(tmp_path, request):
    return compile_particle_probe(tmp_path), 64


def compile_particle_probe(tmp_path, baseline=None):
    source = particle_fixture()
    if baseline:
        source = source.replace('GasStateView<Real,Time,ugkwp::SstCoefficients,double>',
                                'GasStateView<Real,Time,ugkwp::SstCoefficients>')
        source = source.replace('s.rhoE=new double[8]{};', 's.rhoE=new Real[8]{};')
        source = source.replace('s.rhoENext=new double[8]{};', 's.rhoENext=new Real[8]{};')
    root = Path(baseline) if baseline else ROOT
    legacy = (HERE / 'gas_state_probe.cpp').read_text()
    path = tmp_path / 'particle_budget.cpp'
    path.write_text(legacy[:legacy.index('void hashBytes')] + source)
    exe = tmp_path / 'particle_budget'
    result = subprocess.run(['g++', '-std=c++17', '-O0', '-I'+str(HERE), '-I'+str(root/'common'),
                             '-I'+str(root/'common/gasNumerics'), str(path), '-o', str(exe)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return exe


def observe(exe, scenario):
    run = subprocess.run([str(exe), str(scenario)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(run.stdout)
    for stage in ('before', 'gas_published', 'after'):
        result[stage] = dict(zip(FIELDS, result[stage]))
    return result


def budget(state):
    """Independent physical totals evaluated from the fields actually stored."""
    with localcontext() as ctx:
        ctx.prec = 70
        s = {k: Decimal.from_float(float(v)) for k, v in state.items()}
        mechanical = s['mass'] * (sum(s[k]**2 for k in ('px', 'py', 'pz')) / 2
                                   + Decimal('1.5') * s['theta'])
        enthalpy = s['mass'] * 1000 * s['temperature']
        gas_energy = s['eps']*s['ge']
        gas_kinetic = s['eps']*sum(s['g'+axis]**2 for axis in 'xyz')/(2*s['rho'])
        unresolved = Decimal('1.5')*s['mass']*s['theta']
        return {'mass': s['eps']*s['rho']+s['mass'],
                **{axis: s['eps']*s['g'+axis]+s['mass']*s['p'+axis] for axis in 'xyz'},
                'energy': s['eps']*s['ge']+mechanical+enthalpy,
                'mechanical': mechanical, 'enthalpy': enthalpy,
                'gas_energy': gas_energy, 'gas_kinetic': gas_kinetic, 'gas_internal': gas_energy-gas_kinetic,
                'particle_resolved': mechanical-unresolved, 'particle_unresolved': unresolved,
                'gas_mass': s['eps']*s['rho'], 'particle_mass': s['mass'],
                'published_energy': s['eps']*s['ge']+s['published_mechanical']+s['published_h'],
                **{'published_'+axis: s['eps']*s['g'+axis]+s['published_p'+axis] for axis in 'xyz'}}


def changes(observation):
    before, after = budget(observation['before']), budget(observation['after'])
    return {key: float(after[key]-before[key]) for key in before}


def assert_publications(o):
    """Moment publication is checked too, without using it as the total oracle."""
    for stage in ('before', 'after'):
        s = o[stage]
        assert s['mass'] == 100
        assert math.isclose(s['published_h'], s['mass']*1000*s['temperature'],
                            rel_tol=3e-16)
        assert math.isclose(s['published_mechanical'], float(budget(s)['mechanical']),
                            rel_tol=3e-16, abs_tol=1e-20)
        for axis in 'xyz':
            assert math.isclose(s['published_p'+axis], s['mass']*s['p'+axis],
                                rel_tol=2e-16, abs_tol=1e-12)
    assert changes(o)['mass'] == 0


@pytest.mark.parametrize('scenario', [0, 1], ids=['zero_time', 'zero_exchange'])
def test_actual_particle_and_gas_zero_exchange_budgets(probe, scenario):
    exe, _ = probe
    o = observe(exe, scenario)
    assert_publications(o)
    assert o['after'] == o['before']
    assert all(value == 0 for value in changes(o).values())


@pytest.mark.parametrize('scenario', [2, 3], ids=['finite_drag', 'high_kinetic_drag'])
def test_characterize_actual_drag_publications_and_existing_model_residual(probe, scenario):
    exe, _ = probe
    o = observe(exe, scenario)
    assert_publications(o)
    a, b = o['before'], o['after']
    ms, mg, dt = a['mass'], a['eps']*a['rho'], o['dt']
    alpha = math.exp(-dt)
    alpha_pair = math.exp(-dt*(1+ms/mg))
    expected_momentum = {}
    expected_energy = 0
    particle_publication_remainder = 0
    input_cache_remainder = 0
    for axis in 'xyz':
        ug, up = a['g'+axis]/a['rho'], a['p'+axis]
        coupled_particle = (mg*ug+ms*up-mg*(ug-up)*alpha_pair)/(mg+ms)
        frozen_particle = ug+(up-ug)*alpha
        cached_gas_velocity = a['gas_u'+axis+'_cache']
        from_cached_input = cached_gas_velocity+(up-cached_gas_velocity)*alpha
        actual_particle = b['p'+axis]
        particle_publication_remainder += .5*ms*(actual_particle-from_cached_input)*(actual_particle+from_cached_input)
        input_cache_remainder += .5*ms*(from_cached_input-frozen_particle)*(from_cached_input+frozen_particle)
        expected_momentum[axis] = ms*(frozen_particle-coupled_particle)
        expected_energy += .5*ms*(frozen_particle-coupled_particle)*(frozen_particle+coupled_particle)
        # Explicit particle publication checks distinguish it from intended gas-side state.
        assert math.isclose(b['p'+axis], frozen_particle,
                            rel_tol=0, abs_tol=4*(2**-52)*(abs(ug)+abs(up-ug)+1))
    delta = changes(o)
    # This is a positive diagnosis of a preexisting model mismatch, not a relaxed

    assert abs(delta['x']) > 1
    assert abs(delta['energy']) > 1
    for axis in 'xyz':
        tolerance = .000001
        assert abs(delta[axis]-expected_momentum[axis]) < tolerance
    # A forward-error bound scales with the stored mechanical background and
    # native precision. It diagnoses model identity, not energy conservation.
    native_epsilon = 2**(-52)
    mechanical_scale = float(budget(a)['mechanical'])
    arithmetic_budget = 4*native_epsilon*(1+mechanical_scale)+1.2e-7
    assert abs(delta['energy']-expected_energy) < arithmetic_budget
    print(json.dumps({'bits': 64, 'scenario': scenario, 'actual': delta,
                      'analytic_model_energy_residual': expected_energy,
                      'arithmetic_remainder': delta['energy']-expected_energy,
                      'particle_velocity_publication_remainder': particle_publication_remainder,
                      'input_velocity_cache_remainder': input_cache_remainder,
                      'other_gas_moment_arithmetic_remainder': delta['energy']-expected_energy-particle_publication_remainder-input_cache_remainder}, sort_keys=True))


def test_actual_theta_publication_is_measured_in_total_budget(probe):
    exe, _ = probe
    o = observe(exe, 4)
    assert_publications(o)
    a, b = o['before'], o['after']
    assert 0 < b['theta'] < a['theta']
    assert b['temperature'] == a['temperature']
    delta = changes(o)
    assert delta['x'] == delta['y'] == delta['z'] == 0
    # The gas gain is compared with the actual particle theta write, not alpha*dI.
    assert abs(delta['energy']) < 1.2e-7
    print(json.dumps({'bits': 64, 'scenario': 4, 'actual': delta}, sort_keys=True))


@pytest.mark.parametrize('scenario', [5, 6], ids=['finite_heat', 'sub_ulp_particle_heat'])
def test_characterize_actual_temperature_and_material_publication_heat_residual(probe, scenario):
    exe, _ = probe
    o = observe(exe, scenario)
    assert_publications(o)
    a, b = o['before'], o['after']
    rate = 12*o['conductivity']/(o['rhoSolid']*1000*o['diameter']**2)
    expected_temperature = o['tg']+(a['temperature']-o['tg'])*math.exp(-rate*o['dt'])
    cg = a['eps']*a['rho']*o['Rgas']/(o['gamma']-1)
    cs = a['mass']*1000
    model_particle_gain = cs*(o['tg']-a['temperature'])*(-math.expm1(-rate*o['dt']))
    model_gas_loss = cg*cs/(cg+cs)*(o['tg']-a['temperature'])*(-math.expm1(-rate*(1+cs/cg)*o['dt']))
    delta = changes(o)
    assert delta['x'] == delta['y'] == delta['z'] == 0
    assert math.isclose(b['temperature'], expected_temperature, rel_tol=3e-16)
    if scenario == 5:
        assert delta['energy'] > 100000
        assert abs(delta['energy']-(model_particle_gain-model_gas_loss)) < (1e-6)
    else:
        assert b['temperature'] > a['temperature']
        assert abs(delta['energy']) < 1.2e-7
    print(json.dumps({'bits': 64, 'scenario': scenario, 'actual': delta,
                      'analytic_model_energy_residual': model_particle_gain-model_gas_loss,
                      'particle_publication_remainder': delta['enthalpy']-model_particle_gain}, sort_keys=True))


def test_optional_exact_baseline_particle_publication_evidence(tmp_path):
    baseline = os.environ.get('UGKWP_PARTICLE_BUDGET_BASELINE_ROOT')
    if not baseline:
        pytest.skip('Set UGKWP_PARTICLE_BUDGET_BASELINE_ROOT to the exact baseline snapshot')
    exe = compile_particle_probe(tmp_path, baseline)
    current_path = tmp_path / 'current'
    current_path.mkdir()
    current_exe = compile_particle_probe(current_path)
    for scenario in range(7):
        o = observe(exe, scenario)
        assert_publications(o)
        current = observe(current_exe, scenario)
        for field in ('mass', 'px', 'py', 'pz', 'theta', 'temperature',
                      'published_h', 'published_mechanical', 'published_px',
                      'published_py', 'published_pz'):
            assert o['after'][field] == current['after'][field]
        print(json.dumps({'baseline': str(baseline), 'bits': 64, 'scenario': scenario,
                          'actual': changes(o), 'before': o['before'], 'after': o['after']}, sort_keys=True))
