"""Real interface equation, including its numerical domain and degenerate eigenspaces."""
from pathlib import Path
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT
FIXTURE=Path(__file__).parent/'fixtures/interface_composition'

def build(tmp_path,name,ns):
    exe=tmp_path/(name+str(ns))
    command=['g++','-std=c++17','-O2','-ffp-contract=off','-DUGKWP_GAS_SPECIES='+str(ns),'-I'+str(SOURCE_ROOT/'applications/CHMT'),'-I'+str(SOURCE_ROOT/'common'),str(FIXTURE/(name+'.cpp')),'-o',str(exe)]
    done=subprocess.run(command,capture_output=True,text=True)
    assert done.returncode==0,done.stdout+done.stderr
    return exe

def run(exe,*args):
    done=subprocess.run([str(exe),*map(str,args)],capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stdout+done.stderr

@pytest.mark.parametrize('ns',[2,10])
def test_manufactured_equations(tmp_path,ns):
    run(build(tmp_path,'manufactured',ns))

@pytest.mark.parametrize('ns',[1,2,3,10])
def test_degenerate_diffusion_and_flux(tmp_path,ns):
    run(build(tmp_path,'edges',ns))

@pytest.mark.parametrize('ns',[2,3])
def test_endpoints_and_multiple_roots(tmp_path,ns):
    run(build(tmp_path,'multiple',ns))

def test_private_trials_and_typed_outcomes(tmp_path):
    exe=build(tmp_path,'contracts',2)
    for kind in range(9):run(exe,kind)

@pytest.mark.parametrize('ns',[2,10])
def test_exact_zero_and_pure_species(tmp_path,ns):
    run(build(tmp_path,'boundary',ns))

def test_overflow_boundary_rounding(tmp_path):
    run(build(tmp_path,'arithmetic',2),10000)

def test_extreme_pole_trial_does_not_trap(tmp_path):
    run(build(tmp_path,'pole_arithmetic',10))

@pytest.mark.parametrize('ns',[2,10])
def test_proven_unique_solution_avoids_polynomial_construction(tmp_path,ns):
    run(build(tmp_path,'unique',ns))

@pytest.mark.parametrize('ns',[1,2,10])
def test_linear_uniqueness_proof_matches_all_off_diagonals(tmp_path,ns):
    run(build(tmp_path,'unique_proof',ns))

@pytest.mark.parametrize('ns',[1,2,10])
def test_certified_result_publishes_without_repeating_physical_evaluation(tmp_path,ns):
    exe=tmp_path/('publication'+str(ns))
    command=['g++','-std=c++17','-O1','-fno-inline','-finstrument-functions','-ffp-contract=off','-DUGKWP_GAS_SPECIES='+str(ns),'-I'+str(SOURCE_ROOT/'applications/CHMT'),'-I'+str(SOURCE_ROOT/'common'),str(FIXTURE/'publication.cpp'),'-o',str(exe)]
    done=subprocess.run(command,capture_output=True,text=True)
    assert done.returncode==0,done.stdout+done.stderr
    run(exe)

@pytest.mark.parametrize('ns',[1,2,10])
def test_prepared_inputs_preserve_variable_density_residual_and_jacobian(tmp_path,ns):
    run(build(tmp_path,'prepared',ns))
