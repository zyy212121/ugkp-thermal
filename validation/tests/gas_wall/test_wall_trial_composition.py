"""Accepted wall Newton states must equal the feasible states used by their residual."""
from pathlib import Path
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT
FIXTURE=Path(__file__).parent/'fixtures/wall_trial_composition.cpp'
@pytest.fixture(scope='module')
def real_wall(tmp_path_factory):
 p=tmp_path_factory.mktemp('wall-trial-composition')/'probe'
 q=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(SOURCE_ROOT),str(FIXTURE),'-o',str(p)],capture_output=True,text=True)
 assert q.returncode==0,q.stdout+q.stderr
 return p
@pytest.mark.parametrize('case',[11,25])
def test_valid_chemical_wall_keeps_accepted_composition_feasible(real_wall,case):
 q=subprocess.run([str(real_wall),str(case)],capture_output=True,text=True)
 assert q.returncode==0,q.stdout+q.stderr

def test_multiple_species_accepted_newton_state_is_a_simplex(tmp_path):
 executable=tmp_path/'multi'
 q=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(SOURCE_ROOT),str(FIXTURE.with_name('wall_simplex_update.cpp')),'-o',str(executable)],capture_output=True,text=True)
 assert q.returncode==0,q.stdout+q.stderr
 q=subprocess.run([str(executable)],capture_output=True,text=True)
 assert q.returncode==0,q.stdout+q.stderr

def test_simplex_projection_satisfies_kkt_and_is_idempotent(tmp_path):
 executable=tmp_path/'projection'
 q=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(SOURCE_ROOT),str(FIXTURE.with_name('wall_simplex_projection.cpp')),'-o',str(executable)],capture_output=True,text=True)
 assert q.returncode==0,q.stdout+q.stderr
 q=subprocess.run([str(executable)],capture_output=True,text=True)
 assert q.returncode==0,q.stdout+q.stderr
