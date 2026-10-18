"""Selected surface-root configuration errors retain actual physical evidence."""
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT

@pytest.fixture(scope="module")
def helper(tmp_path_factory):
    out=tmp_path_factory.mktemp("surface-preflight");exe=out/"probe"
    command=["g++","-std=c++17","-O2","-ffp-contract=off","-DUGKWP_GAS_SPECIES=2","-I"+str(SOURCE_ROOT/"applications/CHMT"),"-I"+str(SOURCE_ROOT/"common"),str(SOURCE_ROOT/"validation/tests/fixtures/surface_preflight_provenance.cpp"),"-o",str(exe)]
    p=subprocess.run(command,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    return exe

@pytest.mark.parametrize("mode",["R-inf","R-inf-primitive","R-negative-primitive","D-inf","model-invalid","valid","inactive","boundary-preflight"])
def test_surface_owner_configuration_provenance(helper,mode):
    p=subprocess.run([str(helper),mode],capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
