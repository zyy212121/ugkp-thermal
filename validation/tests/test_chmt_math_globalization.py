"""Mathematical domain safeguards exercise the real CHMT production operators."""
import pathlib,subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT
@pytest.fixture(scope="module")
def math_helper(tmp_path_factory):
    out=tmp_path_factory.mktemp("chmt-math");exe=out/"helper"
    cmd=["g++","-std=c++17","-O1","-ffp-contract=off","-DUGKWP_GAS_SPECIES=2","-I"+str(SOURCE_ROOT/"applications/CHMT"),"-I"+str(SOURCE_ROOT/"common"),str(SOURCE_ROOT/"validation/tests/fixtures/chmt_math_globalization.cpp"),"-o",str(exe)]
    p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0,p.stdout+p.stderr
    return exe
@pytest.mark.parametrize("mode",["thermal800","arrhenius","rate-zero","surface706","surface500","missing"])
def test_chmt_math_domain_and_error_controls(math_helper,mode):
    p=subprocess.run([str(math_helper),mode],capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
