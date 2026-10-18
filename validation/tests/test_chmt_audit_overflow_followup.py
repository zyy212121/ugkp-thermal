"""Real material/caller overflow cases, including OpenFOAM's enabled FPE traps."""
import os
import pathlib
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT

@pytest.fixture(scope="module")
def helper(tmp_path_factory):
    folder=tmp_path_factory.mktemp("chmt-audit-helper");exe=folder/"helper"
    cmd=["g++","-std=c++17","-O2","-DUGKWP_GAS_SPECIES=2","-I"+str(SOURCE_ROOT/"applications/CHMT"),"-I"+str(SOURCE_ROOT/"common"),str(SOURCE_ROOT/"validation/tests/fixtures/chmt_audit_overflow.cpp"),"-o",str(exe)]
    subprocess.run(cmd,check=True,capture_output=True,text=True);return exe

@pytest.mark.parametrize("mode",["film","gas","packet"])
def test_real_helper_overflow_and_normal_controls(helper,mode):
    result=subprocess.run([str(helper),mode],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr

@pytest.fixture(scope="module")
def native(tmp_path_factory):
    project=pathlib.Path(os.environ.get("WM_PROJECT_DIR","/opt/openfoam10"))
    libbin=pathlib.Path(os.environ.get("FOAM_LIBBIN",str(project/"platforms/linux64GccDPInt32Opt/lib")))
    mpi=os.environ.get("FOAM_MPI","openmpi-system")
    if not (project/"src/OpenFOAM/lnInclude/error.H").exists() or not (libbin/"libOpenFOAM.so").exists():
        pytest.skip("native OpenFOAM10 headers/libraries unavailable")
    folder=tmp_path_factory.mktemp("chmt-audit-native");exe=folder/"native"
    cmd=["g++","-std=c++17","-O2","-pthread","-DWM_DP","-DWM_LABEL_SIZE=32","-DUGKWP_GAS_SPECIES=2","-DCHMT_NATIVE_FOAM","-I"+str(SOURCE_ROOT/"applications/CHMT"),"-I"+str(SOURCE_ROOT/"common"),"-I"+str(project/"src/OpenFOAM/lnInclude"),"-I"+str(project/"src/OSspecific/POSIX/lnInclude"),str(SOURCE_ROOT/"validation/tests/fixtures/chmt_audit_overflow.cpp"),"-L"+str(libbin),"-L"+str(libbin/mpi),"-lOpenFOAM","-lPstream","-o",str(exe)]
    subprocess.run(cmd,check=True,capture_output=True,text=True)
    return exe,libbin,mpi,project

@pytest.mark.parametrize("mode",["film","gas","packet"])
@pytest.mark.parametrize("warnings",["1","0"])
def test_native_foam_fpe_and_dispatch(native,mode,warnings):
    exe,libbin,mpi,project=native;env=os.environ.copy();env.update(FOAM_SIGFPE="1",UGKP_DIAGNOSTIC_WARNINGS=warnings,WM_PROJECT_DIR=str(project),FOAM_ETC=str(project/"etc"))
    env["LD_LIBRARY_PATH"]=str(libbin)+":"+str(libbin/mpi)+":"+env.get("LD_LIBRARY_PATH","")
    result=subprocess.run([str(exe),mode],capture_output=True,text=True,env=env)
    assert result.returncode==0,result.stdout+result.stderr
    assert "verified "+mode in result.stdout
    if mode!="gas":
        assert ("historical representative sample" in result.stderr+result.stdout)==(warnings=="1")
