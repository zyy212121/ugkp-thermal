"""Independent root-boundary and arithmetic checks for the actual new solver."""
import hashlib
import json
import os
import pathlib
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT

REVIEW_ROOT=pathlib.Path(os.environ.get("UGKP_COMPOSITION_REVIEW_ROOT",str(SOURCE_ROOT)))
HEADERS=["applications/CHMT/ablation/InterfaceMath.H","applications/CHMT/ablation/InterfaceComposition.H","applications/CHMT/ablation/InterfaceCompositionArithmetic.H"]

@pytest.fixture(scope="module")
def executables(tmp_path_factory):
    if not (REVIEW_ROOT/HEADERS[1]).exists():
        pytest.skip("new composition solver is not integrated in this source tree")
    out=tmp_path_factory.mktemp("composition-independent");result={}
    hashes={name:hashlib.sha256((REVIEW_ROOT/name).read_bytes()).hexdigest() for name in HEADERS}
    for ns in (2,3,10):
        exe=out/f"probe-ns{ns}"
        cmd=["g++","-std=c++17","-O2","-ffp-contract=off",f"-DUGKWP_GAS_SPECIES={ns}","-I"+str(REVIEW_ROOT/"applications/CHMT"),"-I"+str(REVIEW_ROOT/"common"),str(SOURCE_ROOT/"validation/tests/fixtures/composition_independent_review/probe.cpp"),"-o",str(exe)]
        proc=subprocess.run(cmd,text=True,capture_output=True)
        assert proc.returncode==0,proc.stdout+proc.stderr
        assert hashes=={name:hashlib.sha256((REVIEW_ROOT/name).read_bytes()).hexdigest() for name in HEADERS},"review source changed while compiling"
        result[ns]=exe
    (out/"source-sha256.json").write_text(json.dumps({"source_root":str(REVIEW_ROOT),"sha256":hashes},indent=2))
    return result

@pytest.mark.parametrize("ns,mode",[(2,"boundary"),(10,"boundary"),(3,"endpoint"),(2,"near"),(10,"near"),(2,"arithmetic")])
def test_independent_composition_boundaries_and_contracts(executables,ns,mode):
    proc=subprocess.run([str(executables[ns]),mode],text=True,capture_output=True)
    assert proc.returncode==0,proc.stdout+proc.stderr
