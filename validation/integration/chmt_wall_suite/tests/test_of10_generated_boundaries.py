"""Regression of generated OF10 wall-distance and slip-plane inputs."""
from pathlib import Path
import os
import sys,re
import pytest
sys.path.insert(0,str(Path(os.environ.get('UGKP_CASES_ROOT',Path(__file__).resolve().parents[4].parent/'ugkp-thermal-cases'))/'test/tools/chmt_wall_suite'))
from generate import prepare

@pytest.mark.parametrize("name",["flatplate_fixed","mss7_fixed"])
@pytest.mark.parametrize("family",["lowRe","wallFunction","boundaryLayer_reactingSst"])
def test_generated_sst_inputs_have_native_distance_and_symmetry(tmp_path,name,family):
    case=tmp_path/name
    prepare(name,case,{"nx":4,"ny":4,"wall":family})
    assert re.search(r"wallDist\s*\{\s*method\s+meshWave\s*;",(case/"system/fvSchemes").read_text())
    mesh=case/("system/blockMeshDict" if name=="flatplate_fixed" else "constant/polyMesh/boundary")
    patches=["top"] if name=="flatplate_fixed" else ["wedgeBack","wedgeFront"]
    for patch in patches:
        assert re.search(r"\b"+patch+r"\s*\{\s*type\s+symmetryPlane\s*;",mesh.read_text())
        for field in (case/"0").iterdir():
            if field.is_file():
                assert re.search(r"\b"+patch+r"\s*\{\s*type\s+symmetryPlane\s*;",field.read_text()),field.name
