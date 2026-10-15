"""Native import of the declared two-layer block geometry; no solver tolerance changes."""
import json,os,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(os.environ.get('UGKP_CASES_ROOT',Path(__file__).resolve().parents[4].parent/'ugkp-thermal-cases'))/'test/tools/chmt_wall_suite'))
from generate import prepare
from run import execute

@pytest.mark.parametrize("wall",["lowRe","boundaryLayer_reactingSst"])
def test_generated_graded_empty_mesh_passes_actual_chmt_import(tmp_path,wall):
    checker=os.environ.get("CHMT_INPUT_CHECKER")
    if not checker:pytest.skip("requires actual OF10 CHMT importer")
    case=tmp_path/"case"
    spec=prepare("flatplate_moving",case,{"nx":8,"ny":6,"solid_ny":4,"grading":8.,"wall":wall})
    assert spec["end_time"]==.01
    result=execute(case,"check-input",input_checker=checker)
    assert result["input_import"]=="PASS",result

@pytest.mark.parametrize("fault",["normal_motion","unpaired_tangent","imported_mesh"])
def test_reject_geometry_outside_declared_construction_without_writes(tmp_path,fault):
    import shutil,subprocess,re
    if shutil.which("blockMesh") is None:pytest.skip("requires blockMesh")
    case=tmp_path/"invalid"
    spec=prepare("flatplate_moving",case,{"nx":4,"ny":4,"solid_ny":4})
    for command in spec["mesh_commands"]:
        subprocess.run([command[0],"-case",str(case)]+command[1:],check=True,capture_output=True)
    path=case/"constant/polyMesh/points"
    text=path.read_text()
    if fault!="imported_mesh":
        match=re.search(r"\n\s*\d+\s*\n\(\s*\n(.*?)\n\)",text,re.S)
        point=re.search(r"\(([^()]+)\)",match.group(1));values=list(map(float,point.group(1).split()))
        values[2 if fault=="normal_motion" else 0]+=1e-6
        begin=match.start(1)+point.start();end=match.start(1)+point.end()
        path.write_text(text[:begin]+"("+" ".join(map(str,values))+")"+text[end:])
    else:spec["generator"]="mss7_moving"
    before={str(p):p.read_bytes() for p in case.glob("constant/**/polyMesh/points")}
    from generated_extrusion import canonicalize
    with pytest.raises(ValueError):canonicalize(case,spec)
    assert before=={str(p):p.read_bytes() for p in case.glob("constant/**/polyMesh/points")}
    assert not (case/"mesh-construction-evidence").exists()
