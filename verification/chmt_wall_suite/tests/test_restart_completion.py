"""Restart completion must match the actual checkpoint, not assume step zero."""
import json,os,shutil
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import validate_completion

@pytest.fixture
def resumed(tmp_path):
    source=os.environ.get("CHMT_RESTART_REGRESSION_CASE")
    if not source:pytest.skip("requires captured native restart evidence")
    case=tmp_path/"resumed"
    shutil.copytree(source,case)
    return case,json.loads((case/"suite_case.json").read_text())

def test_actual_restart_completion(resumed):
    case,spec=resumed
    assert validate_completion(case,spec,solver_returncode=0)["end_marker"]=="CHMT accepted time"

@pytest.mark.parametrize("fault",["missing_step","wrong_start","missing_checkpoint","corrupt_checkpoint","wrong_initial_checkpoint","nonzero_exit","nonfinite_time"])
def test_restart_rejects_incomplete_or_mismatched_history(resumed,fault):
    case,spec=resumed
    if fault=="nonfinite_time":
        p=case/"chmtOutput/accepted-7.csv";lines=p.read_text().splitlines();row=lines[1].split(",");row[0]="nan";p.write_text(lines[0]+"\n"+",".join(row)+"\n")
    if fault=="missing_step":(case/"chmtOutput/accepted-7.csv").unlink()
    if fault=="wrong_start":
        p=case/"chmtOutput/accepted-5.csv";p.write_text(p.read_text().replace("0.0050000000000000001","0.004"))
    if fault=="missing_checkpoint":(case/"restart-from-5/state.chmt").unlink()
    if fault=="corrupt_checkpoint":
        p=case/"restart-from-5/state.chmt";b=p.read_bytes();p.write_bytes(b[:-1]+bytes([b[-1]^1]))
    if fault=="wrong_initial_checkpoint":
        shutil.copyfile(case/"chmtOutput/checkpoint-6/state.chmt",case/"chmtOutput/checkpoint-5/state.chmt")
    with pytest.raises((ValueError,OSError)):
        validate_completion(case,spec,solver_returncode=2 if fault=="nonzero_exit" else 0)
