from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess,sys
ROOT=SOURCE_ROOT
def test_tool_b1_event_and_error_protocol(tmp_path):
    result=subprocess.run([sys.executable,str((VALIDATION_ROOT / 'tools/tests/fixtures/shared_operators/tool_b1_protocol.py')),str(ROOT),str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
