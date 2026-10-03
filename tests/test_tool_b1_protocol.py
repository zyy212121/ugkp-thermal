from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def test_tool_b1_event_and_error_protocol(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'tests/fixtures/shared_operators/tool_b1_protocol.py'),str(ROOT),str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
