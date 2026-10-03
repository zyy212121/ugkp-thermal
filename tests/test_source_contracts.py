from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def test_managed_source_and_field_contracts(tmp_path):
    if (ROOT/'applications/FSH').is_dir():
        commands=[[sys.executable,str(ROOT/'tests/fixtures/source_contract_regression.py'),str(ROOT)]]
    else:
        commands=[[sys.executable,str(ROOT/'tools/managed_mirrors.py')],
                  [sys.executable,str(ROOT/'tools/particle_field_contract.py'),'--cpu','--output',str(tmp_path/'fields')]]
    for command in commands:
        result=subprocess.run(command,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
