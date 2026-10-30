from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import gzip
import json
import subprocess
import sys

APP = (SOURCE_ROOT / 'applications/CHMT')
TOOL = (VALIDATION_ROOT / 'tools/CHMT/compare_restart.py')


def _case(path):
    out = path/'chmtOutput'
    checkpoint = out/'checkpoint-3'
    checkpoint.mkdir(parents=True)
    raw = gzip.decompress(((VALIDATION_ROOT / 'solvers/CHMT/tests/fixtures/checkpoint_schema6/dry-3fe0000000000000.chmt.gz')).read_bytes())
    (checkpoint/'state.chmt').write_bytes(raw)
    for name in ('accepted','gas','material','exchange'):
        (out/(name+'-3.csv')).write_text('time\n0.5\n')
    return checkpoint/'state.chmt'


def test_equal_complete_native_payload_is_accepted(tmp_path):
    left,right = tmp_path/'full',tmp_path/'resumed'
    _case(left)
    _case(right)
    run = subprocess.run([sys.executable,str(TOOL),str(left),str(right)],capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result['state_payload_bitwise_equal']
    assert result['state_bytes'] > 0


def test_any_native_payload_bit_difference_fails_even_with_equal_csv(tmp_path):
    left,right = tmp_path/'full',tmp_path/'resumed'
    _case(left)
    target = _case(right)
    raw = bytearray(target.read_bytes())
    config_size = int.from_bytes(raw[48:56],'little')
    raw[72+config_size] ^= 1
    digest = 14695981039346656037
    for byte in raw[72:]:
        digest = ((digest ^ byte) * 1099511628211) & ((1 << 64) - 1)
    raw[64:72] = digest.to_bytes(8, 'little')
    target.write_bytes(raw)
    run = subprocess.run([sys.executable,str(TOOL),str(left),str(right)],capture_output=True,text=True)
    assert run.returncode != 0
    assert 'HostState payload differs' in run.stderr
