"""Test-only synthetic report data; these tests provide no native CUDA evidence."""
import hashlib
import json
import subprocess
import sys

import pytest

from validation.tools.paths import VALIDATION_ROOT


@pytest.mark.parametrize('optional_checks', [False, True])
def test_native_report_uses_supplied_paths_and_records_evidence(tmp_path, optional_checks):
    script = VALIDATION_ROOT / 'tools/CHMT/devtools/check_native_coupled.sh'
    marker = '''python3 - "$work" "$scenario" "$CHMT_BUILD/CHMT" <<'PY'\n'''
    program = script.read_text().split(marker, 1)[1].split('\nPY', 1)[0]
    work = tmp_path / 'synthetic evidence'
    work.mkdir()
    binary = tmp_path / 'synthetic binary'
    binary.write_bytes(b'TEST ONLY: not a solver executable\x00\xff')
    build = {'artifact_kind': 'TEST_ONLY_SYNTHETIC', 'Ns': 2}
    independent = {'test_only': True, 'check': 'independent'}
    restart = {'test_only': True, 'check': 'restart'}
    records = {'build.json': build,
               'full/independent-check.json': independent,
               'restart-check.json': restart}
    if optional_checks:
        records['frozen-companion/independent-check.json'] = {'test_only': True}
        for check in ('diffusion', 'chemistry', 'refinement'):
            records[check + '-check.json'] = {'test_only': True, 'check': check}
    for relative, record in records.items():
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
    probe = 'TEST ONLY: synthetic probe text\n'
    (work / 'native-cuda-probe.log').write_text(probe)
    result = subprocess.run([sys.executable, '-', str(work), 'thermal', str(binary)],
                            input=program, capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    report_path = work / 'native-acceptance.json'
    expected = {'evidence': 'native-cuda-coupled-runtime', 'scenario': 'thermal',
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'build': build, 'cuda_probe': probe,
                'independent_checks': independent, 'restart': restart}
    if optional_checks:
        expected['same_binary_frozen_companion'] = {'test_only': True}
        for check in ('diffusion', 'chemistry', 'refinement'):
            expected[check] = records[check + '-check.json']
    assert json.loads(report_path.read_text()) == expected
    assert str(report_path) in result.stdout
    assert result.stdout.startswith('PASS: native CUDA coupled thermal evolution')
