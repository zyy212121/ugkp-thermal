"""Explicit bundle execution uses the actual program, independent of audit hashes."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = (VALIDATION_ROOT / 'tools/run_prebuilt_low_level.py')


@pytest.fixture
def bundle(tmp_path):
    marker = tmp_path/'marker'
    exe = tmp_path/'program'
    exe.write_text('#!/bin/sh\necho executed > "'+str(marker)+'"\necho actual-output\nexit "${1:-0}"\n')
    exe.chmod(0o755)
    return tmp_path, marker


def write_manifest(bundle, **changes):
    root, _ = bundle
    row = {'test':'sample','executable':'program','arguments':[['0']], 'expected_stdout':'actual-output', **changes}
    (root/'deferred-run.json').write_text(json.dumps({'programs':[row]}))


def launch(bundle, run=True):
    return subprocess.run([sys.executable, str(SCRIPT), '--bundle', str(bundle[0]), *(['--run'] if run else [])], capture_output=True, text=True)


@pytest.mark.parametrize('hashes', [False, True])
def test_explicit_run_ignores_missing_or_mismatched_audit_identity(bundle, hashes):
    write_manifest(bundle, **({'sha256':'different','build_id':'different'} if hashes else {}))
    if hashes:
        path=bundle[0]/'deferred-run.json'; data=json.loads(path.read_text())
        data['source_sha256']={'nonexistent-source':'different'};path.write_text(json.dumps(data))
    result=launch(bundle)
    assert result.returncode == 0, result.stdout + result.stderr
    assert bundle[1].is_file()


def test_inspection_mode_never_executes_program(bundle):
    write_manifest(bundle)
    result=launch(bundle,False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not bundle[1].exists()


@pytest.mark.parametrize('fault', ['missing','nonexec','nonzero','json','outside'])
def test_real_manifest_or_program_failures_remain_failures(bundle, fault):
    write_manifest(bundle)
    if fault == 'missing': (bundle[0]/'program').unlink()
    elif fault == 'nonexec': (bundle[0]/'program').chmod(0o644)
    elif fault == 'nonzero': write_manifest(bundle,arguments=[['23']])
    elif fault == 'json': (bundle[0]/'deferred-run.json').write_text('{broken')
    else: write_manifest(bundle,executable='../outside')
    result=launch(bundle)
    assert result.returncode != 0
    assert bundle[1].exists() == (fault == 'nonzero')


def test_stdout_acceptance_reports_fail_after_actual_execution(bundle):
    write_manifest(bundle,expected_stdout='unmatched-output')
    result=launch(bundle)
    assert result.returncode == 1 and 'FAIL' in result.stdout
    assert bundle[1].is_file()
    results=json.loads((bundle[0]/'gpu-correctness/results.json').read_text())
    assert results[0]['exit'] == 0 and results[0]['passed'] is False
