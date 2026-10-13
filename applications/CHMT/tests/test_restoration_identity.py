from pathlib import Path
import json
import subprocess
APP=Path(__file__).resolve().parents[1]

def test_selective_restoration_matches_historical_source():
    report=json.loads(subprocess.check_output(['python3',str(APP/'devtools/restoration_audit.py'),'--verify-reference'],text=True))
    assert report['reference_hashes_verified']
    assert report['unchanged_count']>0 and report['adapted_count']>0
    assert all(item['source_sha256']==item['current_sha256'] for item in report['unchanged'])
    assert all(item['source_sha256']!=item['current_sha256'] for item in report['adapted'])


def test_explicit_source_audit_supports_archive_with_unknown_git_identity(tmp_path):
    import importlib.util
    module_path=APP/'devtools/source_fingerprint.py'
    spec=importlib.util.spec_from_file_location('chmt_source_fingerprint_audit',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    app=tmp_path/'applications/CHMT';app.mkdir(parents=True)
    (app/'example.C').write_text('int example(){return 7;}\n')
    data=module.manifest(app)
    assert data['solver_commit']=='UNKNOWN'
    assert len(data['source_fingerprint'])==64
    assert data['git_identity_available'] is False
