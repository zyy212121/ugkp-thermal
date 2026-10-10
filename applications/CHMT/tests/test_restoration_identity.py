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
