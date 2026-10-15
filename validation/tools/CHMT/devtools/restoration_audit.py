#!/usr/bin/env python3
"""Classify reviewed CHMT restorations without re-reviewing unchanged bodies."""
from pathlib import Path as _ValidationPath
import sys as _validation_sys
_validation_sys.path.insert(0, str(next(p.parent for p in _ValidationPath(__file__).resolve().parents if p.name == "validation")))
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

APP=(SOURCE_ROOT / 'applications/CHMT')
EXCLUDED={'.build','.pytest_cache','__pycache__','lnInclude'}
def digest(data):return hashlib.sha256(data).hexdigest()
def current_path(name):
    if name == 'devtools/build_info.py':
        return SOURCE_ROOT / 'tool/CHMT/build_info.py'
    if name in ('devtools/restoration_audit.py', 'devtools/source_fingerprint.py'):
        return VALIDATION_ROOT / 'tools/CHMT' / name
    if name.startswith(('tests/', 'devtools/', 'verification/')):
        if name.endswith('.sh'):
            return VALIDATION_ROOT / 'tools/CHMT' / name
        if name.startswith(('verification/coupled_input/', 'verification/coupled_runtime/')) and not name.endswith('cuda_probe.cu'):
            return CASES_ROOT / 'test/tools/CHMT' / name.removeprefix('verification/')
        return VALIDATION_ROOT / 'solvers/CHMT' / name
    return APP / name


def current_files():
    roots = ((APP, ''),
             (VALIDATION_ROOT / 'solvers/CHMT', ''),
             (VALIDATION_ROOT / 'tools/CHMT', ''),
             (SOURCE_ROOT / 'tool/CHMT', 'devtools/'),
             (CASES_ROOT / 'test/tools/CHMT', 'verification/'))
    for root, prefix in roots:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or any(part in EXCLUDED for part in path.relative_to(root).parts):
                continue
            yield prefix + path.relative_to(root).as_posix(), path

def audit(verify_reference=False):
    manifest=json.loads((APP/'docs/restoration-source-manifest.json').read_text())
    unchanged=[];adapted=[];known=set();reference=manifest['reference_commit']
    for entry in manifest['files']:
        name=entry['path'];known.add(name);path=current_path(name)
        if verify_reference:
            source=subprocess.check_output(['git','show',reference+':applications/CHMT/'+name],cwd=APP)
            if digest(source)!=entry['source_sha256']:raise RuntimeError('reference source hash mismatch: '+name)
        current=digest(path.read_bytes())
        record={'path':name,'source_sha256':entry['source_sha256'],'current_sha256':current}
        (unchanged if current==entry['source_sha256'] else adapted).append(record)
    added=[]
    for name, path in current_files():
        if path.name.lower().startswith('readme'):raise RuntimeError('README is outside the authorized rebuild')
        if name not in known:added.append(name)
    for forbidden in ('gas/AleFlux.H','gas/GasKernels.cu','gas/SstKernels.cu','gpu/GasWindowImplementation.cuh'):
        if (APP/forbidden).exists():raise RuntimeError('private gas solver restored: '+forbidden)
    backend=APP/'gpu/Backend.cu'
    if backend.exists():
        source=backend.read_text()
        for required in ('common/GpuGasAdvance.cuh','advanceGasTrial<','common/operators/computeGasInternalFaceFluxKernel.cuh'):
            if required not in source:raise RuntimeError('backend does not instantiate shared transport: '+required)
        if 'aleRusanov(' in source:raise RuntimeError('private gas flux restored in backend')
    return {'reference_commit' :reference,'reference_hashes_verified':verify_reference,
            'unchanged_count':len(unchanged),'adapted_count':len(adapted),'new_count':len(added),
            'unchanged':unchanged,'adapted':adapted,'new':added}
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-reference',action='store_true')
    print(json.dumps(audit(parser.parse_args().verify_reference),indent=2))
