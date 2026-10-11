#!/usr/bin/env python3
"""Classify reviewed CHMT restorations without re-reviewing unchanged bodies."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

APP=Path(__file__).resolve().parents[1]
EXCLUDED={'.build','__pycache__','lnInclude'}
def digest(data):return hashlib.sha256(data).hexdigest()
def audit(verify_reference=False):
    manifest=json.loads((APP/'docs/restoration-source-manifest.json').read_text())
    unchanged=[];adapted=[];known=set();reference=manifest['reference_commit']
    for entry in manifest['files']:
        name=entry['path'];known.add(name);path=APP/name
        if verify_reference:
            source=subprocess.check_output(['git','show',reference+':applications/CHMT/'+name],cwd=APP)
            if digest(source)!=entry['source_sha256']:raise RuntimeError('reference source hash mismatch: '+name)
        current=digest(path.read_bytes())
        record={'path':name,'source_sha256':entry['source_sha256'],'current_sha256':current}
        (unchanged if current==entry['source_sha256'] else adapted).append(record)
    added=[]
    for path in sorted(APP.rglob('*')):
        if not path.is_file() or any(part in EXCLUDED for part in path.relative_to(APP).parts):continue
        if path.name.lower().startswith('readme'):raise RuntimeError('README is outside the authorized rebuild')
        if str(path.relative_to(APP)) not in known:added.append(str(path.relative_to(APP)))
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
