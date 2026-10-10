#!/usr/bin/env python3
"""Generate exact species and compiled-build headers for this CHMT checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from source_fingerprint import manifest,encoded


def command_version(command):
    result=subprocess.run(command,text=True,capture_output=True,check=True)
    return (result.stdout+result.stderr).strip()


def validate_output_directory(app, output):
    app=Path(app).resolve(); output=Path(output).resolve(); repository=app.parents[1]
    inside_checkout=output==repository or repository in output.parents
    default=app/'.build'
    inside_default=output==default or default in output.parents
    if inside_checkout and not inside_default:
        raise ValueError('custom generated/build output must be outside the checkout or under CHMT/.build')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path); p.add_argument('--species-count',type=int,default=2,help='Shared UGKWP_GAS_SPECIES storage count; runtime model owns names/order')
    p.add_argument('--arch'); p.add_argument('--nvcc',default='nvcc'); p.add_argument('--cxx',default=os.environ.get('CXX','g++'))
    p.add_argument('--compile-only',action='store_true',help='Non-runnable frontend-object identity; never a CUDA build')
    args=p.parse_args()
    if args.species_count<=0: raise ValueError("species-count must be positive")
    foam_header=Path(os.environ.get('WM_PROJECT_DIR','/missing'))/'src/finiteVolume/lnInclude/fvCFD.H'
    if os.environ.get('WM_PROJECT_VERSION')!='10' or not foam_header.is_file():
        raise ValueError('source a Foundation OpenFOAM 10 environment first')
    app=Path(__file__).resolve().parents[1]
    validate_output_directory(app,args.output)
    data=manifest(app)
    host=command_version([args.cxx,'--version'])
    if args.compile_only: cuda='COMPILE_ONLY_NO_CUDA'; arch='COMPILE_ONLY'
    else:
        if not args.arch or not re.fullmatch(r'sm_[0-9]+',args.arch): raise ValueError('explicit CUDA architecture must have form sm_XX')
        cuda=command_version([args.nvcc,'--version']); arch=args.arch
    data.update({'artifact_kind':'COMPILE_ONLY' if args.compile_only else 'CHMT_BUILD_MANIFEST',
                 'species_identity':'runtime name/order hash','Ns':args.species_count,'precision':'FP64','cuda_arch':arch,'cuda_compiler':cuda,
                 'host_compiler':host,'openfoam_version':'10','openfoam_directory':os.environ['WM_PROJECT_DIR'],
                 'shared_adapter_sha256':hashlib.sha256((app/'gpu/SharedGasAdvance.cuh').read_bytes()).hexdigest()})
    data['build_id']=hashlib.sha256(encoded(data)).hexdigest()
    macros={'CHMT_BUILD_ID':data['build_id'],'CHMT_SOURCE_FINGERPRINT':data['source_fingerprint'],
            'CHMT_UPSTREAM_BASE':data['upstream_base'],'CHMT_SOLVER_COMMIT':data['solver_commit'],
            'CHMT_CUDA_COMPILER':cuda,'CHMT_HOST_COMPILER':host,'CHMT_CUDA_ARCH':arch,'CHMT_OPENFOAM_VERSION':'10',
            'CHMT_BUILD_MANIFEST_JSON':json.dumps(data,sort_keys=True,separators=(',',':'))}
    generated='#pragma once\n'+''.join('#define '+k+' '+json.dumps(v)+'\n' for k,v in macros.items())
    if args.compile_only:
        generated+='#ifndef CHMT_FRONTEND_COMPILE_ONLY\n#error "COMPILE_ONLY identity cannot build a CUDA/runtime executable"\n#endif\n'
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'CHMTBuildIdentity.H').write_text(generated)
    (args.output/'build.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    print(data['build_id'])
if __name__=='__main__':
    try: main()
    except (ValueError,OSError,subprocess.SubprocessError) as exc:
        print('CHMT build identity: '+str(exc),file=sys.stderr); sys.exit(2)
