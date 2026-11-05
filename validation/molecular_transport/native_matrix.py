#!/usr/bin/env python3
"""Real OpenFOAM/CUDA applications; every run owns a fresh directory.

Requires sourced OpenFOAM 10, blockMesh, and built gasUGKP/CHT/CHMT.
No solver mocks or substitutes. Inputs are synthetic verification fixtures.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time

HERE=Path(__file__).resolve().parent
HEADER=struct.Struct('<8sQIIIIIIQQQQ')
ENV=dict(os.environ,OMP_NUM_THREADS='4')
for key in ('GAS_UGKP_CUDA_STACK_BYTES','CHT_CUDA_STACK_BYTES','CHMT_CUDA_STACK_BYTES'):
    ENV.setdefault(key,'65536')

def run(command,log,expect_failure=False):
    start=time.perf_counter()
    with log.open('w') as output:
        result=subprocess.run(list(map(str,command)),env=ENV,stdout=output,stderr=subprocess.STDOUT)
    if expect_failure:
        assert result.returncode!=0, f'expected rejection: {log}'
        assert 'identity' in log.read_text() or 'const molecular transport' in log.read_text(), log
    else:
        assert result.returncode==0, f'failed ({result.returncode}): {log}'
    return time.perf_counter()-start

def entry(path,key,value):
    text,n=re.subn(r'\b'+key+r'\s+[^;]+;',key+' '+str(value)+';',path.read_text())
    assert n==1,(path,key,n)
    path.write_text(text)

def copy_input(source,target):
    target.mkdir()
    for name in ('0','constant','system'):
        shutil.copytree(source/name,target/name)

def copy_fixture(source,target):
    """Apply reviewable input overrides; solver code never sees this inheritance."""
    models=json.loads((HERE/'fixtures.json').read_text())
    target.mkdir()
    def overlay(name,parents):
        assert name not in parents,'cyclic fixture inheritance'
        description=models[name]
        if 'inherits' in description:overlay(description['inherits'],parents+(name,))
        for removed in description.get('remove',[]):
            path=target/removed
            assert path.resolve().is_relative_to(target.resolve())
            path.unlink()
        for directory in ('0','constant','system'):
            origin=HERE/'fixtures'/name/directory
            if origin.exists():shutil.copytree(origin,target/directory,dirs_exist_ok=True)
    overlay(source.name,())

def prepare(source,target,meta,constant=False):
    copy_fixture(source,target)
    if constant:(target/'constant/gasTransportProperties').unlink()
    if meta['solver']=='gasUGKP':
        control=target/'system/controlDict'
        end=float(re.search(r'endTime\s+([^;]+)',control.read_text())[1])
        # At least four accepted steps, with a common midpoint write boundary.
        entry(control,'deltaT',end/4);entry(control,'maxDeltaT',end/4)
        entry(control,'writeInterval',end/2)
    run(['blockMesh','-case',target],target/'mesh.log')
    solid=target/'system/blockMeshSolidDict'
    if solid.exists():run(['blockMesh','-case',target,'-region','solid','-dict',solid],target/'mesh-solid.log')
    if meta['wall_geometry']:
        patch=target/'constant/polyMesh/boundary';text=patch.read_text()
        for name in ('right','sides'):
            text=re.sub(r'('+name+r'\s*\{\s*type\s+)wall',r'\1patch',text)
        patch.write_text(text)
    if meta['three_dimensional_sides']:
        # A 3D thin extrusion with slip side faces; avoids the Empty importer.
        for patch in (target/'constant/polyMesh/boundary',target/'constant/solid/polyMesh/boundary'):
            patch.write_text(re.sub(r'\btype\s+empty\s*;','type patch;',patch.read_text()))

def times(case):
    return sorted((p for p in case.iterdir() if p.is_dir() and re.fullmatch(r'[0-9.eE+-]+',p.name) and float(p.name)>0),key=lambda p:float(p.name))

def compare_fields(reference,result,restart=False):
    errors={}
    for path in reference.rglob('*'):
        if not path.is_file() or path.suffix=='.dat':continue
        other=result/path.relative_to(reference)
        assert other.exists(),other
        text=path.read_text();actual=other.read_text()
        if 'internalField' not in text:continue
        numbers=lambda s:[float(x) for x in re.findall(r'(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?',s[s.index('internalField'):])]
        a,b=numbers(text),numbers(actual);assert len(a)==len(b),path
        error=max((abs(x-y)/max(1,abs(x),abs(y)) for x,y in zip(a,b)),default=0)
        errors[str(path.relative_to(reference))]=error
        # Empty-particle dMeanCell is not restored by the existing CHT bridge.
        if not restart or path.name in ('rho','rhoU','rhoE','U','p','T','k','omega','nut') or path.name.startswith(('Y_','rhoY_')):
            assert error<2e-12,(path,error)
    return errors

def terminal(case):
    return max((case/'chmtOutput').glob('checkpoint-*/state.chmt'),key=lambda p:int(p.parent.name.split('-')[-1]))

def compare_chmt(reference,result):
    a,b=terminal(reference),terminal(result)
    def payload(path):
        data=path.read_bytes();header=HEADER.unpack_from(data)
        begin=HEADER.size+header[-3]
        assert begin+header[-2]==len(data)
        return data[begin:],header[2:8]
    assert payload(a)==payload(b),'checkpoint physical state differs'
    step=a.parent.name.split('-')[-1]
    for name in ('gas','material','accepted','exchange'):
        filename=f'{name}-{step}.csv'
        assert (a.parent.parent/filename).read_bytes()==(b.parent.parent/filename).read_bytes(),filename
    return {'state_payload_bitwise_equal':True,'accepted_step':int(step)}

def transport_response(variable,constant,solver,require_change):
    if solver=='CHMT':
        def physical(case):
            data=terminal(case).read_bytes();header=HEADER.unpack_from(data)
            return data[HEADER.size+header[-3]:]
        changed=physical(variable)!=physical(constant)
        assert changed,'transport change did not reach CHMT physical state'
        return {'physical_state_changed':changed}
    left,right=times(variable)[-1],times(constant)[-1];response={}
    for name in ('rhoU','rhoE','T','omega','solid/T','gasConvectiveWallHeatFlux'):
        a,b=left/name,right/name
        if not a.exists():continue
        def numbers(path):
            text=path.read_text();text=text[text.index('internalField'):]
            return [float(x) for x in re.findall(r'(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?',text)]
        x,y=numbers(a),numbers(b);assert len(x)==len(y)
        response[name]=max((abs(v-w) for v,w in zip(x,y)),default=0)
    # Uniform/isothermal fixtures need not change; nontrivial wall/heat cases must.
    if require_change:assert max(response.values())>1e-12,'transport did not reach gas/wall/thermal fields'
    return response

def resume(case,destination,solver,binary,checkpoint=None,mutate=False):
    copy_input(case,destination)
    if solver=='CHMT':
        checkpoint=checkpoint or case/'chmtOutput/checkpoint-1'
        config=destination/'constant/chmtProperties'
        config.write_text(config.read_text()+f'\nrestartDirectory "{checkpoint}";\n')
    else:
        snapshots=times(case);assert len(snapshots)>=2,snapshots
        saved=snapshots[0];shutil.copytree(saved,destination/saved.name)
        entry(destination/'system/controlDict','startTime',saved.name)
    if mutate:
        config=destination/'constant/gasTransportProperties'
        text,n=re.subn(r'\bS\s+([^;]+);',lambda m:'S '+str(float(m[1])+1)+';',config.read_text(),count=1)
        assert n==1;config.write_text(text)
    run([binary,'-case',destination],destination/'run.log',mutate)
    if mutate:return {'changed_coefficients_rejected':True}
    return compare_chmt(case,destination) if solver=='CHMT' else compare_fields(times(case)[-1],times(destination)[-1],True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir',type=Path,required=True)
    parser.add_argument('--baseline-bin-dir',type=Path)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--only',nargs='*')
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args();work=args.work.resolve();work.mkdir(parents=True,exist_ok=False)
    models=json.loads((HERE/'fixtures.json').read_text());results={}
    for name,meta in models.items():
        if args.only and name not in args.only:continue
        solver=meta['solver'];binary=args.bin_dir.resolve()/solver;source=HERE/'fixtures'/name
        candidate=work/(name+'-sutherland');prepare(source,candidate,meta)
        elapsed=run([binary,'-case',candidate],candidate/'run.log')
        data={'sutherland_wall_seconds':elapsed}
        data['restart']=resume(candidate,work/(name+'-restart'),solver,binary)
        data['identity']=resume(candidate,work/(name+'-identity'),solver,binary,mutate=True)
        constant=work/(name+'-const');prepare(source,constant,meta,True)
        data['const_wall_seconds']=run([binary,'-case',constant],constant/'run.log')
        data['transport_response']=transport_response(candidate,constant,solver,name not in ('gas-wave','gas-pure','gas-single'))
        if args.baseline_bin_dir and meta['baseline']:
            before=work/(name+'-baseline');prepare(source,before,meta,True)
            data['baseline_wall_seconds']=run([args.baseline_bin_dir.resolve()/solver,'-case',before],before/'run.log')
            data['const_parity']=compare_chmt(before,constant) if solver=='CHMT' else compare_fields(times(before)[-1],times(constant)[-1])
            if solver=='CHMT':
                data['schema8_const_restart']=resume(constant,work/(name+'-old-restart'),solver,binary,before/'chmtOutput/checkpoint-1')
        if args.memcheck:
            checked=work/(name+'-memcheck');prepare(source,checked,meta)
            run(['compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','9',binary,'-case',checked],checked/'run.log')
            assert 'ERROR SUMMARY: 0 errors' in (checked/'run.log').read_text()
            data['memcheck']=True
        results[name]=data;(work/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,'PASS',flush=True)
if __name__=='__main__':main()
