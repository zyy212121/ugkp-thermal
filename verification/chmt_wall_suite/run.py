#!/usr/bin/env python3
"""Mesh, import-check, or execute a generated native case with explicit evidence."""
import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from foam import fingerprint
from metrics import compare,final_directory,chmt_endpoints

def _save(case,status):
    states=[status.get(k) for k in ('mesh','input_import','native_execution','cpu_reference','validation')]
    status['outcome']='FAIL' if 'FAIL' in states else (status['validation'] if status.get('native_execution')=='COMPLETED' else 'NOT_RUN')
    (Path(case)/'run_status.json').write_text(json.dumps(status,indent=2,sort_keys=True)+'\n')

def cuda_probe():
    names=[ctypes.util.find_library('cudart'),'libcudart.so']
    if os.environ.get('CUDA_HOME'):names.insert(0,str(Path(os.environ['CUDA_HOME'])/'lib64/libcudart.so'))
    errors=[]
    for name in names:
        if not name:continue
        try:
            runtime=ctypes.CDLL(name);count=ctypes.c_int();code=runtime.cudaGetDeviceCount(ctypes.byref(count))
            if code or count.value<1:errors.append(f'{name}: cudaGetDeviceCount={code}, devices={count.value}');continue
            code=runtime.cudaFree(ctypes.c_void_p(0))
            if code:errors.append(f'{name}: CUDA context initialization={code}');continue
            return {'status':'AVAILABLE','device_count':count.value,'runtime':name,'context_initialized':True}
        except (OSError,AttributeError) as e:errors.append(str(e))
    return {'status':'UNAVAILABLE','reason':'; '.join(errors) or 'CUDA runtime not found'}

def validate_completion(case,spec):
    case=Path(case);log=(case/'log.solver').read_text()
    if spec['solver']=='CHMT':
        if 'CHMT accepted time=' not in log:raise ValueError('no accepted native CHMT step in solver log')
        chmt_endpoints(case,spec)
    else:
        if not re.search(r'^End\s*$',log,re.M):raise ValueError('native solver did not complete')
        if spec['generator'] not in ('couette','sod'):
            found=re.search(r'Shared gas model:\s+api=\d+\s+Ns=(\d+)\s+mode=(\d+)',log)
            expected_mode=2 if spec['generator'] in ('reactor','reacting_wave') else 1
            if not found or int(found.group(1))!=spec['species_count'] or int(found.group(2))!=expected_mode:raise ValueError('shared mixture backend identity missing or incompatible')
        final_directory(case,spec['end_time'])
    return True

def execute(case,mode='mesh',solver=None,input_checker=None,*,allow_solver_fallback=True):
    case=Path(case).resolve();spec=json.loads((case/'suite_case.json').read_text())
    status=dict(case_id=spec['id'],source_commit=spec['source_commit'],generated_inputs_sha256=spec['inputs_sha256'],mode=mode,mesh='NOT_RUN',input_import='NOT_RUN',native_execution='NOT_RUN',cpu_reference='NOT_RUN',validation='NOT_RUN',commands=[],command_measurements=[],timing_scope='Monotonic process wall time includes startup and I/O; not isolated GPU kernel timing.',reason='Preparation only')
    def command(args,name):
        status['commands'].append(args);_save(case,status)
        start=time.monotonic()
        with (case/name).open('w') as f:result=subprocess.run(args,cwd=case,stdout=f,stderr=subprocess.STDOUT)
        status['command_measurements'].append(dict(log=name,wall_seconds=time.monotonic()-start,returncode=result.returncode))
        _save(case,status)
        if result.returncode:raise RuntimeError('command failed, see '+name+': '+' '.join(args))
        return (case/name).read_text()
    try:
        if mode not in ('mesh','check-input','execute'):raise ValueError('invalid run mode')
        required=['checkMesh']+(['blockMesh'] if spec['mesh_commands'] else [])
        missing=[exe for exe in required if shutil.which(exe) is None]
        if missing:
            status['reason']='Unavailable OpenFOAM commands: '+', '.join(missing);_save(case,status);return status
        status['mesh']='RUNNING'
        for i,args in enumerate(spec['mesh_commands']):command([args[0],'-case',str(case)]+args[1:],f'log.mesh.{i}')
        for region in ['', 'solid'] if spec['solver']=='CHMT' else ['']:
            text=command(['checkMesh','-case',str(case)]+(['-region',region] if region else []),'log.checkMesh'+('.'+region if region else ''))
            if 'Mesh OK.' not in text:raise RuntimeError('checkMesh did not report Mesh OK')
        status['mesh']='PASS'
        for i,args in enumerate(spec['initialization']):command([args[0],'-case',str(case)]+args[1:],f'log.initialize.{i}')
        status['execution_inputs_sha256']=fingerprint(case)
        if mode=='check-input':
            if spec['solver']!='CHMT':
                if input_checker and (case/'constant/gasModelProperties').exists():
                    status['input_import']='RUNNING';text=command([str(Path(input_checker).resolve()),str(case)],'log.input-check')
                    if 'PRODUCTION_GAS_MODEL_IMPORT_ONLY' not in text:raise RuntimeError('gas model parser did not confirm metadata')
                    status['input_import']='PASS';status['reason']='Production gas thermo/mechanism parser imported metadata; this is not whole-solver or wall-runtime validation.'
                else:
                    if spec['generator'] in ('couette','sod') and not (case/'constant/gasModelProperties').exists():
                        status['input_import']='NOT_APPLICABLE';status['reason']='Legacy single-gas case remains native-executable; shared-mixture metadata import is not applicable. Mesh evidence only.'
                    else:
                        status['input_import']='NOT_RUN';status['reason']='Shared gas metadata importer was not provided; this case is supported but the import check was not run.'
            elif not input_checker or not Path(input_checker).is_file():
                status['reason']='Set --input-checker to the real CHMT-input-check built for the matching species count.'
            else:
                status['input_import']='RUNNING';text=command([str(Path(input_checker).resolve()),'-case',str(case),'-check-input'],'log.input-check')
                if 'CHMT coupled input valid' not in text:raise RuntimeError('input checker did not confirm imported case')
                status['input_import']='PASS';status['reason']='Actual OpenFOAM mesh and CHMT input import passed; native evolution NOT_RUN.'
        elif mode=='execute':
            if (case/'chmtOutput').exists() or any(p.is_dir() and _positive_time(p.name) for p in case.iterdir()):raise RuntimeError('refusing existing evolved output; generate a fresh case')
            if not solver and not allow_solver_fallback:
                status['reason']=f"Native executable not specified for {spec['solver']} Ns={spec['species_count']}; species-specific suite runs do not use PATH fallback."
                _save(case,status);return status
            gpu=cuda_probe();status['cuda_probe']=gpu
            if gpu['status']!='AVAILABLE':status['reason']='Native execution NOT_RUN: '+gpu['reason'];_save(case,status);return status
            path=Path(solver).resolve() if solver else Path(shutil.which(spec['solver']) or '/nonexistent')
            if not path.is_file():status['reason']='Native executable unavailable: '+spec['solver'];_save(case,status);return status
            status['solver_path']=str(path);status['solver_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            if spec['solver']=='CHMT':
                text=command([str(path),'-build-info'],'log.build-info');match=re.search(r'\{.*\}',text,re.S)
                info=json.loads(match.group(0)) if match else {}
                if info.get('artifact_kind')!='CHMT_BUILD_MANIFEST' or info.get('Ns')!=spec['species_count']:raise RuntimeError('refusing nonnative or species-incompatible CHMT executable')
                status['build_manifest']=info
            status['native_execution']='RUNNING';command([str(path),'-case',str(case)],'log.solver')
            # Existing reactor checker consumes its conventional actual solver log.
            if spec['generator']=='reactor':shutil.copyfile(case/'log.solver',case/'log.gasUGKP')
            validate_completion(case,spec);status['native_execution']='COMPLETED';status['cpu_reference']='RUNNING'
            report=compare(case);(case/'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
            status['cpu_reference']='COMPLETED';status['validation']=report['acceptance'];status['reason']='Metrics read actual completed native outputs; report-only cases have no physical PASS claim.'
        else:status['reason']='Actual OpenFOAM mesh passed; solver evolution NOT_RUN.'
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        if status['mesh']=='RUNNING':status['mesh']='FAIL'
        if status['input_import']=='RUNNING':status['input_import']='FAIL'
        if status['native_execution']=='RUNNING':status['native_execution']='FAIL'
        if status['cpu_reference']=='RUNNING':status['cpu_reference']='FAIL'
        status['reason']=str(error)
        status['validation']='FAIL'
    _save(case,status);return status

def _positive_time(name):
    try:return float(name)>0
    except ValueError:return False

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',type=Path);p.add_argument('--mode',choices=['mesh','check-input','execute'],default='mesh');p.add_argument('--solver');p.add_argument('--input-checker');p.add_argument('--required',action='store_true',help='Missing requested evidence exits 3; never counts as PASS');a=p.parse_args()
    result=execute(a.case,a.mode,a.solver,a.input_checker);print(json.dumps(result,indent=2,sort_keys=True))
    if 'FAIL' in [result['mesh'],result['input_import'],result['native_execution'],result['validation']]:raise SystemExit(1)
    requested='native_execution' if a.mode=='execute' else 'input_import' if a.mode=='check-input' else 'mesh'
    if result[requested] in ('NOT_RUN','UNSUPPORTED'):raise SystemExit(3)
if __name__=='__main__':main()
