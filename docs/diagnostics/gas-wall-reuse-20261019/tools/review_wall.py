#!/usr/bin/env python3
"""Differential wall/thermodynamic review probes; writes only to --out."""
import argparse, json, pathlib, subprocess, os
p=argparse.ArgumentParser()
p.add_argument('--baseline',required=True,type=pathlib.Path)
p.add_argument('--source',required=True,type=pathlib.Path)
p.add_argument('--out',required=True,type=pathlib.Path)
p.add_argument('--sanitizers',action='store_true')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
fixtures=pathlib.Path(__file__).resolve().parent;commands=[]
def run(cmd,name):
    q=subprocess.run(list(map(str,cmd)),capture_output=True,text=True,env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0'})
    (a.out/(name+'.log')).write_text(q.stdout+q.stderr)
    commands.append({'command':list(map(str,cmd)),'exit':q.returncode})
    (a.out/'commands.json').write_text(json.dumps(commands,indent=2));q.check_returncode()
    return q.stdout
outputs={}
for name,source in [('baseline',a.baseline),('candidate',a.source)]:
    exe=a.out/name
    run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(source),'-I'+str(a.source),fixtures/'wall_differential.cpp','-o',exe],name+'-build')
    outputs[name]=run([exe],name)
assert outputs['baseline']==outputs['candidate']
print('Wall cases:',len(outputs['baseline'].splitlines()),'byte-identical')
for name in ['cph_differential','caloric_repeat']:
    exe=a.out/name
    run(['g++','-std=c++17','-O2','-I'+str(a.source),'-I'+str(a.source/'common'),fixtures/(name+'.cpp'),'-o',exe],name+'-build')
    print(run([exe],name),end='')
if a.sanitizers:
    exe=a.out/'wall-sanitized'
    run(['g++','-std=c++17','-O1','-g','-ffp-contract=off','-fsanitize=address,undefined','-fno-omit-frame-pointer','-no-pie','-I'+str(a.source),fixtures/'wall_differential.cpp','-o',exe],'sanitized-build')
    assert run([exe],'sanitized')==outputs['candidate']
    print('ASan/UBSan passed; output byte-identical')
