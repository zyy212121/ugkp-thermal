#!/usr/bin/env python3
"""Offline reproduction; no source mutation, production run, or GPU execution."""
import argparse, hashlib, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('repository',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--cuda-compile',action='store_true');a=p.parse_args();root=a.repository.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
fixture=root/'validation/tests/fixtures/interface_composition';result=[]
def run(name,cmd,required=True,cwd=None):
 r=subprocess.run(list(map(str,cmd)),cwd=cwd,text=True,capture_output=True)
 (out/(name+'.log')).write_text('COMMAND '+repr(list(map(str,cmd)))+'\nEXIT '+str(r.returncode)+'\n'+r.stdout+r.stderr)
 result.append({'name':name,'command':list(map(str,cmd)),'exit':r.returncode,'stdout':r.stdout})
 if required and r.returncode:raise RuntimeError(name+' failed; inspect its log')
 return r
run('regression',['python3','-m','pytest','-q','validation/tests/test_interface_composition.py'],cwd=root)
run('diagnostic-sites',['python3','-m','pytest','-q','validation/tests/test_diagnostics_core.py::test_production_diagnostic_callsite_ids_are_unique','validation/tests/test_diagnostics_core.py::test_chmt_direct_api_compiles'],cwd=root)
# Independent header overlay; all baseline mutations remain under --output.
base=out/'baseline';(base/'applications/CHMT/ablation').mkdir(parents=True,exist_ok=True)
for x in (root/'applications/CHMT').iterdir():
 if x.name=='ablation':continue
 target=base/'applications/CHMT'/x.name
 if not target.exists():target.symlink_to(x,target_is_directory=x.is_dir())
for x in (root/'applications/CHMT/ablation').iterdir():
 if x.name in ('InterfaceMath.H','InterfaceComposition.H','InterfaceCompositionArithmetic.H'):continue
 target=base/'applications/CHMT/ablation'/x.name
 if not target.exists():target.symlink_to(x,target_is_directory=x.is_dir())
if not (base/'common').exists():(base/'common').symlink_to(root/'common',target_is_directory=True)
old=subprocess.check_output(['git','show','ea0c64d96fd041826ffa13b7bc2aa0acff4ca689:applications/CHMT/ablation/InterfaceMath.H'],cwd=root)
(base/'applications/CHMT/ablation/InterfaceMath.H').write_bytes(old)
strict=out/'manufactured20000.cpp';strict.write_text((fixture/'manufactured.cpp').read_text().replace('k<1000','k<20000'))
def build(name,ns,src,include):
 exe=out/name;run(name+'-build',['g++','-std=c++14','-O2','-ffp-contract=off','-DUGKWP_GAS_SPECIES='+str(ns),'-I'+str(include/'applications/CHMT'),'-I'+str(include/'common'),src,'-o',exe]);return exe
for ns in (2,10):
 run('strict'+str(ns),[build('strict'+str(ns),ns,strict,root)])
 run('baseline-red'+str(ns),[build('baseline-red'+str(ns),ns,fixture/'manufactured.cpp',base)],required=False)
 for label,include in [('baseline',base),('final',root)]:
  exe=build(label+'-benchmark'+str(ns),ns,fixture/'benchmark.cpp',include)
  for mode in (0,1,2):run(exe.name+'-mode'+str(mode),[exe,str(mode)])
 if a.cuda_compile:
  run('device'+str(ns)+'-build',['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','-arch=sm_89','--expt-relaxed-constexpr','--fmad=false','-Xptxas=-v','-DUGKWP_GAS_SPECIES='+str(ns),'-I'+str(root/'applications/CHMT'),'-I'+str(root/'common'),fixture/'device.cu','-o',out/('device'+str(ns))])
(out/'summary.json').write_text(json.dumps(result,indent=2))
(out/'sources.json').write_text(json.dumps({str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [root/'applications/CHMT/ablation/InterfaceMath.H',root/'applications/CHMT/ablation/InterfaceComposition.H',root/'applications/CHMT/ablation/InterfaceCompositionArithmetic.H']},indent=2))
print('Reproduced',len(result),'commands; CUDA binaries, if requested, were compiled only.')
