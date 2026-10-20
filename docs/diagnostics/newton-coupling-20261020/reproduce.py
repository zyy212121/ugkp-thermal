#!/usr/bin/env python3
"""Reproduce fixed-input CPU counts/stages and optional GPU timing from Git refs.
No production files or cases are changed. Outputs go to the specified directory.
"""
import argparse,csv,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--gpu',action='store_true');p.add_argument('--nvcc',default='/usr/local/cuda-13.1/bin/nvcc');a=p.parse_args()
d=Path(__file__).resolve().parent;r=d.parents[2];a.work=a.work.resolve();a.work.mkdir(parents=True,exist_ok=True)
refs={'before':'957ab60727fc275bfe45df1b742067d342cc4493','after':'b161fa69c8e590244e5236519f0cbd92f7f91c5b'}
inputs=[r/'docs/diagnostics/bvp-coupling-20261020/fixed-input'/n for n in ['continuation-input.csv','of10-input.csv','of10-matched-input.csv']]
commands=[]
def run(cmd,**kw):
 commands.append([str(x)for x in cmd]);return subprocess.run(cmd,check=True,**kw)
for tag,ref in refs.items():
 root=a.work/tag;root.mkdir(exist_ok=True)
 names=subprocess.check_output(['git','ls-tree','-r','--name-only',ref,'common'],cwd=r,text=True).splitlines()
 for name in names:
  q=root/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(subprocess.check_output(['git','show',ref+':'+name],cwd=r))
 if a.gpu:
  run([a.nvcc,'-std=c++17','-O3','-arch=sm_89','--fmad=true','-I'+str(root),'-I'+str(r),'-I'+str(r/'validation/tests/gas_wall/fixtures'),str(d/'fixed-input-gpu.cu'),'-o',str(root/'gpu')])
 h=root/'common/gasWall/ReactingWallLayer.H';s=h.read_text()
 stats="""
#include <chrono>
namespace initprobe {
struct Stats{long point=0,interval=0,row=0,residual=0,newton=0,seed=0;double picard=-1,candidate=-1,initial=-1;int selected=0;double seedTime=0,residualTime=0,newtonTime=0,linearTime=0,publishTime=0;};inline Stats stats;
struct Timer{double&v;std::chrono::steady_clock::time_point start;Timer(double&x):v(x),start(std::chrono::steady_clock::now()){}~Timer(){v+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();}};
}
"""
 s=s.replace('#pragma once','#pragma once\n'+stats,1)
 for name,member in [('pointState','point'),('intervalFlux','interval'),('layerRow','row'),('allResidual','residual'),('newtonStep','newton')]:
  s,n=re.subn(r'(bool '+name+r'\([\s\S]*?\)\{)',lambda m:m[1]+'++::initprobe::stats.'+member+';',s,count=1);assert n==1,name
 s=s.replace('for(int iteration=0;iteration<200;++iteration){','for(int iteration=0;iteration<200;++iteration){++::initprobe::stats.seed;',1)
 for name,member in [('seedSstLayer','seedTime'),('allResidual','residualTime'),('newtonStep','newtonTime'),('solveLayerNewtonSystem','linearTime'),('publishLayer','publishTime')]:
  s,n=re.subn(r'(bool '+name+r'\([\s\S]*?\)\{)',lambda m:m[1]+'::initprobe::Timer stageTimer(::initprobe::stats.'+member+');',s,count=1);assert n==1,name
 h.write_text(s)
 for source,exe in [('fixed-input.cpp','cpu'),('blowing-cost.cpp','blowing')]:
  run(['g++','-std=c++17','-O3','-ffp-contract=off','-DINIT_POLICY=0','-I'+str(root),'-I'+str(r),str(d/source),'-o',str(root/exe)])
for j,tag in enumerate(['before','after','after','before','before','after']):
 for kind in ['cpu','blowing']+(['gpu']if a.gpu else []):
  cmd=[str(a.work/tag/kind)]+([]if kind=='blowing'else [str(p)for p in inputs]);out=run(cmd,capture_output=True,text=True).stdout;(a.work/f'{kind}-{j}-{tag}.csv').write_text(out)
  rows=list(csv.DictReader(out.splitlines()));key='final_residual'if kind=='cpu'else 'residual'if kind=='blowing'else 'max_residual'
  successes=sum(float(v[key])<=1.01e-8 for v in rows);print(kind,tag,'converged rows',successes,'/',len(rows),flush=True)
(a.work/'commands.json').write_text(json.dumps(commands,indent=2))
