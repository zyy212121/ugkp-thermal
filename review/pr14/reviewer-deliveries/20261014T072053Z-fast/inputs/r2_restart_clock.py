import json,shutil,collections
from r2_runner import *
from r2_native import compare_dirs
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n正式thermal原生restart全bytes发现：gas/exchange/checkpoint与full不同，accepted/material相同。先保留red；用802原binary从同本轮checkpoint1复验同输入，判定是否这轮Co改动回归。跨版本对照不扩大容差。SSPRK2/3既有D201formal observer binary再运行相同输入，记录真实packet_dt/微步dt阶段权重，所有原输出应全bytes相同；stageCSV聚合stage=1本身不能证明Euler。各300s。\n')
original=ROOT/'work/cases/chmt/thermal-formal-resume';case=ROOT/'work/cases/chmt/thermal-old802-same-resume';shutil.copytree(D/'inputs/native/chmt/thermal-formal-resume',case)
for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(original/rel,case/rel)
b=OLDROOT/'work/install/ns2/bin/CHMT';t=run('THERMAL-802-SAME-RESTART',cmd(b,'-case',case),[5,9,12,14,26],expected='Unmodified802sameactualcheckpoint1/input; comparetoformalrestart and oldfull without tolerances.',binary=b,variant='old-production')
cmp=compare_dirs(original/'chmtOutput',case/'chmtOutput');out=D/'artifacts/chmt/old802-vs-formal-restart.json';out.write_text(json.dumps(cmp,indent=2)+'\n');assertion(t,cmp['all_files_byte_equal'],'Formalvs802actual samecheckpointrestart allbytesexact '+str(cmp['all_files_byte_equal']),artifacts=[str(out.relative_to(D))]);print('OLDRESTART',cmp['all_files_byte_equal'],flush=True)
for name in ['SST-SSPRK2','SST-SSPRK3']:
 original=ROOT/'work/cases/extra'/name;case=ROOT/'work/clock-cases'/name;shutil.copytree(D/'inputs/native/extra'/name,case)
 for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(original/rel,case/rel)
 b=ROOT/'work/install/clock-formal/bin/CHMT';t=run('ACTUAL-RK-WEIGHTS-'+name,cmd(b,'-case',case),[8,9,14,26,29],expected='Realnative actualRKstagepacket_dt weights from existingD201observer; noadditionalGPUcopies and alloriginalbytesexact.',binary=b,variant='diagnostic-clock-formal')
 raw=(D/t['log']).read_text();microdt={};weights=[]
 for line in raw.splitlines():
  if not line.startswith('F003,'):continue
  x=line.split(',');key=tuple(x[1:4]);val=float.fromhex(x[-1])
  if x[4]=='begin' and x[6]=='micro_dt':microdt[key]=val
  if x[4]=='stage' and x[6]=='packet_dt':weights.append({'window':x[1],'micro':x[3],'packet_dt_hex':x[7],'micro_dt_hex':microdt[key].hex(),'ratio':val/microdt[key]})
 cmp=compare_dirs(original/'chmtOutput',case/'chmtOutput');p=D/'artifacts/extra'/name/'actual-stage-weights.json';p.write_text(json.dumps({'rows':weights,'ratio_counts':dict(collections.Counter(str(x['ratio']) for x in weights)),'alloutput_comparison':cmp},indent=2)+'\n')
 expect=[.5,.5] if name.endswith('2') else [1/6,1/6,2/3];good=len(weights)==100*len(expect) and all(abs(x['ratio']-expect[i%len(expect)])<=2e-16 for i,x in enumerate(weights))
 assertion(t,cmp['all_files_byte_equal'] and good,'Actual'+str(len(weights))+'stages actual'+str(expect)+' and outputbytesexact='+str(cmp['all_files_byte_equal']),artifacts=[str(p.relative_to(D))]);print('RK',name,len(weights),good,flush=True)
