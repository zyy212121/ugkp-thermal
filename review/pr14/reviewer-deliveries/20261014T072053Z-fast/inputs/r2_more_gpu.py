import re,json,shutil,csv,collections
from r2_runner import *
from r2_native import chmt,compare_dirs
amend('CHMT-chemistry-full-initcheck',verdict='BLOCKED',observations='Actual original full Ns10 initcheck exceeded external300s; onlyinitialcheckpoint written, no final sanitizer summary. Preservepartial outputs/checker finaltime FAIL; no numerical acceptance conclusion. Formal uninstrumented fullcase alreadyPASS.')
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nNs10完整chemistry-full initcheck实际300s外部超时，缺少完整sanitizer结尾，标BLOCKED；原5秒chemistry-control输入保持原样另做initcheck作为Ns10受影响Co路径有限证据，不把它冒充full原input sanitizer通过。正式thermal中点checkpoint原生恢复复验：从本轮formal完整100步checkpoint-1恢复至同原end，检查所有terminal CSV/checkpoint bytes；不沿用旧source断点冒充本轮restart。外部300s。\n')
for ns in [2,10]:
 b=ROOT/'work/fixtures'/f'courant_fresh_{ns}';src=D/'inputs/fixtures/courant_fresh_data.cu'
 t=run('CO-FRESH-'+str(ns),cmd(b),[9,12,14,26,28],expected='Actualsameowner changingCo/status/NaNpayload bitexact uncachedGPUcontrol; capacitycycles,secondowner/default/nonblocking streams.',inputs=[src],binary=b)
 raw=(D/t['log']).read_text();assertion(t,t['exit_code']==0 and raw.count('FRESH_DATA ')==168,'Actual168 fresharray/status comparisons in Ns'+str(ns)+'; originalunchangeddata suite also preserved; twoowners interleaved, no hostthreadconcurrency claim.')
b=ROOT/'work/fixtures/courant_fresh_10';t=run('CO-FRESH-10-INITCHECK',cmd('compute-sanitizer','--tool','initcheck','--error-exitcode','86',b),[9,12,14,26,28],expected='Allsameactual Co tail/freshdata cases under initcheck; no sourcechanges.',binary=b,kind='SANITIZER');raw=(D/t['log']).read_text();assertion(t,'ERROR SUMMARY: 0 errors' in raw,raw[-1800:])
lib=ROOT/'work/actual-chemistry-volume-observer-v2.so'
for name in ['SST-SSPRK2','SST-SSPRK3','moving-chemistry']:
 original=ROOT/'work/cases/extra'/name;case=ROOT/'work/volume-cases'/(name+'-S206');shutil.copytree(D/'inputs/native/extra'/name,case)
 for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(original/rel,case/rel)
 b=ROOT/('work/install/ns10/bin/CHMT' if name=='moving-chemistry' else 'work/install/ns2/bin/CHMT')
 t=run('ACTUAL-KERNELS-'+name+'-S206',cmd('env','LD_PRELOAD='+str(lib),b,'-case',case),[8,10,14,26,29],expected='ActualCUDA13kernelhandles: chemicaldt/old/newvolumeargs andRKblendweights; allunobservedCSV/checkpoint/geometry bytesunchanged.',inputs=[D/'inputs/actual-chemistry-volume-observer-v2.cpp'],binary=b,variant='diagnostic-volume-S206',kind='DIAGNOSTIC')
 raw=(D/t['log']).read_text();rows=[x.split(',') for x in raw.splitlines() if x.startswith('CHEM_VOLUME,')];values=[{'sequence':int(x[1]),'dt_hex':x[2],'native_refinement':int(x[3]),'pointer':x[4],'copy_code':int(x[5]),'volume_hex':x[6:]} for x in rows]
 blends=[x.split(',')[1:] for x in raw.splitlines() if x.startswith('ACTUAL_RK_BLEND,')];cmp=compare_dirs(original/'chmtOutput',case/'chmtOutput');payload={'volume_inputs':values,'blends':blends,'blend_frequencies':dict(collections.Counter('/'.join(x) for x in blends)),'alloutput_comparison':cmp,'actual_kernel_total':[x for x in raw.splitlines() if x.startswith('ACTUAL_KERNEL_TOTAL,')]};p=D/'artifacts/extra'/name/'actual-kernel-arguments-S206.json';p.write_text(json.dumps(payload,indent=2)+'\n')
 expect=bool(values) and all(x['copy_code']==0 for x in values) if name=='moving-chemistry' else bool(blends)
 assertion(t,t['exit_code']==0 and cmp['all_files_byte_equal'] and expect,'Actualchemistrylaunches='+str(len(values))+' RKblends='+str(len(blends))+' outputbytesexact='+str(cmp['all_files_byte_equal']),artifacts=[str(p.relative_to(D))]);print(name,payload['actual_kernel_total'],payload['blend_frequencies'],len(values),flush=True)
case=ROOT/'work/cases/chmt/thermal-formal-resume';shutil.copytree(OLD/'inputs/native/chmt/thermal-full',case)
for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(ROOT/'work/cases/chmt/thermal-full'/rel,case/rel)
q=case/'constant/chmtProperties';q.write_text(q.read_text()+'\nrestartDirectory "'+str(ROOT/'work/cases/chmt/thermal-full/chmtOutput/checkpoint-1')+'";\n')
inp=D/'inputs/native/chmt/thermal-formal-resume';shutil.copytree(case,inp,ignore=shutil.ignore_patterns('polyMesh'))
b=ROOT/'work/install/ns2/bin/CHMT';t=run('THERMAL-FORMAL-RESTART',cmd(b,'-case',case),[5,9,12,14,26],expected='Formalnative owner reconstructed after midpointactualcheckpoint; allterminal writtenbytes equalformalfull.',inputs=[p for p in inp.rglob('*') if p.is_file()],binary=b)
shutil.copytree(case/'chmtOutput',D/'artifacts/chmt/thermal-formal-resume');expected=ROOT/'work/thermal-terminal-reference';expected.mkdir()
for name in ['accepted-2.csv','gas-2.csv','material-2.csv','exchange-2.csv','checkpoint-2']:
 q=ROOT/'work/cases/chmt/thermal-full/chmtOutput'/name
 if q.is_dir():shutil.copytree(q,expected/name)
 else:shutil.copy2(q,expected/name)
actual=ROOT/'work/thermal-terminal-resume';actual.mkdir()
for name in [p.name for p in expected.iterdir()]:
 q=case/'chmtOutput'/name
 if q.is_dir():shutil.copytree(q,actual/name)
 elif q.exists():shutil.copy2(q,actual/name)
cmp=compare_dirs(expected,actual);p=D/'artifacts/chmt/formal-restart-terminal-comparison.json';p.write_text(json.dumps(cmp,indent=2)+'\n');assertion(t,cmp['all_files_byte_equal'],'AllterminalCSV/checkpoint bytecomparison '+str(cmp['all_files_byte_equal']),artifacts=[str(p.relative_to(D))]);print('restart',cmp,flush=True)
chmt('chemistry-control','initcheck')
