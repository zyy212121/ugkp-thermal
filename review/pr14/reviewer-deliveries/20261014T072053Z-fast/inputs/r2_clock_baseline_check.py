from r2_runner import *
from r2_native import compare_dirs
import json
case=ROOT/'work/clock-cases/baseline';saved=case/'saved-clock-output'
assert (case/'chmtOutput').resolve().is_relative_to((ROOT/'work').resolve())
(case/'chmtOutput').rename(saved)
b=OLDROOT/'work/install/base/bin/CHMT'
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nD追加同目录baseline无observer对照：原不可变baseline preflight的三个checkpoint hash不匹配，新trace CSV/geometry完全匹配。为区分路径/元数据与observer干扰，保留全部trace输出，将同case原输出重命名后，用既有真实d611原版binary于完全相同case路径/输入重跑一次（300s，不计效率），比较全部bytes并使用该版原checker。\n')
t=run('TRACE-BASELINE-UNINSTRUMENTED-SAMECASE',cmd(b,'-case',case),[5,14,15,27],expected='Originald611 executable sameabsolutecase/inputs afterpreservingtraceoutput; isolate checkpoint identity vs observer.',binary=b,variant='baseline')
cmp=compare_dirs(saved,case/'chmtOutput');p=D/'artifacts/clock-trace/baseline-samecase-uninstrumented-comparison.json';p.write_text(json.dumps(cmp,indent=2)+'\n')
assertion(t,cmp['all_files_byte_equal'],'All original noobserver vs trace output atsamecase bytes equal '+str(cmp['all_files_byte_equal']),artifact=str(p.relative_to(D)))
for rel in ['checkpoint-0/state.chmt','checkpoint-1/state.chmt','checkpoint-2/state.chmt']:
 a=OLD/'artifacts/performance/native/baseline-preflight-0'/rel;b=case/'chmtOutput'/rel;x=a.read_bytes();y=b.read_bytes();idx=[i for i in range(min(len(x),len(y))) if x[i]!=y[i]]
 print(rel,'lengths',len(x),len(y),'differences',len(idx),'first',idx[:30]);
 if idx:
  first=idx[0];print('expected',x[max(0,first-70):first+150]);print('current',y[max(0,first-70):first+150])
checker=ROOT/'work/clock-source/baseline-v1/applications/CHMT/verification/coupled_runtime/check_case.py'
t=run('TRACE-BASELINE-PHYSICAL-CHECK',cmd('python3',checker,case),[3,5,9,14,27],expected='Actuald611 own originalphysical thresholds/checkschema.',inputs=[checker],kind='ANALYSIS',variant='baseline');assertion(t,t['exit_code']==0,(D/t['log']).read_text())
