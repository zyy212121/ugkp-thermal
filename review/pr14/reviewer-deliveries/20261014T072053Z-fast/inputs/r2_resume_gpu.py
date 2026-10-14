import json,shutil,re
from r2_runner import *
from r2_native import compare_dirs,chmt
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n续跑环境确认：用户报告死机后，114条完整结果及日志哈希一致、无孤立spec，无生产源码diff。追加一次既有真实Co二进制健康观察，单独标为环境确认，不增加验收覆盖。此前SST-RK实例实际ddt=Euler，保留原结果；新增仅ddtSchemes默认配置SSPRK2/SSPRK3的两个短程实例，热力参数、终点、阈值与原thermal保持一致，检验真实SST累计审计/实际stage packet及完整库存。Ns10补原chemistry-full initcheck，保留相同输入/步数/flags。均外部300s，GPU串行。\n')
b=ROOT/'work/fixtures/courant_reduction'
if not (D/'logs/RESUME-GPU-HEALTH.log').exists():
 t=run('RESUME-GPU-HEALTH',cmd(b),[26],expected='Existing actual CUDA operator executable completes after reboot; environment smoke only, no duplicated acceptance count.',binary=b,kind='ENVIRONMENT');assertion(t,t['exit_code']==0,(D/t['log']).read_text())
done={x['id']:x for x in json.loads((D/'results.json').read_text())['tests']}
for method in ['SSPRK2','SSPRK3']:
 name='SST-'+method;case=ROOT/'work/cases/extra'/name
 if not case.exists():
  shutil.copytree(ROOT/'work/cases/extra/SST-RK',case,ignore=shutil.ignore_patterns('chmtOutput','log*'))
  q=case/'system/fvSchemes';s=q.read_text();s,n=re.subn(r'\bdefault\s+Euler\s*;','default '+method+';',s);assert n==1;q.write_text(s)
 inp=D/'inputs/native/extra'/name
 if not inp.exists():shutil.copytree(case,inp,ignore=shutil.ignore_patterns('polyMesh'))
 b=ROOT/'work/install/ns2/bin/CHMT'
 t=done.get('EXTRA-'+name) or run('EXTRA-'+name,cmd(b,'-case',case),[8,9,14,22,26,29],expected='Actual native active SST, configured '+method+'; full inventories, actual cumulative weighted audit and packets; independent closure without new tolerance.',inputs=[p for p in inp.rglob('*') if p.is_file()],binary=b)
 if (case/'chmtOutput').is_dir():shutil.copytree(case/'chmtOutput',D/'artifacts/extra'/name,dirs_exist_ok=True)
 ck=done.get('CHECK-'+name) or run('CHECK-'+name,cmd('python3',ROOT/'applications/CHMT/verification/coupled_runtime/check_case.py',case),[8,9,14,26],expected='Unchanged original native mass/energy/EOS/packet assertions.',kind='ANALYSIS')
 assertion(ck,ck['exit_code']==0,(D/ck['log']).read_text())
 reader=ROOT/'work/fixtures/checkpoint-ASan-2'
 for i in range(3):
  cp=case/'chmtOutput'/('checkpoint-'+str(i))
  rd=run('HOST-DECODE-'+name+'-'+str(i)+'-DIR',cmd(reader,cp),[8,9,26],expected='Actual checkpoint decoder under host ASan/UBSan for actual GPU RK/SST inventory; API takes directory, prior wrong-file argument retained.',binary=reader,kind='SANITIZER',inputs=[cp/'state.chmt'])
  raw=(D/rd['log']).read_text();data=json.loads(raw);(D/'artifacts/extra'/name/('checkpoint-'+str(i)+'-decoded.json')).write_text(json.dumps(data,indent=2)+'\n');assertion(rd,rd['exit_code']==0,'Actual current codec read under ASan/UBSan; decoded actual GPU state retained.')
 amend(t['id'],verdict='INCONCLUSIVE',observations='Configured actual '+method+' and original physical checker '+ck['verdict']+'; independent stage/budget closure follows.',artifacts=['artifacts/extra/'+name])
chmt('chemistry-full','initcheck')
