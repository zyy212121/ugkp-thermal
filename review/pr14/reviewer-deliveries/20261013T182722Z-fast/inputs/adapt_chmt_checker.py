import pathlib,re,json,hashlib,difflib
from run_fast import ROOT,D,run,cmd,amend
source=ROOT/'applications/CHMT/restart/Checkpoint.C';schema=re.search(r'const char\* const FieldSchema="([^"]+)";',source.read_text()).group(1)
value=14695981039346656037
for b in schema.encode():value=((value^b)*1099511628211)&((1<<64)-1)
checker=ROOT/'applications/CHMT/verification/coupled_runtime/check_case.py';old=checker.read_text();new=old.replace('FIELD_SCHEMA_HASH = 15664811910203657810',f'FIELD_SCHEMA_HASH = {value}')
assert old!=new
target=D/'inputs/check_case_current.py';target.write_text(new)
(D/'patches/P001-checkpoint-checker-schema.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/applications/CHMT/verification/coupled_runtime/check_case.py',tofile='b/applications/CHMT/verification/coupled_runtime/check_case.py')))
(D/'patches/P001-checkpoint-checker-schema.json').write_text(json.dumps({'classification':'test-support-only','production_source_modified':False,'root_cause':'Pinned checker fingerprint predates removal of HostState pending-state fields; current production FieldSchema changed. No acceptance threshold or payload assertion deleted.','old_hash':15664811910203657810,'expected_current_hash':value,'derivation':'Independent Python FNV-1a over frozen source FieldSchema literal; expected value not copied from output checkpoint.','schema_literal':schema,'Checkpoint_C_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'original_checker_sha256':hashlib.sha256(checker.read_bytes()).hexdigest(),'adapted_checker_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'wire_budget_tail_verified':'45+5*Ns+Ne binary64 scalars and one uint64 consumedPackets; time+dQQQ preceding tail unchanged.'},ensure_ascii=False,indent=2)+'\n')
with (D/'TEST_PLAN.md').open('a') as f:f.write(f'\n## 追加 P001：失效的独立checkpoint检查器\n\n原版GPU输出先保留。冻结源FieldSchema的独立FNV-1a期待值为{value}，原checker常量15664811910203657810过时；仅适配该常量，保留全部数值/格式/assertion。预算与输入输出不变，不重跑未失效GPU计算；以原输出新增CHECKER1复验，原失败记录不覆盖。\n')
print('Expected schema '+str(value))
for case in sorted((ROOT/'work/cases/chmt').iterdir()):
    if not (case/'chmtOutput').is_dir():continue
    args=['python3',target,case];topics=[3,5,7,8,9,11,14,19,29]
    if case.name.endswith('-resumed'):
        scenario=case.name.removesuffix('-resumed');sourcecp=max((ROOT/'work/cases/chmt'/(scenario+'-split')/'chmtOutput').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
        args+=['--reference',ROOT/'work/cases/chmt'/(scenario+'-full'),'--restart-from',sourcecp]
    if case.name.endswith('-full'):
        scenario=case.name.removesuffix('-full');fine=case.with_name(scenario+'-refined')
        if fine.is_dir():args+=['--refinement',fine]
        if scenario in ['diffusion','chemistry']:args += ['--diffusion-control' if scenario=='diffusion' else '--chemistry-control',case.with_name(scenario+'-control')]
    res=run('CHECKER1-CHMT-'+case.name,cmd(*args),topics+([10] if case.name.startswith('chemistry') else []),300,'Pinned release schema fingerprint adapted independently; original conservation/EOS/packet/refinement/restart assertions all retained.',inputs=[target,*list((case/'chmtOutput').rglob('state.chmt'))],kind='ANALYSIS')
    amend(res['id'],verdict='PASS' if res['exit_code']==0 else 'FAIL',observations=(D/res['log']).read_text(errors='replace'),test_support_patch='P001')
