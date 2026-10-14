from r2_runner import *
from collections import defaultdict,Counter
import re
review=ROOT/'review/pr14'; rel=str(D.relative_to(review))
m=json.loads((review/'manifest.json').read_text())
r=json.loads((D/'results.json').read_text());c=json.loads((D/'coverage.json').read_text());f=json.loads((D/'findings.json').read_text())
oldr=json.loads((OLD/'results.json').read_text());oldf=json.loads((OLD/'findings.json').read_text())
gap_topics={q['id'].removeprefix('NOT-RUN-'):q.get('topics',[]) for q in oldr['acceptance_units'] if q['id'].startswith('NOT-RUN-')}
for g in c['remaining_gaps']:g['topics']=gap_topics[g['gap_id']]
for t in c['topics']:
 t['remaining_gaps']=[g['gap_id'] for g in c['remaining_gaps'] if t['topic_id'] in g['topics'] and not g['verdict'].startswith('CLOSED')]
 t['finding_ids']=[q['id'] for q in f['findings'] if int(t['topic_id'].split('-')[-1]) in q['topics']]
 if t['remaining_gaps'] or any(q['verdict'] in ['FAIL','BLOCKED','NOT_RUN'] and t['topic_id'] in q['topics'] for q in r['acceptance_units']):t['verdict']='PARTIAL_RECHECK_OPEN_GAPS'
 elif t['acceptance_unit_ids']:t['verdict']='PASS_SCOPED_NOT_WHOLE_TOPIC'
 else:t['verdict']='REUSED_OR_ADMINISTRATIVE_SCOPE'
(D/'coverage.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
r['remaining_gap_status']=c['remaining_gaps']
(D/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
# Historical observations remain at their immutable prior source and delivery.
for q in f['findings']:
 if q['id'] in {x['id'] for x in oldf['findings']}:
  h=next(x for x in oldf['findings'] if x['id']==q['id'])
  keep={k:q[k] for k in ['id','priority','title','topics','recheck_status','recheck_observations'] if k in q}
  keep.update(status=q.get('recheck_status','RECHECK_PENDING'),source_commit=SOURCE,historical_status=h['status'],historical_finding={'fixed_commit':PATHS['old_delivery_commit'],'path':str((OLD/'findings.json').relative_to(OLDROOT)),'sha256':sha(OLD/'findings.json'),'id':q['id'],'original_source_commit':PATHS['old_source']},scope='Formal repair rechecked only within named acceptance units; no whole-topic or merge approval.')
  q.clear();q.update(keep)
 for k in ['evidence']:
  if k in q:q[k]=[p.replace('S202-fixture-host-device-owner-split.patch','S202-fixture-host-device-split.patch') for p in q[k]]
f['identity']='PR14_REVIEW_ONLY';f['interaction_class']='FAST';f['source_tree']=PATHS['source_tree']
(D/'findings.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n')
# Reference identical immutable payloads without adding their bytes again.
old_by_sha={e['sha256']:e for e in m['evidence'] if e['path'].startswith('reviewer-deliveries/20261013T182722Z-fast/')}
dev_by_sha={e['sha256']:e for e in m['evidence'] if e['path'].startswith('developer-deliveries/20261014T055427Z-fast/')}
refs=[]; verified={};removed_bytes=0
for p in sorted(D.rglob('*')):
 if not p.is_file():continue
 # Keep all newly captured raw logs/specs and all new analysis/metadata/patches.
 if p.relative_to(D).parts[0] not in ['inputs','artifacts']:continue
 if '__pycache__' in p.parts: p.unlink();continue
 digest=sha(p);e=old_by_sha.get(digest) or dev_by_sha.get(digest)
 if e is None:continue
 commit=PATHS['old_delivery_commit'] if e['path'].startswith('reviewer-deliveries/') else PATHS['delivery_parent']
 repo_path='review/pr14/'+e['path'];key=(commit,repo_path)
 if key not in verified:
  payload=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+repo_path]);assert hashlib.sha256(payload).hexdigest()==digest,key
  verified[key]=True
 assert p.resolve().is_relative_to(D.resolve())
 refs.append({'alias':str(p.relative_to(D)),'fixed_commit':commit,'repo_path':repo_path,'sha256':digest,'bytes':p.stat().st_size,'evidence_id':e['id'],'why':'Content identical to prior immutable evidence; new execution is separately pinned by its current raw log/spec.'})
 removed_bytes+=p.stat().st_size;p.unlink()
refobj={'identity':'PR14_REVIEW_ONLY','source_commit':SOURCE,'references':refs,'referenced_alias_count':len(refs),'bytes_not_duplicated':removed_bytes,'all_fixed_Git_blobs_verified':True,'hydration':'Use inputs/hydrate_immutable.py from the current delivery against this repository. Local original test case files remain preserved in the authorized test area.'}
(D/'metadata/immutable-references.json').write_text(json.dumps(refobj,indent=2)+'\n')
hydrate='''#!/usr/bin/env python3
"""PR14_REVIEW_ONLY. Restore only absent aliases whose immutable content hash is verified."""
from pathlib import Path
import json,hashlib,subprocess
D=Path(__file__).resolve().parents[1]
ROOT=D.parents[3]
data=json.loads((D/'metadata/immutable-references.json').read_text())
for q in data['references']:
 p=D/q['alias']
 assert p.resolve().is_relative_to(D.resolve()), q['alias']
 content=subprocess.check_output(['git','-C',str(ROOT),'show',q['fixed_commit']+':'+q['repo_path']])
 assert hashlib.sha256(content).hexdigest()==q['sha256'], q['alias']
 if p.exists(): assert hashlib.sha256(p.read_bytes()).hexdigest()==q['sha256'], q['alias']
 else: p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
print('Verified/restored aliases:',len(data['references']))
'''
(D/'inputs/hydrate_immutable.py').write_text(hydrate)
with (D/'REPRODUCE.md').open('a') as h:h.write('\n## 已有不可变数据\n\n先在已获取固定历史提交的本仓库运行 `python3 '+str(D.relative_to(ROOT))+'/inputs/hydrate_immutable.py`，按 metadata/immutable-references.json 验证 Git blob 与 SHA256，再恢复原分析命令使用的缺省路径。脚本只恢复已列明的缺失文件；新 raw logs/specs 和失败数据保留在本轮目录。恢复出来的旧别名不属于本轮新增 owned_paths，不在新 SHA256SUMS 中计为新证据。\n')
with (D/'TEST_PLAN.md').open('a') as h:h.write('\n## 最终收口\n\n本轮221条执行记录，25个限定验收单元(18PASS/4FAIL/2BLOCKED/1NOT_RUN)，跨主题不叠加；18PASS包括明确标识的诊断、host sanitizer和资源统计，不是18项正式GPU数值通过。未受改动影响的36旧单元只按原范围引用；复用表、14旧gap ID、所有范围红/支持红/原版与诊断红完整保留。覆盖矩阵按旧gap实际topics映射，不把无关缺口挂到所有主题。完整生产源码1152路径/所有221原始log hash已校验。没有生产长算。\n')
# Archive exact drivers once, before hashing and manifest registration.
omit={'r2_publication_inspect.py','r2_evidence_inventory.py'}
for p in P.glob('r2_*'):
 if p.is_file() and p.name not in omit:shutil.copy2(p,D/'inputs'/p.name)
scripts=[{'path':str(p.relative_to(D)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted((D/'inputs').glob('r2_*')) if p.is_file()]
(D/'metadata/driver-scripts.json').write_text(json.dumps(scripts,indent=2)+'\n')
# Current source identities accompany stable historical function IDs.
source_by_path={q['path']:q for q in json.loads((D/'metadata/source-hashes.json').read_text())}
src=json.loads((review/'source-index.json').read_text());registry={}
for q in src['files']:
 current=source_by_path.get(q['path'])
 if current:
  registry[q['id']]={'file_id':q['id'],'path':q['path'],'source_commit':SOURCE,'git_blob_sha1':current['blob'],'sha256':current['sha256'],'function_ids':[x['id'] for x in q.get('functions',[])],'historical_symbols':[{'id':x['id'],'symbol':x.get('symbol'),'locator_kind':x.get('locator_kind')} for x in q.get('functions',[])],'locator_note':'Stable IDs and historical symbols only; current blob identity exact. Historical supplementary line numbers are not relabeled current.'}
(D/'metadata/current-source-target-registry.json').write_text(json.dumps(registry,indent=2)+'\n')
for q in f['findings']:
 for name in q.get('evidence',[]):assert (D/name).is_file(),name
assert all(sha(D/t['log'])==t['log_sha256'] for t in r['tests'])
assert all(sha(ROOT/q['path'])==q['sha256'] for q in source_by_path.values())
assert not git('diff','--name-only','--','applications','common','tests','tools','README.md')
print('FROZEN_CONTENT',len(refs),'immutable aliases',removed_bytes,'bytes referenced; current scripts',len(scripts),flush=True)
# Validator and checksums are owned, but not manifest evidence (avoid self-reference).
(D/'logs/validate-records.json').write_text('{}\n');(D/'SHA256SUMS').write_text('')
existing={e['sha256']:e['id'] for e in m['evidence']};byid={e['id']:e for e in m['evidence']};emap={}
for p in sorted(D.rglob('*')):
 if not p.is_file() or p.name in ['SHA256SUMS','validate-records.json']:continue
 digest=sha(p);eid=existing.get(digest)
 if eid is None:
  eid='E014-'+digest[:16]
  if eid in byid and byid[eid]['sha256']!=digest:eid='E014-'+digest[:24]
  e={'id':eid,'path':str(p.relative_to(review)),'sha256':digest,'bytes':p.stat().st_size,'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','source_commit':SOURCE,'scope':'Independent second-round scoped evidence. Original and diagnostic results distinguished in REPORT/results. No merge authorization.'}
  m['evidence'].append(e);existing[digest]=eid;byid[eid]=e
 emap[str(p.relative_to(D))]=eid
for q in refs:emap[q['alias']]=q['evidence_id']
main_names=['REPORT.md','TEST_PLAN.md','coverage.json','results.json','findings.json','versions.json','environment.json','SOURCE.json','metadata/source-hashes.json','metadata/current-source-target-registry.json','metadata/immutable-references.json','metadata/reused-evidence.json']
mainids=[emap[x] for x in main_names]
for t in m['topics']:
 cv=next(x for x in c['topics'] if x['topic_id']==t['id']);tests=[q for q in r['tests'] if t['id'] in q['topics']]
 ids=mainids+[emap[n] for q in tests for n in [q['log'],q['log'].replace('.log','.spec.json')] if n in emap]
 ids=list(dict.fromkeys(ids));fp=[registry[x] for x in t.get('file_ids',[]) if x in registry]
 obj=json.loads((review/t['response_path']).read_text());entry={'id':t['id']+'-REVIEWER-'+D.name,'question_id':t['id'],'author':'Codex independent reviewer; actual RTX4060 GPU execution','recorded_utc':utc(),'interaction_class':'FAST','source_commit':SOURCE,'source_tree':PATHS['source_tree'],'record_parent':PATHS['delivery_parent'],'baseline_commit':PATHS['baseline'],'byte_baseline_production':PATHS['old_source'],'prior_evidence_commit':PATHS['old_delivery_commit'],'file_ids':t['file_ids'],'function_ids':list(dict.fromkeys(i for q in fp for i in q['function_ids'])),'source_blobs':[{k:q[k] for k in ['file_id','path','git_blob_sha1','sha256']} for q in fp],'delivery_path':rel,'report':rel+'/REPORT.md','plan':rel+'/TEST_PLAN.md','coverage':rel+'/coverage.json','results':rel+'/results.json','versions':rel+'/versions.json','environment':rel+'/environment.json','exact_commands_and_timestamps':rel+'/commands.txt and logs/<test-id>.spec.json','input_binary_dependency_hashes':rel+'/results.json tests[] and versions.json; fixed source blob/hash registry above','evidence_ids':ids,'verdict':cv['verdict'],'acceptance_unit_ids':cv['acceptance_unit_ids'],'executed_test_ids':cv['executed_test_ids'],'reused_unit_ids':cv['reused_unit_ids'],'remaining_gaps':cv['remaining_gaps'],'findings':cv['finding_ids'],'diagnostic_PASS_does_not_close_original':True,'counts_not_additive_across_topics':True,'merge_ready':False,'merge_authorized':False}
 if not any(q['id']==entry['id'] for q in obj.get('reviewer_results',[])):obj.setdefault('reviewer_results',[]).append(entry)
 if cv['acceptance_unit_ids'] or cv['remaining_gaps'] or cv['reused_unit_ids']:
  recheck={**entry,'id':t['id']+'-RECHECK-'+D.name,'reply_to':[q['id'] for q in obj.get('developer_replies',[]) if q.get('source_commit')==SOURCE],'reviewer_result_id':entry['id'],'status':'FORMAL_RECHECK_SCOPED_OPEN_LIMITATIONS'}
  if not any(q['id']==recheck['id'] for q in obj.get('recheck_results',[])):obj.setdefault('recheck_results',[]).append(recheck)
 obj.setdefault('reviewer_delivery_decision_history',[]).append({'recorded_utc':entry['recorded_utc'],'source_commit':SOURCE,'delivery_path':rel,'status':'FAST_SCOPED_RECHECK_COMPLETE_NOT_MERGE_READY','remaining_gaps':cv['remaining_gaps'],'evidence_ids':mainids[:5],'merge_authorized':False})
 (review/t['response_path']).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');t['evidence_ids']=list(dict.fromkeys(t['evidence_ids']+ids));t['latest_reviewer_result']=entry['id'];t['status']=cv['verdict']
pending=json.loads((review/'pending-findings.json').read_text());pending.setdefault('independent_reviewer_delivery_history',[]).append({'recorded_utc':utc(),'delivery_path':rel,'findings':f['findings'],'evidence_ids':[emap['findings.json'],emap['REPORT.md']],'source_commit':SOURCE,'formal_fix_commits':[SOURCE],'merge_authorized':False});pending['latest_reviewer_delivery']=rel;pending['gpu_acceptance_final']=False;pending['limits']='Round2 formal scoped rechecks recorded; GPU-F006 formal repair, two test support defects and remaining coverage gaps prevent whole-PR merge readiness.'
(review/'pending-findings.json').write_text(json.dumps(pending,ensure_ascii=False,indent=2)+'\n')
with (review/'INDEX.md').open('a') as h:h.write('\n## 第二轮独立真实GPU FAST '+D.name+'\n\n正式修复4b5c480b；交付起点75cfd885。25个限定单元(18PASS/4FAIL/2BLOCKED/1NOT_RUN)，不可跨主题相加。原指定F001/F002/F004/F005输入定向复验、实际clock位差链已记录。GPU-F006既有wet材料索引问题及两支持夹具问题待开发者；其余严格红/阻断/未测范围保留，不满足整体合并条件。\n\n- [报告]('+rel+'/REPORT.md)\n- [计划]('+rel+'/TEST_PLAN.md)\n- [覆盖与旧gap ID]('+rel+'/coverage.json)\n- [原始结果]('+rel+'/results.json)\n- [发现]('+rel+'/findings.json)\n- [复现与不可变引用]('+rel+'/REPRODUCE.md)\n')
m.setdefault('reviewer_deliveries',[]).append({'batch':D.name,'path':rel,'source_commit':SOURCE,'source_tree':PATHS['source_tree'],'record_parent':PATHS['delivery_parent'],'evidence_ids':mainids,'classification':'SECOND_ROUND_REAL_GPU_FAST_SCOPED_NOT_MERGE_READY','shared_content_policy':'Prior immutable payload bytes referenced by fixed commit/SHA; new raw run provenance retained. One evidence ID per content SHA256.'})
m['latest_reviewer_delivery']=rel;m['reviewer_gpu_acceptance']='SECOND_ROUND_SCOPED_FORMAL_RECHECK_COMPLETE_OPEN_FINDINGS_AND_GAPS';m['global_review_status']='FAST2_NOT_MERGE_READY_GPU_F006_SUPPORT_AND_COVERAGE_OPEN';m['merge_authorized']=False
m['owned_paths']=sorted('review/pr14/'+str(p.relative_to(review)) for p in review.rglob('*') if p.is_file())
(review/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
(ROOT/'work/round2-evidence-map.json').write_text(json.dumps(emap,indent=2)+'\n')
q=subprocess.run(['python3',str(review/'validate_records.py')],capture_output=True,text=True);(D/'logs/validate-records.json').write_text(q.stdout+q.stderr);print(q.stdout,q.stderr,flush=True);assert q.returncode==0
with (D/'SHA256SUMS').open('w') as h:
 for p in sorted(D.rglob('*')):
  if p.is_file() and p.name!='SHA256SUMS':h.write(sha(p)+'  '+str(p.relative_to(D))+'\n')
print('FINAL',sum(p.is_file() for p in D.rglob('*')),'new-delivery files;',len(m['evidence']),'shared evidence;',len(m['owned_paths']),'owned paths',flush=True)
