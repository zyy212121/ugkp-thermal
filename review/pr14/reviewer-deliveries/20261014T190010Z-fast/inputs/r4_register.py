from r4_runner import *
import gzip
review=ROOT/'review/pr14';m=json.loads((review/'manifest.json').read_text());rel=str(D.relative_to(review));r=json.loads((D/'results.json').read_text());c=json.loads((D/'coverage.json').read_text());f=json.loads((D/'findings.json').read_text())
for suffix in ['md','json']:
 p=P/('r4_independent_source_review.'+suffix);shutil.copy2(p,D/'metadata'/('independent-source-review.'+suffix))
assert json.loads((D/'metadata/independent-source-review.json').read_text())['interaction_class']=='FAST'
(D/'metadata/independent-source-review-receipt.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','reviewer':'source_review_fast4','files':{str(p.relative_to(D)):sha(p) for p in (D/'metadata').glob('independent-source-review.*')},'read_only_independent_agent':True,'GPU_results_owned_by_primary_receipts':True},indent=2)+'\n')
for p in P.glob('r4_*'):
 if p.is_file() and p.name not in ['r4_independent_source_review.md','r4_independent_source_review.json']:shutil.copy2(p,D/'inputs'/p.name)
source=json.loads((D/'metadata/source-hashes.json').read_text());src={q['path']:q for q in source};registry={}
for q in json.loads((review/'source-index.json').read_text())['files']:
 if q['path'] not in src:continue
 s=src[q['path']];registry[q['id']]={'file_id':q['id'],'path':q['path'],'source_commit':SOURCE,'git_blob_sha1':s['blob'],'sha256':s['sha256'],'function_ids':[i['id'] for i in q.get('functions',[])],'historical_symbols':[{'id':i['id'],'symbol':i.get('symbol'),'locator_kind':i.get('locator_kind')} for i in q.get('functions',[])],'locator_note':'Actualcurrentblob/SHA bound; stablehistoricalIDs are not claimed as compiler-proved completecallgraph.'}
(D/'metadata/current-source-target-registry.json').write_text(json.dumps(registry,indent=2)+'\n')
assert not git('diff','--name-only','--','applications','common','tests','test','tools','README.md')
assert len(source)==1264 and all(sha(ROOT/q['path'])==q['sha256'] and git('rev-parse',SOURCE+':'+q['path'])==q['blob'] for q in source)
assert all(sha(D/q['log'])==q['log_sha256'] for q in r['tests'])
refs=json.loads((D/'metadata/immutable-references.json').read_text())['references'];aliases={q['alias'] for q in refs};packed=json.loads((D/'metadata/lossless-storage.json').read_text())['entries'];packedaliases={q['alias'] for q in packed}
def resolves(n):return (D/n).exists() or n in aliases or n in packedaliases
for u in r['acceptance_units']:assert all(resolves(n) for n in u['evidence_paths']),(u['id'],u['evidence_paths'])
for q in f['findings']:assert all(resolves(n) for n in q['evidence']),(q['id'],q['evidence'])
(D/'inputs/materialize_evidence.py').write_text('''from pathlib import Path
import sys,json,subprocess,hashlib,gzip,shutil
D=Path(__file__).resolve().parents[1];repo=D.parents[3];target=Path(sys.argv[1]).resolve();area=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test').resolve()
assert target.is_relative_to(area) and not target.is_relative_to(repo/'review'), 'Use an independent authorizedtestoutput directory, never historicalreview paths'
target.mkdir(parents=True,exist_ok=True)
for p in D.rglob('*'):
 if p.is_file():
  q=target/p.relative_to(D);assert q.resolve().is_relative_to(target);q.parent.mkdir(parents=True,exist_ok=True)
  if not q.exists():shutil.copy2(p,q)
for r in json.loads((D/'metadata/immutable-references.json').read_text())['references']:
 b=subprocess.check_output(['git','-C',str(repo),'show',r['fixed_commit']+':'+r['repo_path']]);assert hashlib.sha256(b).hexdigest()==r['sha256'];p=target/r['alias'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
for r in json.loads((D/'metadata/lossless-storage.json').read_text())['entries']:
 p=target/r['alias'];p.parent.mkdir(parents=True,exist_ok=True)
 with gzip.open(D/r['stored_path'],'rb') as a,p.open('wb') as b:shutil.copyfileobj(a,b)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['original_sha256']
print('Exacthistorical anddecompressed rawbyte aliases verified:',target)
''')
(D/'REPRODUCE.md').write_text(f'''# PR14_REVIEW_ONLY / FAST4 复现与证据

固定生产 `{SOURCE}` / tree `{PATHS['source_tree']}`，记录父 `{PATHS['delivery_parent']}`。SOURCE.json记录最初工作树和发布前状态，推送后PUBLICATION.json以独立提交记录实际已验证的主证据提交；不把SHA写入自身。

完整命令是commands.txt和results.json tests[].argv/UTC/cwd/spec，计划不是已执行结果。正式native build原flags如versions.json；输入/source/object/binary哈希在spec和metadata。所有automatic_time_limit=null。工具环境logs/META-ENVIRONMENT.log、inputs/env.sh及initial版本保留。先阅读REPORT/coverage/每个acceptance unit，不能一键将所有生成器当成生产任务。

52个大日志/数据lossless gzip：metadata/lossless-storage.json记录compressed和decompressed SHA/原alias，全部解压后SHA已验证；原始未压缩字节保存在同授权test区work/fast4-raw-cache。results[].log指向实际压缩payload，raw_log_alias/raw_log_sha256保持原记录。原raw日志未加前缀或篡改。已上传同字节旧输入/数据只以metadata/immutable-references.json固定commit/path/SHA引用，无重复上传。

需要离线分析时，在独立授权test输出目录物化，例如`python3 inputs/materialize_evidence.py {ROOT}/work/materialized-fast4`，取回固定Git对象并校验解压byte SHA。不要回填旧owned目录。若移动到其他机器，先配置已授权test根；不要指向正式源码、安装、结果或原checkpoint。

inputs/r4_paths.json与drivers保存执行时的绝对路径；重跑须在新的隔离test副本更新这些配置，遵循原输入/端点/flags/阈值。历史脚本仅记录编排，不要在现有delivery重跑覆盖原证据。inputs/r4_stage_audit-initial.py、r4_physical_audit-initial.py和S407版本与原spec哈希完全匹配；S408patch留最终geometry选择修改。S401/S403/S406与D401/D402patch/源单列；S405初编排异常stdout未完整捕获这一限制明确保留。

可执行binary/object不上传，exactSHA/native构建命令、最终源/patch足够复编，所有本轮binary仍在授权test安装/cache。NativeGPU原版、diagnostic/support、host、NOT_RUN分别解释。Nsightrawtrace/tables用于资源计数，非profiling样本独立；新观察器D2H/printf不能用于效率。不得通过恢复productionaudit、改输入/终点/精度/容差或新停算门禁得到通过。

最终formalfixture/strictrestart/旧API测试红保留。14原gap都没有因subsetPASS自动关闭。完整三应用gross/当前selectedowners/混合非零speciesV1/更多实际资源错误仍需开发修复或下一次独立定向证据，当前不merge-ready。
''')
for p in (D/'inputs').rglob('*.pyc'):p.unlink()
(D/'metadata/driver-scripts.json').write_text(json.dumps([{'path':str(p.relative_to(D)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted((D/'inputs').glob('r4_*')) if p.is_file()],indent=2)+'\n')
envspecs={q['environment_script_sha256'] for q in r['tests']};envs={sha(p) for p in (D/'inputs').glob('env*.sh')};assert envspecs.issubset(envs)
(D/'logs/validate-records.json').write_text('{}\n');(D/'SHA256SUMS').write_text('')
existing={q['sha256']:q['id'] for q in m['evidence']};byid={q['id']:q for q in m['evidence']};emap={}
for p in sorted(D.rglob('*')):
 if not p.is_file() or p.name in ['SHA256SUMS','validate-records.json']:continue
 h=sha(p);eid=existing.get(h)
 if eid is None:
  eid='E014-'+h[:16]
  if eid in byid and byid[eid]['sha256']!=h:eid='E014-'+h[:24]
  obj={'id':eid,'path':str(p.relative_to(review)),'sha256':h,'bytes':p.stat().st_size,'identity':'PR14_REVIEW_ONLY','interaction_class':'FAST','source_commit':SOURCE,'scope':'FAST4exactsource realGPU independent scoped evidence; formalred/support/host/diagnostic/unexecuted separated; no wholeFASTor merge approval.'};m['evidence'].append(obj);existing[h]=eid;byid[eid]=obj
 emap[str(p.relative_to(D))]=eid
for q in refs:emap[q['alias']]=q['evidence_id']
for q in packed:emap[q['alias']]=emap[q['stored_path']]
main=['REPORT.md','TEST_PLAN.md','SOURCE.json','VALIDATION.json','coverage.json','results.json','findings.json','environment.json','versions.json','REPRODUCE.md','metadata/source-hashes.json','metadata/current-source-target-registry.json','metadata/immutable-references.json','metadata/lossless-storage.json','metadata/evidence-reuse.json','metadata/independent-source-review.json','metadata/developer-remote-evidence-verification.json'];mainids=[emap[n] for n in main]
for t in m['topics']:
 cv=next(q for q in c['topics'] if q['topic_id']==t['id']);selected=[q for q in r['tests'] if q['id'] in cv['executed_test_ids']];ids=list(mainids)
 for q in selected:
  names=[q['log'],q.get('raw_log_alias',q['log']).replace('.log','.spec.json')];ids += [emap[n] for n in names if n in emap]
 for u in r['acceptance_units']:
  if t['id'] in u['topics']:ids += [emap[n] for n in u['evidence_paths'] if n in emap]
 ids=list(dict.fromkeys(ids));targets=[registry[q] for q in t.get('file_ids',[]) if q in registry];response=json.loads((review/t['response_path']).read_text());entry={'id':t['id']+'-REVIEWER-'+D.name,'question_id':t['id'],'author':'Codex independent actualGPUreviewer +read-onlysource/evidencescope reviewer','recorded_utc':utc(),'interaction_class':'FAST','review_round':'FAST4','source_commit':SOURCE,'source_tree':PATHS['source_tree'],'record_parent':PATHS['delivery_parent'],'baseline_source_commit':PATHS['old_source'],'prior_evidence_commit':PATHS['old_delivery_commit'],'file_ids':t.get('file_ids',[]),'function_ids':list(dict.fromkeys(i for q in targets for i in q['function_ids'])),'source_blobs':[{k:q[k] for k in ['file_id','path','git_blob_sha1','sha256']} for q in targets],'delivery_path':rel,'report':rel+'/REPORT.md','plan':rel+'/TEST_PLAN.md','coverage':rel+'/coverage.json','results':rel+'/results.json','versions':rel+'/versions.json','environment':rel+'/environment.json','toolchain_device':'RTX4060Laptop/CUDA13.1.115/g++11.4/OF10; environment.json andMETA-ENVIRONMENT actuallog','exact_command_input_dependency_binary_utc_exit':'results tests[]/logs.spec.json/commands/versions/source-registry; compressedraworiginalSHA aliases verified','automatic_time_limit':None,'evidence_ids':ids,'verdict':cv['verdict'],'acceptance_unit_ids':cv['acceptance_unit_ids'],'executed_test_ids':cv['executed_test_ids'],'remaining_gaps':cv['remaining_gaps'],'findings':cv['finding_ids'],'support_PASS_not_formal_fixture_PASS':True,'legacy_zero_not_measured_conservation':True,'retired_support_metric':'RETIRED_BY_USER','counts_not_additive_across_topics':True,'whole_FAST_acceptance':False,'merge_ready':False,'merge_authorized':False}
 assert not any(q['id']==entry['id'] for q in response.get('reviewer_results',[]));response.setdefault('reviewer_results',[]).append(entry);response.setdefault('recheck_results',[]).append({**entry,'id':t['id']+'-RECHECK-'+D.name,'reply_to':[q['id'] for q in response.get('developer_replies',[]) if q.get('source_commit')==SOURCE],'reviewer_result_id':entry['id'],'status':'FAST4_SCOPED_REAL_GPU_RECHECK_OPEN_LIMITATIONS'});response.setdefault('reviewer_delivery_decision_history',[]).append({'recorded_utc':entry['recorded_utc'],'source_commit':SOURCE,'delivery_path':rel,'status':'FAST4_FORMAL_RED_PARTIAL_GAPS_RETAINED','remaining_gaps':cv['remaining_gaps'],'evidence_ids':mainids[:7],'merge_authorized':False});(review/t['response_path']).write_text(json.dumps(response,ensure_ascii=False,indent=2)+'\n');t['evidence_ids']=list(dict.fromkeys(t.get('evidence_ids',[])+ids));t['latest_reviewer_result']=entry['id'];t['status']=cv['verdict']
pending=json.loads((review/'pending-findings.json').read_text());pending.setdefault('independent_reviewer_delivery_history',[]).append({'recorded_utc':utc(),'delivery_path':rel,'source_commit':SOURCE,'findings':f['findings'],'evidence_ids':[emap['findings.json'],emap['REPORT.md']],'merge_authorized':False});pending['latest_reviewer_delivery']=rel;pending['gpu_acceptance_final']=False;pending['limits']='FAST4exact470freshnativeflags/currentGPU evidence; formalchemistryfixture/staleAPI/strictrestart red and14partialscopegaps; oldsupport metricretired, realgrossunmeasured.';(review/'pending-findings.json').write_text(json.dumps(pending,ensure_ascii=False,indent=2)+'\n')
with (review/'INDEX.md').open('a') as h:h.write('\n## FAST4 独立真实GPU '+D.name+'\n\n固定生产47034e4e；新原flags完整构建、原evap完整.001/319windows、Ns10化学两half、9组SST真实净/gross/RK低Re物理范围、22实际资源槽、activecold2D、限定双向FSH端点和Nsight真实资源证据。正式fixture/API编译/strictrestart仍FAIL；14gap全部PARTIAL、旧support总账RETIRED_BY_USER；不满足整体FAST，无合并授权。\n\n- [报告]('+rel+'/REPORT.md)\n- [计划]('+rel+'/TEST_PLAN.md)\n- [覆盖]('+rel+'/coverage.json)\n- [实际命令与结果]('+rel+'/results.json)\n- [发现]('+rel+'/findings.json)\n- [原始数据物化与复现]('+rel+'/REPRODUCE.md)\n')
m.setdefault('reviewer_deliveries',[]).append({'batch':D.name,'path':rel,'source_commit':SOURCE,'source_tree':PATHS['source_tree'],'record_parent':PATHS['delivery_parent'],'evidence_ids':mainids,'classification':'FAST4_REAL_GPU_SCOPED_FORMAL_RED_PARTIAL_GAPS','publication_commit':'Pendingnormalpush; immutableGitrecordhistory andpostpushPUBLICATION sidecar','shared_content_policy':'OneEIDperSHA; priorimmutablebytes referenced; necessarynewrawdata gzipdecompressedSHA verified.'});m['latest_reviewer_delivery']=rel;m['reviewer_gpu_acceptance']='FAST4_FORMAL_RED_PARTIAL_SCOPES_REMAIN';m['global_review_status']='FAST4_NOT_MERGE_READY';m['merge_authorized']=False;m['owned_paths']=sorted('review/pr14/'+str(p.relative_to(review)) for p in review.rglob('*') if p.is_file());(review/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
q=subprocess.run(['python3',str(review/'validate_records.py')],capture_output=True,text=True);(D/'logs/validate-records.json').write_text(q.stdout+q.stderr);print(q.stdout,q.stderr,flush=True);assert q.returncode==0
with (D/'SHA256SUMS').open('w') as h:
 for p in sorted(D.rglob('*')):
  if p.is_file() and p.name!='SHA256SUMS':h.write(sha(p)+'  '+str(p.relative_to(D))+'\n')
print('REGISTERED',len(r['tests']),'actualcommands',len(r['acceptance_units']),'units',sum(p.is_file() for p in D.rglob('*')),'newfiles',len(m['evidence']),'sharedEIDs',flush=True)
