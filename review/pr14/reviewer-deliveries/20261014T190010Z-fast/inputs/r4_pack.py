from r4_runner import *
import gzip,difflib
for p in P.glob('r4_*'):
 if p.is_file():shutil.copy2(p,D/'inputs'/p.name)
# Reconstruct exact versions of the three edited analysis inputs, then require the original spec SHA.
versions=[]
s=(P/'r4_stage_audit.py').read_text();s=s.replace("  if '--only' in sys.argv and log.name!='D401-GPU-'+sys.argv[sys.argv.index('--only')+1]+'.log':continue\n",'').replace("  print('ANALYZE_START',log.name,flush=True)\n",'')
for name,payload,expected in [('r4_stage_audit-initial.py',s,'bb19f605a7f238b71672641a0d1a379120a32d38bfafb80eabcabb8cd4ef6369')]:
 assert hashlib.sha256(payload.encode()).hexdigest()==expected,(name,hashlib.sha256(payload.encode()).hexdigest());p=D/'inputs'/name;p.write_text(payload);versions.append({'path':str(p.relative_to(D)),'sha256':sha(p),'matches_original_executed_spec':True})
p=D/'inputs/r4_physical_audit-initial.py';assert sha(p)=='c53493f047df23a241114203c23d4ad2ab374de5f10d5faf1b20da6caace382a';versions.append({'path':str(p.relative_to(D)),'sha256':sha(p),'matches_original_executed_spec':True})
s=(P/'r4_physical_audit.py').read_text();begin=s.index("  if name in ['phase-melting','phase-solidification']:");end=s.index("  if stageindex%r['rk']==0:",begin);s=s[:begin]+s[end:]
s=s.replace("'phase_replay_selection':'Final committed HostStageGeometry.newVolume matches actual observer byte values; original one microstep/two material solves, earlier discarded outer trial excluded.' if name.startswith('phase-') else None,",'')
assert hashlib.sha256(s.encode()).hexdigest()=='78fb5bc63b799a103f1dcdf6370b4f70bca7d8a51aa1dfc7eb3c818e3d685349';p=D/'inputs/r4_physical_audit-S407.py';p.write_text(s);versions.append({'path':str(p.relative_to(D)),'sha256':sha(p),'matches_original_executed_spec':True})
patch=D/'patches/S408-select-actual-committed-geometry.patch';patch.write_text(''.join(difflib.unified_diff(s.splitlines(True),(P/'r4_physical_audit.py').read_text().splitlines(True),fromfile='a/test/r4_physical_audit-S407.py',tofile='b/test/r4_physical_audit-S408.py')))
(D/'metadata/analysis-input-versions.json').write_text(json.dumps(versions,indent=2)+'\n')
for p in (D/'artifacts').glob('*-D402-SST-physical.json'):
 q=json.loads(p.read_text());q['weighted_gross_scope']='Independent evaluated source algebra: omega sums include constrained wall rows whose RHS is suppressed. They are not applied whole-domain omega inventory or full macro closure.';q['floor_branch_scope']='All actual active_floors counts are0; only inactive-floor recovery is exercised, not an activated floor branch.';p.write_text(json.dumps(q,indent=2)+'\n')
p=D/'REPORT.md';s=p.read_text();s=s.replace('selected数值壁owner、graph/rollback、完整macro constraints和三应用gross仍开放','floor全部未激活；weighted omega gross包含被约束/实际RHS被抑制的行，仅证明源代数，不是应用库存。selected数值壁owner、graph/rollback、完整macro constraints和三应用gross仍开放');s=s.replace('模块global符号memKind5/correlation0事件','64个Device Static模块符号memKind5/correlation0事件');p.write_text(s)
(D/'metadata/S405-analysis-failure-history.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','case':'FSH-active16-ASCII-S306','actual_GPU_receipt':'GPU-FSH-active16-ASCII-S306','initial_analysis_failure':'KeyError pThermalEnthalpyResidual after actualGPUcompleted','classification':'Hostsupportanalysis/schemaassumption, notGPUfailure','original_rule_provenance':'Previous0aa FSH64 caloricprofile helper uses get(field, zeros); physicalfield absent inoriginalFSH64 schema','remediation':'Onlyexistingoutput reanalyzed underoriginaldoubleprofile rule; noGPU rerun orproductionchange','full_initial_orchestration_stdout_available_in_delivery':False,'limitation':'Original exception preserved as fact from session; full initial orchestration stdout was not captured by subprocess runner, so no invented original log.'},indent=2)+'\n')
# Do not publish executable artifacts. Preserve the exact binaries in the authorized local test cache.
cache=ROOT/'work/fast4-raw-cache';cache.mkdir(exist_ok=True);omitted=[]
for p in D.rglob('*'):
 if not p.is_file():continue
 if p.suffix=='.pyc':
  assert p.resolve().is_relative_to(D.resolve());p.unlink();continue
 with p.open('rb') as h:magic=h.read(4)
 if magic==b'\x7fELF':
  dest=cache/'executables'/p.relative_to(D);dest.parent.mkdir(parents=True,exist_ok=True);assert dest.resolve().is_relative_to(ROOT.resolve());omitted.append({'original_delivery_path':str(p.relative_to(D)),'sha256':sha(p),'bytes':p.stat().st_size,'local_cache':str(dest),'reason':'Executable/object evidence hashes andrebuildcommands retained; no enormousbinaries inPR'});shutil.move(p,dest)
(D/'metadata/local-only-binaries.json').write_text(json.dumps(omitted,indent=2)+'\n')
# Content-address immutable evidence already uploaded on this PR; never hydrate or duplicate its historical owner paths.
m=json.loads((ROOT/'review/pr14/manifest.json').read_text());bysha={q['sha256']:q for q in m['evidence']};refs=[];checked={}
for p in sorted(D.rglob('*')):
 if not p.is_file() or p.relative_to(D).parts[0] not in ['inputs','artifacts']:continue
 h=sha(p);e=bysha.get(h)
 if not e:continue
 name='review/pr14/'+e['path']
 if name not in checked:
  b=subprocess.check_output(['git','-C',str(ROOT),'show',PATHS['delivery_parent']+':'+name]);assert hashlib.sha256(b).hexdigest()==h;checked[name]=True
 refs.append({'alias':str(p.relative_to(D)),'fixed_commit':PATHS['delivery_parent'],'repo_path':name,'sha256':h,'bytes':p.stat().st_size,'evidence_id':e['id']});p.unlink()
(D/'metadata/immutable-references.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','references':refs,'all_fixed_blobs_verified':True,'bytes_not_duplicated':sum(q['bytes'] for q in refs)},indent=2)+'\n')
print('PACK_IMMUTABLE_REFS',len(refs),sum(q['bytes'] for q in refs),flush=True)
packed=[]
for p in sorted(D.rglob('*')):
 if not p.is_file() or p.stat().st_size<524288 or p.suffix=='.gz' or p.name in ['results.json','REPORT.md','TEST_PLAN.md','SOURCE.json','coverage.json','findings.json','VALIDATION.json']:continue
 if p.relative_to(D).parts[0] not in ['logs','artifacts']:continue
 original=sha(p);size=p.stat().st_size;dest=p.with_name(p.name+'.gz')
 with p.open('rb') as src,dest.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as h:shutil.copyfileobj(src,h)
 digest=hashlib.sha256()
 with gzip.open(dest,'rb') as h:
  while b:=h.read(1048576):digest.update(b)
 assert digest.hexdigest()==original
 receipt={'alias':str(p.relative_to(D)),'stored_path':str(dest.relative_to(D)),'original_sha256':original,'original_bytes':size,'compressed_sha256':sha(dest),'compressed_bytes':dest.stat().st_size,'decompressed_bytes_SHA256_verified':True};packed.append(receipt)
 saved=cache/'payloads'/p.relative_to(D);saved.parent.mkdir(parents=True,exist_ok=True);assert saved.resolve().is_relative_to(ROOT.resolve());shutil.move(p,saved)
 if len(packed)%8==0:print('PACK_PROGRESS',len(packed),sum(q['original_bytes'] for q in packed),sum(q['compressed_bytes'] for q in packed),flush=True)
(D/'metadata/lossless-storage.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','entries':packed,'raw_local_cache':str(cache/'payloads'),'all_decompressed_hashes_verified':True},indent=2)+'\n')
compressed={q['alias']:q for q in packed};r=json.loads((D/'results.json').read_text())
for q in r['tests']:
 if q['log'] in compressed:
  entry=compressed[q['log']];assert q['log_sha256']==entry['original_sha256'];q['raw_log_alias']=q['log'];q['raw_log_sha256']=q['log_sha256'];q['log']=entry['stored_path'];q['log_sha256']=entry['compressed_sha256'];q['log_storage']='LOSSLESS_GZIP_RAW_DECOMPRESSED_SHA_VERIFIED'
 q['artifacts']=[compressed.get(n,{}).get('stored_path',n) for n in q.get('artifacts',[])]
 q['environment_reference']='environment.json'
for u in r['acceptance_units']:u['evidence_paths']=[compressed.get(n,{}).get('stored_path',n) for n in u['evidence_paths']]
(D/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
f=json.loads((D/'findings.json').read_text())
for q in f['findings']:q['evidence']=[compressed.get(n,{}).get('stored_path',n) for n in q['evidence']]
(D/'findings.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n')
print('PACK_DONE',len(packed),'losslesspayloads',sum(q['original_bytes'] for q in packed),'->',sum(q['compressed_bytes'] for q in packed),'bytes',flush=True)
