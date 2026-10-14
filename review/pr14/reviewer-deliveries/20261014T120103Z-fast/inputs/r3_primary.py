from r3_runner import *
import re,csv,sys
oldtests=json.loads((OLD/'results.json').read_text())['tests']
def prepare_case(name,oldname=None,kind='extra'):
 oldname=oldname or name;src=OLDROOT/'work/cases'/kind/oldname;case=ROOT/'work/cases'/kind/name
 assert src.is_dir() and not case.exists();shutil.copytree(src,case,ignore=shutil.ignore_patterns('chmtOutput','core','core.*'))
 assert case.resolve().is_relative_to(Path(PATHS['repo']).resolve())
 return case
def inputs(case):return [p for p in case.rglob('*') if p.is_file() and 'chmtOutput' not in p.parts]
def match_prior_inputs(case,tid,prefix):
 q=next(x for x in oldtests if x['id']==tid);rows=[]
 for original,digest in q['input_hashes'].items():
  if '/'+prefix+'/' not in original:continue
  rel=original.split('/'+prefix+'/',1)[1];p=case/rel
  if not p.is_file():
   payload=old_read(prefix+'/'+rel);assert hashlib.sha256(payload).hexdigest()==digest;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
  assert p.is_file() and sha(p)==digest,(rel,sha(p) if p.is_file() else None,digest)
  rows.append({'path':rel,'sha256':digest,'prior_original_path':original,'fixed_result_commit':PATHS['old_delivery_commit']})
 assert rows
 (D/'metadata'/('same-input-'+case.name+'.json')).write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','prior_test':tid,'all_original_recorded_physical_input_hashes_equal':True,'original_result_commit':PATHS['old_delivery_commit'],'files':rows,'additional_mesh_hashes':{str(p.relative_to(case)):sha(p) for p in inputs(case) if 'polyMesh' in p.parts},'controlDict':(case/'system/controlDict').read_text(),'parameters':(case/'constant/chmtProperties').read_text()},indent=2)+'\n')
 return rows
def check(case,name):
 checker=ROOT/'applications/CHMT/verification/coupled_runtime/check_case.py'
 t=run('ORIGINAL-CHECK-'+name,cmd('python3',checker,case),[3,5,6,10,14,26,29],300,'Originalcurrent physicalcheck foractualoriginalinput; fullM/E/P independent audit separate.',inputs=[checker,case/'verification.json'],kind='ANALYSIS');assertion(t,t['exit_code']==0,(D/t['log']).read_text());return t
def execute(case,name,budget=300,extra=(),ns=2,kind='GPU',prior=None,prefix=None):
 b=ROOT/f'work/install/ns{ns}/bin/CHMT';assert b.is_file()
 if prior:match_prior_inputs(case,prior,prefix)
 t=run(name,cmd(*extra,b,'-case',case),[3,5,6,8,10,13,14,20,21,26,28,29],budget,'Formalnative source; exactoriginalphysicalinput/end/flags; finalnumeric/ledgeracceptance separatelyverified.',inputs=inputs(case),binary=b,kind=kind,progress_dirs=[case/'chmtOutput']);out=archive_output(case,name)
 if (case/'chmtOutput').is_dir():check(case,name)
 if t['external_timeout']:amend(t['id'],verdict='BLOCKED',observations='Finiteexternalbudget exhausted; preservepartialstates/progress andabsenceoffinalcompletion. Not a numeric pass/failure.')
 else:amend(t['id'],observations='Actualnative exit recorded; complete global/perrecipientM/E/P and accepted timeline audit pending.',artifacts=[str(out.relative_to(D))])
 return t
mode=sys.argv[1]
if mode=='primary':
 fixture=ROOT/'work/fixtures/chemistry-trial-formal';b=fixture/'chemistry_trial';assert b.is_file()
 t=run('CHEMISTRY-FORMAL-FIXTURE',cmd(b),[10,12,14,26],300,'NoS202/S203/S208overlay.16failedcellcases,actualCo owner/fullacceptednativepartition,exhaustedterminal,requested1e-6/direct1e-6publiccontrols; fullsemanticpositiveaudit/budgets.',inputs=[fixture/'chemistry_trial.cu',fixture/'command.json',ROOT/'tests/fixtures/chemistry_trial/chemistry_trial.cu.in'],binary=b)
 raw=(D/t['log']).read_text();kinds=list(map(int,re.findall(r'PASS failed-cell kind=(\d+)',raw)));marks=[s for s in ['PASS requested owner accepted trace','PASS terminal owner completion','PASS original successful requested interval','PASS original direct chemistry success'] if s in raw]
 accepts=[{'dt_hex':a,'stamp_hex':z} for a,z in re.findall(r'OWNER_ACCEPT dt=(\S+) stamp=(\S+)',raw)]
 assessment={'raw_exit':t['exit_code'],'failed_kind_sequence':kinds,'all16markers':kinds==list(range(16)),'four_owner_success_markers':marks,'accepted_trace':accepts,'actual_full_semantic_assertions_executed':t['exit_code']==0,'scope_limit':'Failure-specific six-field status evidence and transaction coverage separately inspected, strings alone notfullacceptance. Normalcleanup only, notgeneralresourcefault recovery.'}
 (D/'artifacts/chemistry-fixture-summary.json').write_text(json.dumps(assessment,indent=2)+'\n')
 assertion(t,t['exit_code']==0 and kinds==list(range(16)) and len(marks)==4 and bool(accepts),'FormalfixtureactualCUDA complete; fullsemanticpositivecomparisons executed and16failuremarkers. Negative-specificsemantic/status coverage remains separatelyqualified.',artifacts=['artifacts/chemistry-fixture-summary.json'])
 print(raw[-11000:],flush=True)
if mode in ['primary','wet']:
 case=ROOT/'work/cases/extra/wet-film-formal'
 if not case.exists():case=prepare_case('wet-film-formal','wet-film-S209')
 execute(case,'WET-FILM-FORMAL',prior='EXTRA-wet-film-S209',prefix='inputs/native/extra/wet-film-S209')
 cp=case/'chmtOutput/checkpoint-1';assert cp.is_dir()
 restart=prepare_case('wet-film-formal-restart','wet-film-S209');b=ROOT/'work/install/ns2/bin/CHMT'
 with (restart/'constant/chmtProperties').open('a') as h:h.write('\nrestartDirectory "'+str(cp)+'";\n')
 t=run('WET-FILM-FORMAL-RESTART',cmd(b,'-case',restart),[3,5,6,13,14,20,21,26,29],300,'Sameoriginalwetphysicalinputfromactualcheckpoint1; exactterminalstate/consumer comparisonandfirstdivergence independentlyrecorded.',inputs=inputs(restart)+[cp/'state.chmt'],binary=b,progress_dirs=[restart/'chmtOutput']);archive_output(restart,t['id']);check(restart,t['id'])
 case=prepare_case('wet-evaporation-formal-900','wet-evaporation-S209');execute(case,'WET-EVAPORATION-FORMAL-900',900,prior='EXTRA-wet-evaporation-S209-D204',prefix='inputs/native/extra/wet-evaporation-S209')
elif mode=='initcheck':
 case=prepare_case('chemistry-full-initcheck-1800','chemistry-full-initcheck','chmt');match_prior_inputs(case,'CHMT-chemistry-full-initcheck','inputs/native/chmt/chemistry-full')
 control=(case/'system/controlDict').read_text()
 with (D/'TEST_PLAN.md').open('a') as h:h.write('\nNs10扩大预算执行前实际完整controlDict核对：\n```\n'+control+'\n```\n原所有记录输入SHA完全一致，原基准source4b普通运行/802比较复用，1800s外部initcheck新执行须finalsummary+原checker；不改变physicalproblem。先行计划中的0.0007只为占位，完整终点以这里原controlDict为准。\n')
 execute(case,'NS10-REACTING-FULL-INITCHECK-1800',1800,['compute-sanitizer','--tool','initcheck','--target-processes','all','--error-exitcode','86'],10,'SANITIZER')
elif mode not in ['primary','wet']:raise SystemExit(mode)
