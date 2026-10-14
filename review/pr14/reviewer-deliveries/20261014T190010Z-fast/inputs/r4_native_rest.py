from r4_runner import *
for app,script in [('gasUGKP','private_backend/build_private_backend.sh'),('FSH','private_backend/build_private_backend.sh'),('CHT','tools/build_cuda_solver.sh')]:
 p=ROOT/'applications'/app/script
 t=run('BUILD-NATIVE-'+app,cmd('env','UGKWP_CUDA_ARCH=sm_89','UGKP_DEVELOPMENT_PROBES=0','UGKWP_GAS_SPECIES=2','bash',p),[4,9,10,12,17,18,19,20,23,24,25,26,28,33],expected='Originalnative build script, O3andoriginalfmadpolicy, finalshared/common code; numericalrun separate.',inputs=[p],kind='BUILD',estimate_seconds=600)
 if t['exit_code']==0:
  b=ROOT/'work/install/ns2/bin'/app;amend(t['id'],binary=str(b),binary_sha256=sha(b));print('NATIVE_READY',app,sha(b),flush=True)
fixture=D/'fixtures/chemistry-formal'
t=run('BUILD-CHEMISTRY-FORMAL',cmd('python3',ROOT/'tests/fixtures/chemistry_trial/prepare_chemistry_trial.py',ROOT,fixture,'--nvcc','/usr/local/cuda/bin/nvcc','--arch','sm_89'),[10,12,14,26],expected='Unchanged formalgenerator O3 fmadtrue, noS301/S302 overlay, realnativeowner andactualCUDAkernels; numericalrun andmemcheck separate.',inputs=[ROOT/'tests/fixtures/chemistry_trial/prepare_chemistry_trial.py',ROOT/'tests/fixtures/chemistry_trial/chemistry_trial.cu.in'],kind='BUILD',estimate_seconds=600)
if t['exit_code']==0:
 b=fixture/'chemistry_trial';amend(t['id'],binary=str(b),binary_sha256=sha(b),generated_source_sha256=sha(fixture/'chemistry_trial.cu'))
 for name,argv,kind in [('CHEMISTRY-FORMAL-GPU',cmd(b),'GPU'),('CHEMISTRY-FORMAL-MEMCHECK',cmd('compute-sanitizer','--tool','memcheck','--error-exitcode','99',b),'SANITIZER')]:
  q=run(name,argv,[10,12,14,26],expected='Allformalassertions,16cell16transaction32five-memberFAIL_STATUS+2directpublicreplay andfouroriginalcontrols; scopegaps separately assessed.',inputs=[fixture/'chemistry_trial.cu',fixture/'command.json'],binary=b,kind=kind,estimate_seconds=300)
  raw=(D/q['log']).read_text();counts={k:raw.count(k) for k in ['PASS failed-cell kind=','PASS failed-transaction kind=','FAIL_STATUS kind=','PASS public-replay kind=','PASS requested owner accepted trace','PASS terminal owner completion','PASS original successful requested interval','PASS original direct chemistry success']}
  ok=q['exit_code']==0 and counts['PASS failed-cell kind=']==16 and counts['PASS failed-transaction kind=']==16 and counts['FAIL_STATUS kind=']==32 and counts['PASS public-replay kind=']==2 and all(counts[k]==1 for k in list(counts)[4:])
  if kind=='SANITIZER':ok=ok and 'ERROR SUMMARY: 0 errors' in raw
  assertion(q,ok,'Actualformalassertions andmarker matrix='+json.dumps(counts)+'; independentkind7/15 fullpublicowner replay andaggregateP absent fromformalfixture, supplementarytest required. Memchecknotgeneralallocation/free/sync recovery.',marker_counts=counts)
  print('FORMAL_CHEM',name,ok,counts,flush=True)
