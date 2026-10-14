import ast,shutil,json,subprocess
from r2_runner import *
inp=D/'inputs/fixtures';inp.mkdir(parents=True,exist_ok=True)
# Preserve generated translation units and exact commands; production sources are unchanged.
p=ROOT/'tests/fixtures/chemistry_trial/prepare_chemistry_trial.py';out=ROOT/'work/fixtures/chemistry'
t=run('PREP-CHEM-OWNER',cmd('python3',p,ROOT,out,'--generate-only'),[10,14],expected='Generate fixture including actual native CUDA translation unit and requested owner.',inputs=[p,ROOT/'tests/fixtures/chemistry_trial/chemistry_trial.cu.in'],kind='SETUP')
assert t['exit_code']==0
shutil.copy2(out/'chemistry_trial.cu',inp/'chemistry_trial.cu');args=json.loads((out/'command.json').read_text());shutil.copy2(out/'command.json',inp/'chemistry-command.json')
t=run('BUILD-CHEM-OWNER',cmd(*args),[10,14,26],900,'Native Ns2 owner fixture link; includes unchanged actual production CUDA.',inputs=[inp/'chemistry_trial.cu'],kind='BUILD')
if t['exit_code']==0:
 t=run('CHEM-OWNER',cmd(out/'chemistry_trial'),[10,14,26],expected='Actual StepLimit→rollback→two accepted native halves cover4e-6; exhausted effort completes public terminal; full fields/audits/ledger match; 16 failed-resource/metadata combinations.',binary=out/'chemistry_trial',inputs=[inp/'chemistry_trial.cu'])
 raw=(D/t['log']).read_text();assertion(t,'PASS real CUDA chemistry Trial routing' in raw and 'FAIL ' not in raw,raw)
# Build two full native cold fixture translation units with application-specific flags.
for app in ['CHT','FSH']:
 p=ROOT/'tests/fixtures/pr14_fast_cold/run.py';out=ROOT/'work/fixtures'/('cold-'+app)
 t=run('BUILD-COLD-FIXTURE-'+app,cmd('python3',p,ROOT,out,app,'--compile-only'),[12,18,23,26,33],900,'Complete production CUDA unit plus actual cold initializers/configuration; native fmad flag.',inputs=[p,ROOT/'tests/fixtures/pr14_fast_cold/initialization.cu'],kind='BUILD')
 for name in ['command.txt','build.log']:
  if (out/name).exists():shutil.copy2(out/name,D/'metadata'/('cold-'+app+'-'+name))
 if t['exit_code']:continue
 t=run('COLD-FIXTURE-'+app,cmd(out/'initialization'),[12,18,23,26,33],expected='Four real initialized profiles/phase regions/nonzero carry; CHT configure/patch order/mixed/toggle retains actual downloaded face values; FSH exact double template.',binary=out/'initialization',inputs=[ROOT/'tests/fixtures/pr14_fast_cold/initialization.cu'])
 raw=(D/t['log']).read_text();assertion(t,'PASS actual CUDA cold-wall initialization' in raw and 'FAIL ' not in raw,raw,limits=['Initializer/configuration witness; application active mixed-face heat budget and actual copy failure recovery separate.'])
# Extract exactly the production reduction operators, then exercise their GPU algorithms.
p=ROOT/'tests/gas_transport/test_chmt_courant_reduction_cuda.py';tree=ast.parse(p.read_text());strings={}
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ['PREAMBLE','PROBE'] for x in node.targets):strings[node.targets[0].id]=ast.literal_eval(node.value)
backend=(ROOT/'applications/CHMT/gpu/Backend.cu').read_text();start=backend.index('struct CoupledCourantMaximum {');end=backend.index('static bool currentGasCourant(',start)
cu=inp/'courant_reduction.cu';cu.write_text(strings['PREAMBLE']+backend[start:end]+strings['PROBE'])
binary=ROOT/'work/fixtures/courant_reduction';binary.parent.mkdir(parents=True,exist_ok=True)
t=run('BUILD-CO-FIXTURE',cmd('nvcc','-std=c++17','-O2','-arch=sm_89','-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),cu,'-o',binary),[9,12,14,15,26,28],900,'Production16-byte Co/read/reduce/max and production owner allocator; GPU fixture.',inputs=[cu,p],kind='BUILD')
if t['exit_code']==0:
 t=run('CO-FIXTURE',cmd(binary),[9,12,14,15,26,28],expected='Empty/single/31/32/33/513/32768, finite/signedzero/NaN bit-preserving equivalence on default/nonblocking streams,100 repeats each; actual distinct active allocations,capacity growth,reuse,real huge allocation error/retry/destruction.',inputs=[cu],binary=binary)
 raw=(D/t['log']).read_text();assertion(t,'CUDA Courant reuse/equivalence PASS' in raw,raw,limits=['Original fixture repeats unchanged data; separate changing-data same-owner recheck required.'])
