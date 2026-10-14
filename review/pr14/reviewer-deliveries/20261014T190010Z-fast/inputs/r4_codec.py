from r4_runner import *
source=D/'inputs/checkpoint-full-wet-observer.cpp';source.write_bytes(old_read('inputs/checkpoint-full-wet-observer.cpp'))
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nS404实际checkpoint decoder：复用固定0aa/9d已记录只读完整physical字段源，新编最终Checkpoint.C/HostState接口ASan/UBSan，解析新GPU产物。原格式兼容Budget/ChemistryAudit若打印只标明UNCOLECTED，不作物理预算；完整gas/solid/pore/film/particle/geometry/packet字段从真实checkpoint取。纯hostcodec不替代GPU物理验收。\n')
app=ROOT/'applications/CHMT';b=ROOT/'work/fixtures/checkpoint-S404';b.parent.mkdir(parents=True,exist_ok=True)
t=run('BUILD-CHECKPOINT-S404-ASAN',cmd('g++','-std=c++17','-O1','-g','-DUGKWP_GAS_SPECIES=2','-fsanitize=address,undefined','-fno-omit-frame-pointer','-I'+str(app),'-I'+str(ROOT/'common'),source,app/'restart/Checkpoint.C','-o',b),[5,6,13,19,20,26,28],expected='Actualnativecodec/physicalfield output onnewfixed-sourceGPU states, hostASan/UBSan, noCUDA surrogate.',inputs=[source,app/'restart/Checkpoint.C'],kind='BUILD',variant='SUPPORT_S404')
assert t['exit_code']==0;amend(t['id'],binary=str(b),binary_sha256=sha(b))
for name in ['thermal-full','wet-film','SST-RK','SST-SSPRK2','SST-SSPRK3','moving-GCL','phase-melting','phase-solidification']:
 case=ROOT/'work/cases'/name
 for cp in sorted((case/'chmtOutput').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1])):
  q=run('DECODE-S404-'+name+'-'+cp.name,cmd(b,cp),[5,6,13,19,20,26,28],expected='ReadactualGPUstatewithactualnativecodec/ASanUBSan; decode success only not conservation.',inputs=[cp/'state.chmt',source],binary=b,kind='HOST_SANITIZER',variant='SUPPORT_S404')
  if q['exit_code']==0:
   data=json.loads((D/q['log']).read_text());out=D/'artifacts'/name/(cp.name+'-physical-decoded.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(data,indent=2)+'\n');assertion(q,True,'Actualnativecodec decoded storedfieldswithoutobservedASan/UBSanerror; legacybudget/auditUNCOLECTED andnotphysicalmeasurement.',artifacts=[str(out.relative_to(D))])
