from r4_runner import *
app='FSH';base=ROOT/'work/cases/FSH-active16-ASCII-S306';oldversions=json.loads(old_read('versions.json'))
oldfront=OLDROOT/'work/install/ns2/bin/FSH';oldback=OLDROOT/'work/install/ns2/bin/FSHCudaBackend';newfront=ROOT/'work/install/ns2/bin/FSH';newback=ROOT/'work/install/ns2/bin/FSHCudaBackend'
assert sha(oldfront)==oldversions['actual_binaries_objects'][str(oldfront.relative_to(OLDROOT))];assert sha(oldback)==oldversions['actual_binaries_objects'][str(oldback.relative_to(OLDROOT))]
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nV1双向真实endpoint限定补测：使用实际旧9d交付所验证FSH frontend/backend与本轮470新端点，原16active cold2D输入/full原1e-5终点，原native flags/runtime，所有原物理文件bytes对比。旧前端实际binary与新前端hash相同如实标明（旧frontend生产依赖未改）；backend哈希不同。只闭合该原Legacy模式热/边界/profile路径；不证明未执行的非零多species混合wire/对应gasUGKP双向互通，不把双新host往返当对端验证。\n')
receipts=[]
for direction,front,back in [('old-to-new',oldfront,newback),('new-to-old',newfront,oldback)]:
 case=ROOT/'work/cases'/('FSH-wire-'+direction);case.mkdir()
 for n in ['0','constant','system']:shutil.copytree(base/n,case/n)
 t=run('GPU-WIRE-FSH-'+direction,cmd('env',f'FSH_CUDA_BACKEND={back}',front,'-case',case),[1,5,12,18,19,23,26,28,33],expected='Real opposite-version private endpoint; full original cold2D input/end, all actual output files physical bytes identical.',inputs=[p for n in ['0','constant','system'] for p in (case/n).rglob('*') if p.is_file()],binary=front)
 rows=[]
 for folder in base.iterdir():
  try:stamp=float(folder.name)
  except ValueError:continue
  if stamp==0 or not folder.is_dir():continue
  for p in folder.rglob('*'):
   if p.is_file():rows.append({'path':str(p.relative_to(base)),'formal_sha256':sha(p),'paired_sha256':sha(case/p.relative_to(base)),'byte_equal':p.read_bytes()==(case/p.relative_to(base)).read_bytes()})
 receipt={'direction':direction,'old_source_evidence_commit':PATHS['old_delivery_commit'],'old_source_production':PATHS['old_source'],'frontend_path':str(front),'frontend_sha256':sha(front),'backend_path':str(back),'backend_sha256':sha(back),'old_frontend_and_new_frontend_binary_identical':sha(oldfront)==sha(newfront),'all_actual_output_bytes_equal':all(q['byte_equal'] for q in rows),'physical_files':rows,'scope':'Actual old/new endpoints, original FSH Legacy-mode heat/boundary/profile case only. Nonzero mixture species and gasUGKP opposite endpoints NOT_RUN.'};receipts.append(receipt)
 artifact=D/'artifacts'/('FSH-wire-'+direction+'.json');artifact.write_text(json.dumps(receipt,indent=2)+'\n');assertion(t,t['exit_code']==0 and receipt['all_actual_output_bytes_equal'],'Actual endpoint same output bytes; nonzero species/gasUGKP scope open.',artifacts=[str(artifact.relative_to(D))],paired_backend_sha256=sha(back),paired_backend_source=PATHS['old_source'] if direction=='new-to-old' else SOURCE)
