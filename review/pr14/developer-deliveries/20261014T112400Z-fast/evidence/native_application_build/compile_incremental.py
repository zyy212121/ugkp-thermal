import pathlib,json,hashlib,subprocess,datetime,shlex
root=pathlib.Path('/workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014')
out=pathlib.Path('/tmp/pr14_fast3_compile')
prior=json.loads(pathlib.Path('/workspace/scratch/98b6339ec81b/pr14_fast_compile_delivery_20261014/compile_receipts.json').read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
records=[]
for ns in (2,10):
 s=next(s for s in prior['scopes'] if s['name']==f'CHMT native solver Ns={ns}')
 obj=pathlib.Path(f'/tmp/pr14_compile_checks/chmt_native_ns{ns}/Backend.o')
 assert sha(obj)==s['backend_object_sha256'], 'old CUDA object changed'
 unchanged=[]
 for p in sorted(list((root/'common').rglob('*'))+list((root/'applications/CHMT').rglob('*'))):
  if not p.is_file():continue
  rel=p.relative_to(root).as_posix()
  if rel=='applications/CHMT/materials/CpuMaterialDriver.C':continue
  try:old=subprocess.check_output(['git','show',f'4b5c480b572f234ddecf5e89164fe59b0d2e9735:{rel}'],cwd=root,stderr=subprocess.DEVNULL)
  except subprocess.CalledProcessError:continue
  assert p.read_bytes()==old, 'CUDA dependency region changed: '+rel
  unchanged.append({'path':rel,'sha256':sha(p)})
 build=out/f'ns{ns}';gen=build/'generated';gen.mkdir(parents=True,exist_ok=True)
 prep=['python3','applications/CHMT/devtools/build_info.py','--species-count',str(ns),'--arch','sm_89','--nvcc','/tmp/pr14_cuda13/nvidia/cu13/bin/nvcc','--output',str(gen)]
 cmd=s['commands'][1].copy();cmd=[str(gen) if x==f'/tmp/pr14_compile_checks/chmt_native_ns{ns}/generated' else x.replace(f'-I/tmp/pr14_compile_checks/chmt_native_ns{ns}/generated',f'-I{gen}') for x in cmd]
 cmd[-1]=str(build/'CHMT')
 source_paths=[pathlib.Path(x) for x in cmd if x.endswith('.C')]
 before={str(p.relative_to(root)):sha(p) for p in source_paths}
 started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 env_script='/workspace/scratch/98b6339ec81b/openfoam10_cpu/environment.sh'
 shell='set +e; source '+shlex.quote(env_script)+'; set -e; unset FOAM_USER_APPBIN; '+shlex.join(prep)+'; '+shlex.join(cmd)
 (build/'command.sh').write_text(shell+'\n')
 with (build/'build.log').open('w') as log: result=subprocess.run(['bash','-c',shell],cwd=root,stdout=log,stderr=subprocess.STDOUT)
 after={str(p.relative_to(root)):sha(p) for p in source_paths}
 assert before==after,'host sources changed during build'
 rec={'scope':f'CHMT Ns={ns} native host compilation and final application link','started':started,'finished':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':result.returncode,'commands':[prep,cmd],'host_source_sha256':before,'gpu_runtime_executed':False,'cuda_object_reused':str(obj),'cuda_object_sha256':sha(obj),'unchanged_cuda_region':unchanged,'binary_sha256':sha(build/'CHMT') if result.returncode==0 else None}
 records.append(rec);(out/'receipts.json').write_text(json.dumps(records,indent=2)+'\n')
 print(rec['scope'], 'exit=',result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
