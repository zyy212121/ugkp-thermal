from pathlib import Path
import hashlib,json,subprocess,re,shutil
root=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013');out=Path(__file__).parent
names='''GpuAutomaticCsrSchedule.cuh GpuCollisionPoolCorrection.cuh GpuCollisionPoolSampling.cuh GpuMobilePackingHost.cuh GpuParticlePayload.cuh GpuParticleThermalCarry.H GpuParticleThermalConsumers.H GpuParticleTransport.cuh GpuPressurePipeline.cuh GpuReductionTaskMaterialize.cuh GpuReductionTaskWrite.cuh GpuThermalParticleFinalization.cuh operators/FiniteContactRelaxation.inl operators/ParticleDragMomentum.inl operators/accumulateParticleWallRepresentedContactAreaKernel.cuh operators/gatherSelectedParticlesKernel.cuh operators/rebuildWallBoundParticleDirectoryKernel.cuh wall/GpuAluminaProperties.H wall/GpuCapillaryDetachment.H wall/GpuFiniteWallContact.H wall/GpuParticleWallContactHeat.H wall/GpuSommerfeldSticking.H wall/GpuWallInterfaceHeat.H'''.split()
paths=['common/'+n for n in names]+['tests/test_particle_routing_continuation.py']
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=[]
for name in paths:
 p=root/name;b=p.read_bytes();before=subprocess.run(['git','show','55b7192:'+name],cwd=root,capture_output=True)
 target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
 manifest.append({'path':name,'sha256':sha(b),'bytes':len(b),'baseline_sha256':sha(before.stdout) if before.returncode==0 else None})
(out/'source-manifest.json').write_text(json.dumps(manifest,indent=2));(out/'owned-paths.txt').write_text('\n'.join(paths)+'\n')
patch=subprocess.check_output(['git','diff','55b7192','--',*paths],cwd=root)
(out/'particle-terminal.patch').write_bytes(patch)
# Conservative literal quoted-include closure of all real CUDA compile inputs.
seen=set();todo=list((out/'cuda').glob('*.cu'));records=[]
while todo:
 p=todo.pop().resolve()
 if p in seen:continue
 seen.add(p);b=p.read_bytes();records.append({'path':str(p),'sha256':sha(b),'mtime_ns':p.stat().st_mtime_ns})
 for inc in re.findall(r'^\s*#\s*include\s*"([^"]+)"',b.decode(),re.M):
  for base in [p.parent,root/'common',root/'common/wall']:
   child=base/inc
   if child.is_file():todo.append(child);break
(out/'cuda/source-closure.json').write_text(json.dumps(records,indent=2))
first=min(p.stat().st_mtime_ns for p in (out/'cuda').glob('*.log'))
print('owned',len(paths),'closure',len(records),'dependencies newer than first compile log:',[r['path'] for r in records if r['mtime_ns']>first])
