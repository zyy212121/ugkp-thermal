from pathlib import Path
import hashlib,json,subprocess
root=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013')
out=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/implementation/frontend-terminal')
base='55b7192cf269f06017d5286d302c1a0757e8a39c'
paths=['applications/CHMT/ablation/InterfaceMath.H','applications/CHMT/configuration/InitialConditions.H','applications/CHMT/gpu/StateValidation.H','applications/CHMT/io/CoupledOutput.H','applications/CHMT/io/MeshIO.H','applications/CHMT/materials/GasThermoAdapter.H','applications/CHMT/materials/MaterialTransport.C','applications/CHMT/materials/Reaction.H','applications/CHMT/materials/Thermodynamics.H','applications/CHMT/mesh/EmptyExtrusion.H','applications/CHMT/mesh/Geometry.C','applications/CHMT/mesh/Geometry.H','applications/gasUGKP/gpu/BoundaryLayerInput.H','applications/gasUGKP/gpu/SharedGasModelInput.H','common/gasTransport/GasModelIO.H']
records={}
for path in paths:
 data=(root/path).read_bytes();old=subprocess.check_output(['git','show',base+':'+path],cwd=root)
 for label,contents in [('source',data),('original',old)]:
  p=out/label/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(contents)
 records[path]={'original_sha256':hashlib.sha256(old).hexdigest(),'current_sha256':hashlib.sha256(data).hexdigest()}
patch=subprocess.check_output(['git','diff',base,'--',*paths],cwd=root);(out/'frontend-terminal.patch').write_bytes(patch)
inputs={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for prefix in ['compile','fixtures'] for p in (out/prefix).glob('*') if p.is_file()}
(out/'freeze.json').write_text(json.dumps({'base':base,'files':records,'patch_sha256':hashlib.sha256(patch).hexdigest(),'inputs_and_objects':inputs,'runtime_execution_after_scope_change':False,'compiled_only':['OF10 combined real frontend headers and EmptyExtrusion specializations','actual Geometry.C','actual MaterialTransport.C','nvcc sm_80 caloric/interface CHMT_HD instantiations'],'link_or_full_application_build':False,'gpu_available':False},indent=2)+'\n')
