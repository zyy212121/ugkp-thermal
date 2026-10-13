from pathlib import Path
import runpy,subprocess,json,hashlib
root=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013');out=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/implementation/b7');receipts=[]
test=(root/'applications/CHMT/tests/test_final_gas_consumers.py').read_text();setup=test[test.index(' p.enableSst=true;'):test.index(' wallDestination=v.gasBoundaryLayerModel.input;')]
body=setup+''' chmt::transport::GasCourantMaxima co;const bool ok=chmt::currentGasCourant(*b,.0001,co);
 std::cout.precision(17);std::cout<<"ok="<<ok<<" mass="<<v.gasPhiRho[wall]<<" prescribed="<<-.013<<" convective="<<co.convective<<" diffusion="<<co.diffusion<<" sst="<<co.sst<<" max="<<co.maximum<<" status="<<v.gasSpecies.faceStatus[wall]<<"\\n";
 assert(ok);assert(v.gasPhiRho[wall]==EXPECTED_MASS);
}\n'''
for tag,tree,expected in [('base-5f',out/'baseline',0),('working',root,-.013)]:
 folder=out/'tmp'/('cold-pinned-v2-'+tag);folder.mkdir(exist_ok=True)
 app=tree/'applications/CHMT';helper=runpy.run_path(str(app/'tests/test_coupled_runtime_transactions.py'))
 source=helper['prepare_sequential_launches'](folder)
 fixture=runpy.run_path(str(app/'tests/test_reacting_wall_storage.py'))['storage_source']();fixture=fixture[:fixture.index(' chmt::SharedGasDeviceStorage storage;')]
 source.write_text(source.read_text()+'\n#include <iostream>\n'+fixture+body)
 exe=folder/'probe';cmd=helper['HELPER']['compiler_flags'](folder)+['-I'+str(tree),'-DEXPECTED_MASS='+str(expected),str(source),str(app/'mesh/Geometry.C'),'-o',str(exe)]
 run=subprocess.run(cmd,capture_output=True,text=True);(folder/'build.log').write_text(run.stdout+run.stderr);assert run.returncode==0,run.stderr
 run=subprocess.run([str(exe)],capture_output=True,text=True);(folder/'run.log').write_text(run.stdout+run.stderr);print(tag,run.stdout,run.stderr);assert run.returncode==0
 receipts.append({'tag':tag,'command':cmd,'run':[str(exe)],'output':run.stdout,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'base':'5f1a39ac96e2e7ff9a0640a4db344d9cccd34245' if tag.startswith('base') else 'working'})
(out/'cold-pinned-comparison.json').write_text(json.dumps(receipts,indent=2)+'\n')
