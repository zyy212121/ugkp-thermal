from pathlib import Path
import sys,runpy,subprocess,json,hashlib
repo=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_work_20261013');out=Path('/workspace/scratch/98b6339ec81b/chmt_remaining_audit_20261013/implementation/b7');sys.path.insert(0,str(repo/'applications/CHMT/tests'));fixture=runpy.run_path(str(repo/'applications/CHMT/tests/test_shared_reconfiguration.py'))
body=fixture['SOURCE'].split(' chmt::SharedGasDeviceStorage storage;')[0]+r'''
 gas.mode=ugkwp::GasMode::MixtureChemistry;
 {chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));
 size_t resident=0;for(const auto& p:live)resident+=p.second;std::printf("resident allocations=%zu bytes=%zu chemistryStatusBytes=%zu ownerBytes=%zu\n",live.size(),resident,sizeof(ugkwp::ChemistryStatus),sizeof(storage));
 allocs=frees=zeros=syncs=0;zeroBytes=h2d=d2h=d2d=0;assert(storage.clearTrialStatus());std::printf("clear zero_calls=%d zero_bytes=%zu\n",zeros,zeroBytes);
 assert(storage.checkStatus(error));std::printf("check D2H_bytes=%zu sync=%d allocations=%d frees=%d\n",d2h,syncs,allocs,frees);
 }assert(live.empty());}
'''
prior=json.loads((out/'measured-costs.json').read_text());records=[]
for name,tree in [('base-5f',out/'baseline'),('working',repo)]:
 if name=='base-5f':
  records.append(prior[0]);continue
 p=out/'tmp'/('cost-final-'+name);p.mkdir(exist_ok=True);(p/'cuda_runtime.h').write_text(fixture['COUNTERS']);(p/'source.cpp').write_text(body)
 cmd=['g++','-std=c++17','-O1','-I'+str(p),'-I'+str(tree/'applications/CHMT'),'-I'+str(tree),'-I'+str(tree/'common'),str(p/'source.cpp'),'-o',str(p/'probe')]
 r=subprocess.run(cmd,capture_output=True,text=True);(p/'build.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 r=subprocess.run([str(p/'probe')],capture_output=True,text=True);(p/'run.log').write_text(r.stdout+r.stderr);print(name,r.stdout);assert r.returncode==0,r.stderr
 records.append({'name':name,'command':cmd,'result':r.stdout,'source_sha256':hashlib.sha256(body.encode()).hexdigest()})
(out/'measured-costs.json').write_text(json.dumps(records,indent=2)+'\n')
