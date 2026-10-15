"""Execute real wall restore and canonical owner helpers at CUDA boundaries."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import importlib.util
import subprocess
import re
import json
ROOT=SOURCE_ROOT
BASE='bcd63cf5a02f13eb7e5cd06ff87f552c3ffda4b4'
spec=importlib.util.spec_from_file_location('l1',(VALIDATION_ROOT / 'tests/test_gas_wall_ledger_lifecycle.py'));l1=importlib.util.module_from_spec(spec);spec.loader.exec_module(l1);h=l1.h


def code(revision=None):
    s=l1.native(revision,restore_stub=False)
    s=s.replace('#include <vector>','#include <vector>\n#include <map>\nstd::map<void*,size_t> live;int predicateChecks=0;',1)
    s=s.replace('*p=std::malloc(n);return *p?0:1;','*p=std::malloc(n);if(*p)live[*p]=n;return *p?0:1;')
    s=s.replace('int cudaFree(void* p){std::free(p);return 0;}','int cudaFree(void* p){if(p){assert(live.erase(p)==1);std::free(p);}return 0;}')
    s=s.replace('return value == 0.0;', 'return (++predicateChecks, value == 0.0);')
    owner=l1.source('applications/CHT/gpu/GpuResidentStrict.cu',revision)
    progress='unsigned char particleWallEnergyRestoreProgress=0;' if 'particleWallEnergyRestoreProgress' in owner else ''
    s+=r'''
using GpuWallEnergy=double;
struct DeviceState {GpuWallEnergyState wallEnergy;DeviceState*deviceState=nullptr;int nFaces=5,nInternalFaces=1;int particleWallHeatTransferEnabled=0;double*particleWallDepositedEnergy=nullptr,*particleWallReflectedEnergy=nullptr;PROGRESS};
char lastError[2048];
void setLastError(const char*name,int){std::snprintf(lastError,sizeof lastError,"%s",name);}void setLastErrorText(const char*name){setLastError(name,0);}
template<class T>int allocate(T*&p,size_t n,const char*){p=nullptr;if(!n)return 0;return cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T));}
'''.replace('PROGRESS',progress)
    sync=h.block(owner,'int syncDeviceState');s+=sync+'\n'
    name='extern "C" int ugkpPrepareParticleWallEnergyRestore'
    if name in owner:
        helper=h.block(owner,name).replace('DeviceState* s = asState(handle);','DeviceState* s = static_cast<DeviceState*>(handle);');s+=helper
    s+=r'''
void setup(DeviceState&s,DeviceState&d,double*gas,unsigned char*mask){s.deviceState=&d;s.wallEnergy.deviceState=&d.wallEnergy;s.wallEnergy.nFaces=5;s.wallEnergy.nInternalFaces=1;s.wallEnergy.gasWallEnergy=gas;s.wallEnergy.gasWallEnergyMask=mask;s.wallEnergy.errorBuffer=lastError;}
void cleanup(DeviceState&s){cudaFree(s.particleWallDepositedEnergy);cudaFree(s.particleWallReflectedEnergy);s.particleWallDepositedEnergy=s.particleWallReflectedEnergy=nullptr;}
'''
    return s


def test_disabled_nonzero_restore_has_real_canonical_owners(tmp_path):
    h.run_cpp(tmp_path,code()+r'''
int main(){DeviceState s,d;double gas[5]={11,12,13,14,15};unsigned char mask[5]{};setup(s,d,gas,mask);double g[]={7,8},dep[]={-.25,std::numeric_limits<double>::infinity()},ref[]={-0.,.5};counters();
 assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,1,2,g,dep,ref)==0);
 assert(s.particleWallDepositedEnergy==s.wallEnergy.particleWallDepositedEnergy&&s.particleWallDepositedEnergy==d.particleWallDepositedEnergy&&s.particleWallReflectedEnergy==d.wallEnergy.particleWallReflectedEnergy);
 assert(std::memcmp(s.particleWallDepositedEnergy+1,dep,sizeof dep)==0&&std::memcmp(s.particleWallReflectedEnergy+1,ref,sizeof ref)==0&&gas[1]==7);
 assert(s.particleWallDepositedEnergy[0]==0&&s.particleWallDepositedEnergy[3]==0&&s.particleWallReflectedEnergy[4]==0);
 assert(allocations==2&&memsets==2&&h2d==4&&d2h==0&&syncs==0);
 dep[1]=std::numeric_limits<double>::quiet_NaN();auto* saved=s.particleWallDepositedEnergy;counters();assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,1,2,g,dep,ref)==0);assert(allocations==0&&memsets==0&&h2d==3&&s.particleWallDepositedEnergy==saved&&std::memcmp(saved+1,dep,sizeof dep)==0);
 double zero[2]={0.,-0.};counters();assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,1,2,g,zero,zero)==0);assert(h2d==1&&allocations==0&&memsets==0&&s.particleWallDepositedEnergy[1]==-.25);
 double outG[2],outD[2]={99,99},outR[2]={99,99};counters();assert(ugkwpGpuResidentStrictPeekWallEnergyLedgerRange(&s,1,2,outG,outD,outR)==0);assert(outD[0]==0&&outD[1]==0&&outR[0]==0&&outR[1]==0&&d2h==1&&saved[1]==-.25);
 cleanup(s);assert(live.empty());}
''')


def test_allocation_initialisation_publication_retry_progress(tmp_path):
    result=h.run_cpp(tmp_path,code()+r'''
int main(){
 for(int kind=0;kind<3;++kind)for(int fault=1;fault<=(kind==2?4:2);++fault){DeviceState s,d;double gas[5]={};unsigned char mask[5]{};setup(s,d,gas,mask);double g=7,dep=-3,ref=5;counters();
  if(kind==0)failAlloc=fault;if(kind==1)failMemset=fault;if(kind==2)failCopy=fault;
  assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,1,1,&g,&dep,&ref)!=0);
  const int acquired=live.size();auto* saved=s.particleWallDepositedEnergy;
  if(kind==0)assert(acquired==fault-1);if(kind==1)assert(acquired==fault);if(kind==2)assert(acquired==(fault==1?0:2));
  if(kind==1)assert(s.particleWallEnergyRestoreProgress&(fault==1?1:2));
  if(kind==2&&fault==2)assert(s.particleWallEnergyRestoreProgress==4);
  // Old disabled zero restore cannot consume/reset pending work or retained data.
  unsigned char pending=s.particleWallEnergyRestoreProgress;double zero=-0.;counters();assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,1,1,&g,&zero,&zero)==0);assert(h2d==1&&allocations==0&&memsets==0&&s.particleWallEnergyRestoreProgress==pending);
  counters();assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,1,1,&g,&dep,&ref)==0);
  assert(allocations==2-acquired);if(saved)assert(s.particleWallDepositedEnergy==saved);assert(s.particleWallEnergyRestoreProgress==0&&live.size()==2);
  int zeroPasses=kind==0?3-fault:kind==1?3-fault:fault==1?2:0;assert(memsets==zeroPasses);
  assert(s.particleWallDepositedEnergy[1]==dep&&s.particleWallReflectedEnergy[1]==ref&&d.particleWallDepositedEnergy==s.particleWallDepositedEnergy&&d.wallEnergy.particleWallReflectedEnergy==s.particleWallReflectedEnergy);
  std::printf("retry kind=%d fault=%d alloc=%d zero=%d h2d=%d copies=%d sync=%d\n",kind,fault,allocations,memsets,h2d,copies,syncs);
  cleanup(s);assert(live.empty());
 }
}
''')
    (tmp_path/'retry-cost.log').write_text(result)


def test_retained_single_owner_and_actual_copy_errors(tmp_path):
    h.run_cpp(tmp_path,code()+r'''
int main(){for(int existing=0;existing<2;++existing){DeviceState s,d;double gas[5]={};unsigned char mask[5]{};setup(s,d,gas,mask);double*owner=nullptr;assert(allocate(owner,5,"seed")==0);for(int i=0;i<5;++i)owner[i]=19+i;
 if(existing==0)s.particleWallDepositedEnergy=owner;else s.particleWallReflectedEnergy=owner;assert(syncDeviceState(&s,"seed sync")==0);
 double g=-1,dep=3,ref=5;counters();assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,1,1,&g,&dep,&ref)==0);assert(allocations==1&&memsets==1&&h2d==4&&owner[0]==19&&owner[2]==21);
 for(int fault=1;fault<=3;++fault){s.particleWallDepositedEnergy[1]=11;s.particleWallReflectedEnergy[1]=13;counters();failCopy=fault;assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,1,1,&g,&dep,&ref)!=0);assert(allocations==0&&memsets==0&&copies==fault);assert(s.particleWallDepositedEnergy[1]==(fault>2?3:11)&&s.particleWallReflectedEnergy[1]==13);}
 counters();assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,1,1,&g,nullptr,&ref)!=0&&copies==0);cleanup(s);assert(live.empty());}}
''')


def test_original_complete_owner_payload_and_operation_parity(tmp_path):
    main=r'''
#include <iostream>
#include <iomanip>
int main(){DeviceState s,d;double gas[5]={11,12,13,14,15};unsigned char mask[5]{};setup(s,d,gas,mask);assert(allocate(s.particleWallDepositedEnergy,5,"seed")==0);assert(allocate(s.particleWallReflectedEnergy,5,"seed")==0);for(int i=0;i<5;++i){s.particleWallDepositedEnergy[i]=20+i;s.particleWallReflectedEnergy[i]=-30-i;}assert(syncDeviceState(&s,"seed sync")==0);
 for(int enabled:{0,1}){s.particleWallHeatTransferEnabled=enabled;s.wallEnergy.particleWallHeatTransferEnabled=enabled;double g[2]={.125,.5},dep[2]={0.,-0.},ref[2]={-0.,0.};if(enabled){dep[0]=7;ref[1]=-9;}counters();predicateChecks=0;assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,1,2,g,dep,ref)==0);
 std::cout<<"COST "<<enabled<<" "<<allocations<<" "<<memsets<<" "<<copies<<" "<<h2d<<" "<<d2h<<" "<<syncs<<" "<<copiedBytes<<" "<<predicateChecks<<"\n";
 std::cout<<"DATA"<<std::hexfloat;for(int i=0;i<5;++i)std::cout<<" "<<gas[i]<<" "<<s.particleWallDepositedEnergy[i]<<" "<<s.particleWallReflectedEnergy[i];std::cout<<"\n";}
 cleanup(s);assert(live.empty());}
'''
    result={}
    for name,revision in [('old',BASE),('current',None)]:
        d=tmp_path/name;d.mkdir();result[name]=h.run_cpp(d,code(revision)+main)
    assert result['old']==result['current']
    (tmp_path/'parity.json').write_text(json.dumps(result,indent=2))



def test_supplied_range_metadata_keeps_actual_addressing(tmp_path):
    h.run_cpp(tmp_path,code()+r'''
int main(){DeviceState s,d;double gas[5]={11,12,13,14,15};unsigned char mask[5]{};setup(s,d,gas,mask);s.wallEnergy.nInternalFaces=99;s.wallEnergy.gasWallEnergyMask=nullptr;double g=7,dep=3,ref=5;counters();
 assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,0,1,&g,&dep,&ref)==0);assert(gas[0]==7&&s.particleWallDepositedEnergy[0]==3&&s.particleWallReflectedEnergy[0]==5);
 counters();assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,0,0,&g,&dep,&ref)==0&&copies==0&&allocations==0&&memsets==0);
 cleanup(s);assert(live.empty());}
''')


def test_partial_device_publication_retains_referenced_storage(tmp_path):
    probe=code().replace('int copies=0,','bool partialCopy=false;int copies=0,',1)
    probe=probe.replace('if(copies==failCopy)return 1;', 'if(copies==failCopy){if(partialCopy)std::memcpy(d,s,n>8?n-8:n);return 1;}',1)
    h.run_cpp(tmp_path,probe+r'''
int main(){DeviceState s,d;double gas[5]={};unsigned char mask[5]{};setup(s,d,gas,mask);double g=7,dep=3,ref=5;counters();failCopy=2;partialCopy=true;
 assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,1,1,&g,&dep,&ref)!=0);
 assert(s.particleWallEnergyRestoreProgress==4&&d.wallEnergy.particleWallDepositedEnergy==s.particleWallDepositedEnergy&&d.wallEnergy.particleWallReflectedEnergy==s.particleWallReflectedEnergy);
 assert(live.size()==2&&live.count(d.wallEnergy.particleWallDepositedEnergy)&&live.count(d.wallEnergy.particleWallReflectedEnergy));
 assert(d.wallEnergy.particleWallDepositedEnergy[1]==0&&d.wallEnergy.particleWallReflectedEnergy[1]==0);
 auto*deposited=s.particleWallDepositedEnergy;counters();partialCopy=false;
 assert(ugkwpGpuResidentStrictUploadWallEnergyLedgerRange(&s,1,1,&g,&dep,&ref)==0&&allocations==0&&memsets==0&&h2d==4);
 assert(s.particleWallDepositedEnergy==deposited&&d.particleWallDepositedEnergy==deposited&&deposited[1]==3&&s.particleWallEnergyRestoreProgress==0);
 cleanup(s);assert(live.empty());}
''')
