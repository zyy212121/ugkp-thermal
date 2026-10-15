"""Actual CHT owner/upload bodies, with recorded CUDA boundaries (no GPU claim)."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import importlib.util
import json
ROOT=SOURCE_ROOT
spec=importlib.util.spec_from_file_location('stuck',(VALIDATION_ROOT / 'tests/test_stuck_owner_lifecycle.py'));stuck=importlib.util.module_from_spec(spec);spec.loader.exec_module(stuck)
h=stuck.h
BASE='9291ff418f4da00cd9029a40823789e2c5754baa'

def code(rev=None):
    src=stuck.text('CHT',rev)
    c=stuck.code('CHT',rev)
    for name in ['int publishParticleWallEffusivity(']:
        if name in src:
            body=h.block(src,name)
            c=c.replace('extern "C" int ugkwpGpuResidentStrictConfigureParticleStuckModel',body+'\nextern "C" int ugkwpGpuResidentStrictConfigureParticleStuckModel',1)
    c+='\n'+h.block(src,'extern "C" int ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch')
    return c+'\nint upload(DeviceState&s,int start,int count,const double*p){return ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch(&s,start,count,p);}\n'

def test_missing_owner_canonical_restore_and_l3(tmp_path):
    h.run_cpp(tmp_path,code()+r'''
int main(){DeviceState s,d;s.deviceState=&d;double x[]={-0.,-2};counters();assert(upload(s,1,0,nullptr)==0&&allocations==0&&copies==0&&memsets==0);
 assert(upload(s,1,2,x)==0);assert(s.particleWallEffusivityByFace&&d.particleWallEffusivityByFace==s.particleWallEffusivityByFace);assert(std::memcmp(s.particleWallEffusivityByFace+1,x,16)==0);assert(allocations==1&&memsets==0&&copies==3&&copiedBytes==4*sizeof(GpuReal)+sizeof(void*)+16&&syncs==0);auto*p=s.particleWallEffusivityByFace;
 unsigned char mask[]={0,1,0,0};counters();assert(configure(s,mask)==0);assert(s.particleWallEffusivityByFace==p&&std::memcmp(p+1,x,16)==0);assert(!(s.particleStuckInitialisationPending&(4|2048)));
 counters();assert(upload(s,1,2,x)==0&&copies==1&&allocations==0&&memsets==0);cleanup(s);assert(live.empty());}
''',[ROOT/'applications/CHT/thermal'])

def partial(c):
    return c.replace('if(copies==failCopy)return 1;', 'if(copies==failCopy){if(n)std::memcpy(d,s,(n+1)/2);return 1;}').replace('if(memsets==failMemset)return 1;', 'if(memsets==failMemset){if(n)std::memset(d,v,(n+1)/2);return 1;}')

def test_each_partial_failure_retries_only_unfinished_work(tmp_path):
    h.run_cpp(tmp_path,partial(code())+r'''
int main(){for(int fault=0;fault<4;++fault){DeviceState s,d;s.deviceState=&d;double x[]={-0.,std::nan("23")};counters();if(fault==0)failAlloc=1;if(fault==1)failCopy=1;if(fault==2)failCopy=2;if(fault==3)failCopy=3;
 assert(upload(s,1,2,x)!=0);auto*p=s.particleWallEffusivityByFace;
 if(fault==0)assert(!p&&live.empty());else assert(p&&live.count(p));
 if(fault==1)assert((s.particleStuckInitialisationPending&(4|2048))==(4|2048));
 if(fault==2)assert((s.particleStuckInitialisationPending&(4|2048))==2048);
 if(fault==3)assert(!(s.particleStuckInitialisationPending&(4|2048)));
 counters();assert(upload(s,0,0,nullptr)==0&&allocations==0&&memsets==0&&copies==0);
 assert(upload(s,1,2,x)==0);assert(s.particleWallEffusivityByFace==(p?p:s.particleWallEffusivityByFace));assert(d.particleWallEffusivityByFace==s.particleWallEffusivityByFace);assert(std::memcmp(s.particleWallEffusivityByFace+1,x,16)==0);
 assert(allocations==(fault==0));assert(memsets==0);assert(copies==(fault<=1?3:fault==2?2:1));assert(syncs==0);
 std::cout<<"RETRY "<<fault<<" "<<allocations<<" "<<memsets<<" "<<copies<<" "<<copiedBytes<<"\n";cleanup(s);assert(live.empty());}
 DeviceState s,d;s.deviceState=&d;counters();assert(upload(s,1,2,nullptr)!=0&&allocations==0&&copies==0);}
''',[ROOT/'applications/CHT/thermal'])

def test_l3_failed_initialisation_and_publication_then_patch_restore(tmp_path):
    h.run_cpp(tmp_path,partial(code())+r'''
int main(){unsigned char mask[]={0,1,0,0};
 // Discover the actual successful sequence, then inject every failure after effusivity acquisition.
 DeviceState baseline,dev;baseline.deviceState=&dev;counters();assert(configure(baseline,mask)==0);int nAlloc=allocations,nZero=memsets,nCopy=copies;cleanup(baseline);
 int cases=0;for(int kind=0;kind<3;++kind){int total=kind==0?nAlloc:kind==1?nZero:nCopy;for(int at=1;at<=total;++at){DeviceState s,d;s.deviceState=&d;counters();if(kind==0)failAlloc=at;if(kind==1)failMemset=at;if(kind==2)failCopy=at;assert(configure(s,mask)!=0);
 auto*p=s.particleWallEffusivityByFace;if(p){++cases;double x[]={91,-0.};counters();assert(upload(s,1,2,x)==0);assert(s.particleWallEffusivityByFace==p&&d.particleWallEffusivityByFace==p&&std::memcmp(p+1,x,16)==0);assert(allocations==0);
 counters();assert(configure(s,mask)==0);assert(s.particleWallEffusivityByFace==p&&std::memcmp(p+1,x,16)==0);assert(!(s.particleStuckInitialisationPending&(4|2048)));
 unsigned char off[4]{};assert(configure(s,off,0)==0);assert(configure(s,mask)==0);assert(s.particleWallEffusivityByFace==p&&std::memcmp(p+1,x,16)==0);}
 cleanup(s);assert(live.empty());}}
 assert(cases>4);std::cout<<"L3_FAILURE_INTEROP "<<cases<<"\n";}
''',[ROOT/'applications/CHT/thermal'])

def test_complete_owner_old_success_exact_bytes_and_cost(tmp_path):
    main=r'''
int main(){DeviceState s,d;s.deviceState=&d;s.particleWallHeatTransferEnabled=1;double backing[]={11,22,33,44};s.particleWallEffusivityByFace=backing;double x[]={123.25,894.5};counters();assert(upload(s,1,2,x)==0);for(double v:backing){unsigned long long bits;std::memcpy(&bits,&v,8);std::cout<<bits<<" ";}std::cout<<" COST "<<allocations<<" "<<memsets<<" "<<copies<<" "<<copiedBytes<<" "<<syncs<<"\n";assert(upload(s,-10,0,nullptr)==0);}
'''
    results={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();results[label]=h.run_cpp(d,code(rev)+main,[ROOT/'applications/CHT/thermal'])
    assert results['old']==results['current'];(tmp_path/'parity.json').write_text(json.dumps(results,indent=2))

def test_admitted_nonfinite_and_existing_owner_cuda_error(tmp_path):
    h.run_cpp(tmp_path,partial(code())+r'''
int main(){DeviceState s,d;s.deviceState=&d;double backing[]={11,22,33,44};s.particleWallEffusivityByFace=backing;double x[]={std::nan("123"),-std::numeric_limits<double>::infinity()};counters();assert(upload(s,0,2,x)==0&&copies==1&&allocations==0&&memsets==0&&std::memcmp(backing,x,16)==0);
 counters();failCopy=1;assert(upload(s,0,2,x)!=0&&copies==1&&allocations==0&&memsets==0);counters();assert(upload(s,0,2,x)==0&&copies==1&&std::memcmp(backing,x,16)==0);}
''',[ROOT/'applications/CHT/thermal'])

def header(rev=None):
    spec=importlib.util.spec_from_file_location('consumer',(VALIDATION_ROOT / 'tests/test_native_consumer_admission.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    src=c.source('applications/CHT/gpu/GpuResidentStrict.H',rev)
    pre=c.HEADER_PRELUDE.replace('const Boundary&boundaryMesh()const{return patches;}','const Boundary&boundaryMesh()const{return patches;}const Boundary&boundary()const{return patches;}')
    return h.CUDA+pre+r'''
std::vector<int>starts,counts;std::vector<unsigned long long>uploaded;
int ugkwpGpuResidentStrictUploadParticleWallEffusivityPatch(void*,int start,int count,const double*values){++nativeCalls;starts.push_back(start);counts.push_back(count);for(int i=0;i<count;++i){unsigned long long bits;std::memcpy(&bits,values+i,8);uploaded.push_back(bits);}return nativeFault;}
struct Resident{void*handle_=this;const fvMesh*residentMesh_=nullptr;bool particleWallHeatTransferEnabled_=true;Field<unsigned char>particleStuckCandidateFaceMask_={0,0,1,1,1,1};
 void requireInitialised(const char*)const{}void check(int error,const char*)const{if(error)throw std::runtime_error("native error");}
 static bool finiteMirrorScalar(double x){++finiteChecks;return std::isfinite(x)&&std::abs(x)<1e30;}
'''+h.block(src,'    bool particleWallContactEnabledOnPatch')+h.block(src,'    void updateParticleWallEffusivities')+'};\n'

def test_frontend_old_success_exact_and_removed_scans(tmp_path):
    main=r'''
int main(){fvMesh mesh;Resident r;r.residentMesh_=&mesh;List<scalarField>values{{123.25,999},{81.25,27}};labelList patches{1,0};r.updateParticleWallEffusivities(mesh,values,patches);std::cout<<"DATA";for(int n:starts)std::cout<<" s"<<n;for(int n:counts)std::cout<<" n"<<n;for(auto n:uploaded)std::cout<<" v"<<n;std::cout<<"\nCOST "<<nativeCalls<<" "<<finiteChecks<<"\n";assert(r.particleWallContactEnabledOnPatch(mesh,0));r.particleWallHeatTransferEnabled_=false;assert(!r.particleWallContactEnabledOnPatch(mesh,0));}
'''
    results={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();results[label]=h.run_cpp(d,header(rev)+main)
    assert results['old'].splitlines()[0]==results['current'].splitlines()[0]
    assert 'COST 2 4' in results['old'] and 'COST 2 0' in results['current']
    (tmp_path/'frontend-parity.json').write_text(json.dumps(results,indent=2))

def test_frontend_actual_disabled_upload_partial_query_and_errors(tmp_path):
    h.run_cpp(tmp_path,header()+r'''
int main(){fvMesh mesh,foreign;Resident r;r.residentMesh_=&foreign;r.particleStuckCandidateFaceMask_[3]=0;assert(r.particleWallContactEnabledOnPatch(mesh,0));r.particleWallHeatTransferEnabled_=false;assert(!r.particleWallContactEnabledOnPatch(mesh,0));List<scalarField>values{{-0.,std::nan("17")},{-3,std::numeric_limits<double>::infinity()}};labelList patches{1,0};r.updateParticleWallEffusivities(mesh,values,patches);assert(nativeCalls==2&&finiteChecks==0&&starts[0]==4&&starts[1]==2);for(int p=0;p<2;++p)for(int i=0;i<2;++i){unsigned long long bits;std::memcpy(&bits,&values[patches[p]][i],8);assert(uploaded[p*2+i]==bits);}nativeFault=1;bool failed=false;try{r.updateParticleWallEffusivities(mesh,values,patches);}catch(const std::runtime_error&){failed=true;}assert(failed&&nativeCalls==3);}
''')
