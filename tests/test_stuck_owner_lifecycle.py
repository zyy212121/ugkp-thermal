"""Actual paired configuration owners; CUDA boundaries are recorded, not GPU execution."""
from pathlib import Path
import importlib.util
import subprocess
import re
import json
import pytest
ROOT=Path(__file__).resolve().parents[1]
BASE='6e4280d681e246c9324324f6b2d3985744671df7'
spec=importlib.util.spec_from_file_location('h',ROOT/'tests/test_native_cleanup.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def text(app,revision=None):
    path=f'applications/{app}/'+('gpu' if app=='CHT' else 'private_backend')+'/GpuResidentStrict.cu'
    return subprocess.check_output(['git','show',revision+':'+path],cwd=ROOT,text=True) if revision else (ROOT/path).read_text()


def owners(app):
    body=h.block(text(app,BASE),'extern "C" int ugkwpGpuResidentStrictConfigureParticleStuckModel')
    found=re.findall(r'^    (unsigned char|GpuWallEnergy|GpuReal|float|double)\* (\w+) = nullptr;',body,re.M)
    return [(kind,name,re.search(r's->(\w+)\s*=\s*'+name+r'\s*;',body).group(1)) for kind,name in found]


def code(app,revision=None):
    src=text(app,revision);body=h.block(src,'extern "C" int ugkwpGpuResidentStrictConfigureParticleStuckModel').replace('std::vector<','CountedVector<');slots=owners(app)
    cuda=h.CUDA.replace('#include <vector>','#include <vector>\n#include <map>\n#include <string>\n#include <iostream>\n#include <iomanip>\nstd::map<void*,size_t> live;std::vector<std::string>events;',1)
    cuda=cuda.replace('*p=std::malloc(n);return *p?0:1;','*p=std::malloc(n);if(*p){live[*p]=n;std::memset(*p,0xa5,n);}return *p?0:1;')
    cuda=cuda.replace('int cudaFree(void* p){std::free(p);return 0;}','int cudaFree(void* p){if(p){assert(live.erase(p)==1);std::free(p);}return 0;}')
    cuda=cuda.replace('++copies;if(copies==failCopy)', '++copies;events.push_back("copy "+std::to_string(kind)+" "+std::to_string(n));if(copies==failCopy)')
    cuda=cuda.replace('++memsets;if(memsets==failMemset)', '++memsets;events.push_back("zero "+std::to_string(n));if(memsets==failMemset)')
    cuda=cuda.replace('++allocations;if(allocations==failAlloc)', '++allocations;events.push_back("alloc "+std::to_string(n));if(allocations==failAlloc)')
    cuda=cuda.replace('void counters(){','void counters(){events.clear();',1)
    cuda+=r'''
constexpr int cudaErrorMemoryAllocation=2;
int hostBuffers=0;size_t hostFilledBytes=0;
template<class T>struct CountingAllocator:std::allocator<T>{template<class U>struct rebind{using other=CountingAllocator<U>;};T*allocate(size_t n){++hostBuffers;return std::allocator<T>::allocate(n);}};
template<class T>struct CountedVector:std::vector<T,CountingAllocator<T>>{explicit CountedVector(size_t n,T v=T{}):std::vector<T,CountingAllocator<T>>(n,v){hostFilledBytes+=n*sizeof(T);}};
'''
    cuda=cuda.replace('void counters(){','void counters(){hostBuffers=0;hostFilledBytes=0;')
    cuda=cuda.replace('int copies=0,','int hostBuffers;extern size_t hostFilledBytes;\nint copies=0,').replace('int hostBuffers=0;size_t hostFilledBytes=0;','size_t hostFilledBytes=0;')
    includes=f'''\n#include "applications/{app}/thermal/GpuFiniteWallContact.H"\n#include "common/wall/GpuColdWallSolidification.H"\n#include "common/wall/GpuColdWall2DSolidification.H"\n#include "common/ContactCapabilityValidation.h"\n'''
    if app=='CHT':includes+=(ROOT/'applications/CHT/gpu/GpuWallEnergy64.H').read_text().replace('#include <cuda_runtime.h>','')
    fields='\n'.join(kind+'* '+field+'=nullptr;' for kind,name,field in slots)
    common=r'''
struct DeviceState {DeviceState*deviceState=nullptr;int nFaces=4,nInternalFaces=1,particleCapacity=2;int particleStuckModelConfigured=0,particleWallHeatTransferEnabled=0,coldWallSolidificationEnabled=0,coldWall2DEnabled=0;
unsigned char particleWallEnergyRestoreProgress=0;unsigned short particleStuckInitialisationPending=0;
Foam::gpuThermal::ColdWallSolidificationParameters coldWallSolidificationParameters{};
double sommerfeldThreshold=0,particleWallMaximumCoverage=0,particleWallDepositionHeatTransferEfficiency=0,particleWallReflectionHeatTransferEfficiency=0,particleWallAdhesionEnergyScale=0,particleWallContactAngleCosine=0,particleWallDensityKgM3=0,particleWallSpecificHeatJkgK=0,particleWallConductivityWmK=0;
FIELDS
};
char lastError[2048];DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}int validateState(DeviceState*s,const char*){return !s;}
void setLastErrorText(const char*s){std::snprintf(lastError,sizeof lastError,"%s",s);}void setLastError(const char*s,int){setLastErrorText(s);}
template<class T>int allocate(T*&p,size_t n,const char*){p=nullptr;if(!n)return 0;return cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T));}
template<class T>void release(T*&p){cudaFree(p);p=nullptr;}
template<class T>int copyToDevice(T*d,const T*s,size_t n,const char*){return n?cudaMemcpy(d,s,n*sizeof(T),1):0;}
int syncDeviceState(DeviceState*s,const char*){return cudaMemcpy(s->deviceState,s,sizeof(*s),1);}
'''.replace('FIELDS',fields)
    if app=='CHT':common=common.replace('struct DeviceState {','struct DeviceState {GpuWallEnergyState wallEnergy;')
    common=common.replace('int syncDeviceState(DeviceState*s,const char*){return cudaMemcpy(s->deviceState,s,sizeof(*s),1);}',h.block(src,'int syncDeviceState('))
    if app=='CHT':common+=h.block(src,'extern "C" int ugkpPrepareParticleWallEnergyRestore')+'\n'
    for helper in ['int acquireParticleWallEffusivity(', 'cudaError_t initialiseParticleWallEffusivity(']:
        if helper in src:common+=h.block(src,helper).replace('std::vector<', 'CountedVector<')+'\n'

    call='20,heat,.63,1,1,1,'+('1800,710,100,' if app=='FSH' else '')+'145,0,12,2300,20,100000,2500,1000,20,.5,0'
    post='int configure(DeviceState&s,const unsigned char*m,int heat=1){return ugkwpGpuResidentStrictConfigureParticleStuckModel(&s,4,m,'+call+');}\n'
    post+='void cleanup(DeviceState&s){'+''.join('release(s.'+f+');' for _,_,f in slots)+'}\n'
    post+='std::vector<void*> pointers(DeviceState&s){return {'+','.join('s.'+f for _,_,f in slots)+'};}\n'
    post+=r'''
std::map<void*,std::vector<unsigned char>> snapshot(DeviceState&s){std::map<void*,std::vector<unsigned char>> out;for(auto*p:pointers(s))if(p)out[p]=std::vector<unsigned char>((unsigned char*)p,(unsigned char*)p+live.at(p));return out;}
void assertRetained(DeviceState&s,const std::map<void*,std::vector<unsigned char>>&old){for(auto&v:old){assert(live.count(v.first));if(v.first!=s.particleStuckCandidateMask)assert(std::memcmp(v.first,v.second.data(),v.second.size())==0);}}
'''
    return cuda+includes+common+body+post


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_model_switch_retains_canonical_owners(tmp_path,app):
    h.run_cpp(tmp_path,code(app)+r'''
int main(){DeviceState s,d;s.deviceState=&d;unsigned char one[]={0,3,0,0},two[]={0,4,0,0},off[]={0,0,0,0};
 assert(configure(s,one)==0);auto first=snapshot(s);counters();assert(configure(s,two)==0);assertRetained(s,first);auto both=snapshot(s);
 counters();assert(configure(s,off,0)==0);assertRetained(s,both);assert(allocations==0&&memsets==0&&h2d==2&&hostBuffers==0&&hostFilledBytes==0);
 counters();assert(configure(s,one)==0);assertRetained(s,both);assert(allocations==0&&memsets==0&&h2d==2&&hostBuffers==0&&hostFilledBytes==0);
 assert(d.particleWallDepositedEnergy==s.particleWallDepositedEnergy&&d.pColdNodeSpecificEnthalpy==s.pColdNodeSpecificEnthalpy);cleanup(s);assert(live.empty());}
''', [ROOT/'common',ROOT/f'applications/{app}/thermal'])


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_original_first_setup_exact_matrix(tmp_path,app):
    hashes=''
    for _,name,field in owners(app):
        if name.startswith('compact'):continue
        hashes+=f'if(s.{field})std::cout<<" {name}="<<hash(s.{field},live.at(s.{field}));\n'
    main=r'''
unsigned long long hash(const void*p,size_t n){auto*b=(const unsigned char*)p;unsigned long long h=1469598103934665603ULL;for(size_t i=0;i<n;++i)h=(h^b[i])*1099511628211ULL;return h;}
int main(){for(int model=0;model<5;++model)for(int heat:{0,1}){if(model==0&&heat)continue;DeviceState s,d;s.deviceState=&d;unsigned char mask[4]={0,0,0,0};if(model==1)mask[1]=1;if(model==2)mask[1]=3;if(model==3)mask[1]=4;if(model==4){mask[1]=3;mask[2]=4;}counters();assert(configure(s,mask,heat)==0);
 std::cout<<"CASE "<<model<<" "<<heat<<" COST "<<allocations<<" "<<copies<<" "<<h2d<<" "<<memsets<<" "<<zeroedBytes<<" HOST "<<hostBuffers<<" "<<hostFilledBytes<<"\n";
 for(const auto&e:events)std::cout<<"OP "<<e<<"\n";
 std::cout<<"DATA";HASHES
 std::cout<<std::hexfloat<<" "<<s.particleWallContactAngleCosine<<" "<<s.sommerfeldThreshold<<" "<<s.particleWallMaximumCoverage<<" "<<s.coldWallSolidificationEnabled<<" "<<s.coldWall2DEnabled<<"\n";
 // Never hash unread compact scratch as an old physical output.
 cleanup(s);assert(live.empty());}}
'''.replace('HASHES',hashes)
    result={}
    for label,revision in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();result[label]=h.run_cpp(d,code(app,revision)+main,[ROOT/'common',ROOT/f'applications/{app}/thermal'])
    if app == 'FSH':
        assert result['old'] == result['current']
    else:




        before, after = result['old'].splitlines(), result['current'].splitlines()
        assert len(before) == len(after)
        heat = False
        replacements = 0
        for old, current in zip(before, after):
            if old.startswith('CASE '):
                a, b = old.split(), current.split()
                heat = a[2] == '1'
                if heat:


                    assert int(b[5]) == int(a[5]) + 1
                    assert int(b[6]) == int(a[6]) + 1
                    assert int(b[7]) == int(a[7]) - 1
                    assert int(b[8]) == int(a[8]) - 4*8
                    assert int(b[10]) == int(a[10]) + 1
                    assert int(b[11]) == int(a[11]) + 4*8
                    for i in (5, 6, 7, 8, 10, 11):
                        b[i] = a[i]
                assert a == b
            elif old != current:
                assert heat and old == 'OP zero 32' and current == 'OP copy 1 32'
                replacements += 1
        assert replacements == 4
    (tmp_path/'first-success-parity.json').write_text(json.dumps(result,indent=2))


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_every_cuda_fault_retains_all_owners_and_retries(tmp_path,app):
    result=h.run_cpp(tmp_path,code(app)+r'''
int main(){int limits[3]={0,0,0};unsigned char mixed[]={0,3,4,0};
 {DeviceState s,d;s.deviceState=&d;counters();assert(configure(s,mixed)==0);limits[0]=allocations;limits[1]=memsets;limits[2]=copies;cleanup(s);assert(live.empty());}
 for(int kind=0;kind<3;++kind)for(int fault=1;fault<=limits[kind];++fault){DeviceState s,d;s.deviceState=&d;counters();if(kind==0)failAlloc=fault;if(kind==1)failMemset=fault;if(kind==2)failCopy=fault;
  assert(configure(s,mixed)!=0);auto retained=pointers(s);int owned=0;for(auto*p:retained)if(p){++owned;assert(live.count(p));}assert(owned==int(live.size()));
  counters();assert(configure(s,mixed)==0);auto after=pointers(s);for(size_t i=0;i<retained.size();++i)if(retained[i])assert(retained[i]==after[i]);
  assert(s.particleStuckInitialisationPending==0&&s.particleWallEnergyRestoreProgress==0);
  std::cout<<"retry "<<kind<<" "<<fault<<" owned="<<owned<<" alloc="<<allocations<<" zero="<<memsets<<" copies="<<copies<<"\n";
  cleanup(s);assert(live.empty());
 }
}
''',[ROOT/'common',ROOT/f'applications/{app}/thermal'])
    (tmp_path/'retry-cost.log').write_text(result)


def test_cht_l2_complete_and_pending_owners_are_reused(tmp_path):
    h.run_cpp(tmp_path,code('CHT')+r'''
int main(){unsigned char mixed[]={0,3,4,0};for(int kind=0;kind<4;++kind)for(int fault=1;fault<=(kind==0?1:kind==3?1:2);++fault){DeviceState s,d;s.deviceState=&d;counters();if(kind==1)failAlloc=fault;if(kind==2)failMemset=fault;if(kind==3)failCopy=fault;
 int rc=ugkpPrepareParticleWallEnergyRestore(&s);assert((rc==0)==(kind==0));auto* dep=s.particleWallDepositedEnergy;auto* ref=s.particleWallReflectedEnergy;
 if(kind==0){dep[1]=17;ref[2]=-23;}
 counters();assert(configure(s,mixed)==0);if(dep)assert(dep==s.particleWallDepositedEnergy);if(ref)assert(ref==s.particleWallReflectedEnergy);
 if(kind==0)assert(dep[1]==17&&ref[2]==-23);assert(s.particleWallEnergyRestoreProgress==0&&d.wallEnergy.particleWallDepositedEnergy==s.particleWallDepositedEnergy);
 cleanup(s);assert(live.empty());}}
''',[ROOT/'common',ROOT/'applications/CHT/thermal'])


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_partial_publication_retains_owners_and_retries(tmp_path,app):
    partial=code(app).replace('if(copies==failCopy)return 1;', 'if(copies==failCopy){std::memcpy(d,s,n-8);return 1;}')
    h.run_cpp(tmp_path,partial+r'''
int main(){DeviceState s,d;s.deviceState=&d;unsigned char mixed[]={0,3,4,0};counters();failCopy=PUBLICATION_COPY;
 assert(configure(s,mixed)!=0);auto before=pointers(s);auto device=pointers(d);for(size_t i=0;i+1<device.size();++i)assert(device[i]==before[i]);
 for(auto*p:device)if(p)assert(live.count(p));assert(s.particleStuckInitialisationPending==PUBLICATION_PENDING);
 counters();assert(configure(s,mixed)==0);assert(before==pointers(s));assert(allocations==0&&memsets==0&&hostBuffers==0);
 assert(h2d==EXPECTED);cleanup(s);assert(live.empty());}
'''.replace('EXPECTED','3' if app=='CHT' else '2').replace('PUBLICATION_COPY', '5' if app=='CHT' else '4').replace('PUBLICATION_PENDING', '0x800' if app=='CHT' else '0'),[ROOT/'common',ROOT/f'applications/{app}/thermal'])


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_actual_thermal_copy_swap_keeps_canonical_orientation(tmp_path,app):
    extra=[('double','puxOld','compactPuxOld'),('double','puyOld','compactPuyOld'),('double','puzOld','compactPuzOld'),('unsigned char','pStuck','compactPStuck'),('int','pStuckFaceId','compactPStuckFaceId')]
    extra += [('double',name,'compactP'+name[1:]) for name in ['pDepositionArea','pContactDuration','pContactMaximumArea','pContactPeakFraction','pContactAge']]
    fields=''.join(f'{t}*{a}=nullptr,*{b}=nullptr;' for t,a,b in extra)
    base=code(app).replace('DeviceState*deviceState=nullptr;','DeviceState*deviceState=nullptr;'+fields)
    base+=r'''
#define __host__
#define __device__
#define __forceinline__ inline
struct alignas(16) float4 {float x,y,z,w;};
namespace Foam{namespace gpuWall{template<class S>void publishWallBoundParticleIndex(S&,int);}}
#include "common/GpuCellLocalThermalFields.cuh"
'''
    init=''.join(f'assert(allocate(s.{f},2,"extra")==0);s.{f}[0]=1;s.{f}[1]=2;' for _,a,b in extra for f in [a,b])
    release=''.join(f'release(s.{f});' for _,a,b in extra for f in [a,b])
    main=r'''
int main(){DeviceState s,d;s.deviceState=&d;unsigned char mixed[]={0,3,4,0},off[]={0,0,0,0};assert(configure(s,mixed)==0);EXTRA
 // Seed both sides before using actual copy/swap; compact is scratch, never an old initialized output.
 for(auto*p:pointers(s))if(p!=s.particleStuckCandidateMask)std::memset(p,0,live.at(p));
 s.pColdNodeSpecificEnthalpy[0]=31;s.pCold2DNodeSpecificEnthalpy[0]=47;s.pColdFrozenArea[0]=13;s.pCold2DFrozenArea[0]=17;
 auto*oldActive=s.pColdNodeSpecificEnthalpy;auto*oldCompact=s.compactPColdNodeSpecificEnthalpy;
 CellLocalThermalExtraFields<true,PACKED>::copyContactAge(s,0,1);
 CellLocalThermalExtraFields<true,PACKED>::copy(s,0,1);
 CellLocalThermalExtraFields<true,PACKED>::swap(s);
 assert(s.pColdNodeSpecificEnthalpy==oldCompact&&s.compactPColdNodeSpecificEnthalpy==oldActive);
 assert(s.pColdNodeSpecificEnthalpy[Foam::gpuThermal::coldWallAxialNodeCount]==31);
 assert(s.pCold2DNodeSpecificEnthalpy[Foam::gpuThermal::coldWall2DNodeCount]==47);
 auto saved=snapshot(s);auto orientation=pointers(s);counters();assert(configure(s,off,0)==0);assertRetained(s,saved);assert(orientation==pointers(s));
 counters();assert(configure(s,mixed)==0);assertRetained(s,saved);assert(orientation==pointers(s));assert(allocations==0&&memsets==0&&hostBuffers==0&&h2d==2);
 cleanup(s);RELEASE assert(live.empty());}
'''.replace('EXTRA',init).replace('RELEASE',release).replace('PACKED','true' if app=='CHT' else 'false')
    h.run_cpp(tmp_path,base+main,[ROOT/'common',ROOT/f'applications/{app}/thermal'])


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_supplied_metadata_admission_and_missing_input(tmp_path,app):
    probe=code(app).replace('20,heat,.63,1,1,1,','std::nan(""),heat,std::nan(""),1,1,1,')
    h.run_cpp(tmp_path,probe+r'''
int main(){DeviceState s,d;s.deviceState=&d;unsigned char unknown[]={2,99,0,0};
 counters();assert(configure(s,nullptr,0)!=0);assert(allocations==0&&copies==0);
 assert(configure(s,unknown,0)==0);assert(std::isnan(s.sommerfeldThreshold)&&std::isnan(s.particleWallMaximumCoverage));
 assert(std::memcmp(s.particleStuckCandidateMask,unknown,4)==0);
 counters();assert(configure(s,unknown,0)==0);assert(allocations==0&&memsets==0&&hostBuffers==0&&h2d==2);
 cleanup(s);assert(live.empty());}
''',[ROOT/'common',ROOT/f'applications/{app}/thermal'])
