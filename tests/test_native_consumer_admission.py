"""Actual consumer bodies, recorded CUDA/host boundaries; no GPU execution claim."""
from pathlib import Path
import importlib.util
import subprocess
import re
import json
import pytest
ROOT=Path(__file__).resolve().parents[1]
BASE='5efe8c853bc3f08591dd24e6d5ada55e67a766a7'
spec=importlib.util.spec_from_file_location('ledger',ROOT/'tests/test_gas_wall_ledger_lifecycle.py');ledger=importlib.util.module_from_spec(spec);spec.loader.exec_module(ledger);h=ledger.h

def source(p,rev=None):
    return subprocess.check_output(['git','show',rev+':'+p],cwd=ROOT,text=True) if rev else (ROOT/p).read_text()

def native(app,rev=None):
    p=f'applications/{app}/'+('gpu' if app=='CHT' else 'private_backend')+'/GpuResidentStrict.cu'
    s=source(p,rev)
    names=['DownloadSst','DownloadSourceResidualMass']+(['UploadGasBoundaryTemperaturePatch','DownloadParticleWallOccupiedArea'] if app=='CHT' else ['DownloadAndResetParticleWallHeatLedgers'])
    bodies='\n'.join(h.block(s,'extern "C" int ugkwpGpuResidentStrict'+name) for name in names)
    return h.CUDA.replace('++copies;if(copies==failCopy)', '++copies;if((n&&(!d||!s))||copies==failCopy)')+r'''
using GpuReal=double;
#define GPU_R(x) (x)
int finiteChecks=0,areaPrepares=0;
bool checkedFinite(double x){++finiteChecks;return std::isfinite(x);}
struct DeviceState{DeviceState*deviceState=this;int nFaces=4,nInternalFaces=2,nCells=2,hostTurbulenceModel=3,sstConfigured=1,nBoundarySources=2,particleStuckModelConfigured=1,particleWorkGrid=1,reductionBlockThreads=1,particleWallHeatTransferEnabled=1;
 double TgasMin=200;double*k=nullptr,*omega=nullptr,*nut=nullptr,*sourceResidualMass=nullptr,*gasBoundaryT=nullptr,*riemannBoundaryT=nullptr,*particleWallRepresentedContactArea=nullptr,*particleWallContactAreaScale=nullptr,*particleWallDepositedEnergy=nullptr,*particleWallReflectedEnergy=nullptr;int*sourceFace=nullptr;};
DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}int validateState(DeviceState*s,const char*){return !s;}
void setLastErrorText(const char*){}void setLastError(const char*,int){}
template<class T>int copyToHost(T*d,const T*s,size_t n,const char*){return n?cudaMemcpy(d,s,n*sizeof(T),2):0;}
template<class T>int copyToDevice(T*d,const T*s,size_t n,const char*){return n?cudaMemcpy(d,s,n*sizeof(T),1):0;}
int prepareParticleWallContactAreaScale(DeviceState*,int){++areaPrepares;return 0;}
'''+bodies.replace('std::isfinite(', 'checkedFinite(')


def test_ledger_complete_backing_ignores_certificates_preserves_disabled_zero(tmp_path):
    h.run_cpp(tmp_path,ledger.native()+r'''
int main(){GpuWallEnergyState s,d;s.deviceState=&d;s.nFaces=4;s.nInternalFaces=2;double g[]={1,2,3,4},p[]={5,6,7,8},r[]={9,10,11,12},out[4],dep[4],ref[4];unsigned char mask[4]{};s.gasWallEnergy=g;s.gasWallEnergyMask=mask;s.particleWallDepositedEnergy=p;s.particleWallReflectedEnergy=r;s.particleWallHeatTransferEnabled=0;
 counters();assert(ugkwpGpuResidentStrictPeekWallEnergyLedgerRange(&s,0,2,out,dep,ref)==0);assert(out[0]==1&&out[1]==2&&dep[0]==0&&ref[1]==0&&p[0]==5&&d2h==1);
 counters();assert(ugkwpGpuResidentStrictPeekParticleWallHeatLedgers(&s,2,dep,ref)==0);assert(dep[0]==5&&ref[1]==10&&d2h==2);
 s.gasWallEnergyMask=nullptr;counters();assert(ugkwpGpuResidentStrictPeekGasWallEnergy(&s,2,out)==0&&out[1]==2&&d2h==1);
 counters();assert(ugkwpGpuResidentStrictResetParticleWallHeatLedgers(&s)==0&&memsets==2&&syncs==1&&p[0]==0&&r[3]==0);
 counters();assert(ugkwpGpuResidentStrictResetGasWallEnergy(&s)==0&&memsets==1&&syncs==1);
 s.particleWallDepositedEnergy=nullptr;counters();assert(ugkwpGpuResidentStrictPeekParticleWallHeatLedgers(&s,2,dep,ref)!=0&&copies==0);
}
''')


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_actual_downloads_accept_existing_backing_not_model_history(tmp_path,app):
    h.run_cpp(tmp_path,native(app)+r'''
int main(){DeviceState s;s.hostTurbulenceModel=0;s.sstConfigured=0;double k[]={-1,std::nan("")},w[]={2,std::numeric_limits<double>::infinity()},n[]={-0.,-3},a[2],b[2],c[2];s.k=k;s.omega=w;s.nut=n;
 counters();assert(ugkwpGpuResidentStrictDownloadSst(&s,a,b,c)==0);assert(std::memcmp(a,k,16)==0&&std::memcmp(b,w,16)==0&&std::memcmp(c,n,16)==0&&d2h==3);
 for(int f=1;f<=3;++f){counters();failCopy=f;assert(ugkwpGpuResidentStrictDownloadSst(&s,a,b,c)!=0&&copies==3);}
 s.k=nullptr;counters();assert(ugkwpGpuResidentStrictDownloadSst(&s,a,b,c)!=0&&copies==3);
 int faces[]={1,3},out[2],count=0;s.sourceFace=faces;s.sourceResidualMass=k;counters();assert(ugkwpGpuResidentStrictDownloadSourceResidualMass(&s,&count,1,out,a)==0&&count==2&&d2h==2&&out[1]==3); // supplied backing is actually large enough despite metadata1
 counters();assert(ugkwpGpuResidentStrictDownloadSourceResidualMass(&s,&count,0,nullptr,nullptr)==0&&count==2&&copies==0);
}
''')


def test_temperature_upload_and_occupied_area_complete_real_math(tmp_path):
    h.run_cpp(tmp_path,native('CHT')+r'''
int main(){DeviceState s;double a[]={1,2,3,4},b[]={5,6,7,8},in[]={-1,std::nan("")};s.gasBoundaryT=a;s.riemannBoundaryT=b;
 counters();assert(ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch(&s,0,2,in)==0);assert(std::memcmp(a,in,16)==0&&std::memcmp(b,in,16)==0&&h2d==2&&finiteChecks==0);
 counters();assert(ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch(&s,0,0,nullptr)==0&&copies==0);
 for(int f=1;f<=2;++f){counters();failCopy=f;assert(ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch(&s,0,2,in)!=0&&copies==2);}
 counters();assert(ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch(&s,0,2,nullptr)!=0&&copies==0);
 double represented[]={-2,3,4,5},scale[]={2,-3,0,1},out[4];s.particleWallRepresentedContactArea=represented;s.particleWallContactAreaScale=scale;
 counters();assert(ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea(&s,4,out)==0);assert(out[0]==-4&&out[1]==-9&&out[2]==0&&out[3]==5&&d2h==2&&areaPrepares==1&&finiteChecks==0);
 for(int f=1;f<=2;++f){counters();failCopy=f;assert(ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea(&s,4,out)!=0&&copies==f);}
 s.particleStuckModelConfigured=0;counters();assert(ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea(&s,4,out)==0&&out[0]==0&&copies==0);
}
''')


HEADER_PRELUDE=r'''
#include <any>
#include <map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
using label=int;using scalar=double;using word=std::string;using string=std::string;using fileName=std::string;
using std::min;using std::max;using std::endl;
int hostAllocations=0,hostBytes=0,finiteChecks=0;
template<class T>struct Field:std::vector<T>{using std::vector<T>::vector;T*begin(){return this->data();}const T*begin()const{return this->data();}const T*cdata()const{return this->data();}void setSize(size_t n,T value=T{}){this->assign(n,value);}void transfer(Field&x){this->swap(x);}Field&operator=(const T&x){std::fill(this->data(),this->data()+this->size(),x);return *this;}};
using scalarField=Field<double>;using labelList=Field<int>;using wordList=Field<word>;template<class T>using List=Field<T>;
struct boolList:Field<char>{boolList(size_t n,bool v):Field<char>(n,v){++hostAllocations;hostBytes+=n;}};
std::ostream&operator<<(std::ostream&o,const wordList&v){for(const auto&x:static_cast<const std::vector<word>&>(v))o<<x<<',';return o;}
#define forAll(f,i) for(size_t i=0;i<(f).size();++i)
#define forAllConstIter(t,d,i) for(auto i=(d).begin();i!=(d).end();++i)
#define FatalErrorInFunction std::cerr
#define FatalIOErrorInFunction(x) std::cerr
int throwingExit(){throw std::runtime_error("actual header refusal or extraction failure");}
#define exit(x) throwingExit()
#define Info std::cout
struct Entry{std::string key;std::any value;const std::string&keyword()const{return key;}};
struct dictionary{std::vector<Entry>items;std::map<std::string,dictionary>children;
 struct Iter{std::vector<Entry>::const_iterator it;void operator++(){++it;}bool operator!=(const Iter&o)const{return it!=o.it;}const Entry&operator()()const{return *it;}};
 Iter begin()const{return {items.begin()};}Iter end()const{return {items.end()};}
 bool found(const std::string&k)const{if(children.count(k))return true;for(auto&e:items)if(e.key==k)return true;return false;}
 const dictionary&subDict(const std::string&k)const{return children.at(k);}
 template<class T>T lookup(const std::string&k)const{for(auto&e:items)if(e.key==k)return std::any_cast<T>(e.value);throw std::runtime_error("missing dictionary input");}
 template<class T>T lookupOrDefault(const std::string&k,T def)const{return found(k)?lookup<T>(k):def;}
 template<class T>void set(std::string k,T value){for(auto&e:items)if(e.key==k){e.value=value;return;}items.push_back({k,value});}
};
struct polyPatch{int first,count;std::string title;bool wall;int start()const{return first;}size_t size()const{return count;}const word&name()const{return title;}};
struct wallPolyPatch{};template<class T>bool isA(const polyPatch&p){return p.wall;}
struct Boundary:Field<polyPatch>{using Field<polyPatch>::Field;int findPatchID(const word&n)const{for(size_t i=0;i<size();++i)if((*this)[i].title==n)return i;return -1;}};
struct fvMesh{Boundary patches{{2,2,"a",true},{4,2,"b",false}};int nFaces()const{return 6;}int nCells()const{return 2;}const Boundary&boundaryMesh()const{return patches;}};
struct Time{std::string path;bool write=true;bool writeTime()const{return write;}};
bool isFile(const fileName&p){return bool(std::ifstream(p));}
struct volScalarField{scalarField values;int boundaryCalls=0;scalarField&primitiveFieldRef(){return values;}void correctBoundaryConditions(){++boundaryCalls;}};
std::vector<double>arguments;std::vector<unsigned char>mask;int nativeCalls=0,nativeFault=0;
template<class...A>int ugkwpGpuResidentStrictConfigureParticleStuckModel(void*,int n,const unsigned char*m,A...a){++nativeCalls;mask.assign(m,m+n);arguments={double(a)...};return nativeFault;}
'''

def stuck_header(app,rev=None):
    src=source(f'applications/{app}/gpu/GpuResidentStrict.H',rev)
    return h.CUDA+HEADER_PRELUDE+r'''
struct Resident{void*handle_=this;bool particleWallHeatTransferEnabled_=false;word particleStuckWallHeatFluxFieldName_,particleReflectedWallHeatFluxFieldName_;Field<unsigned char>particleStuckCandidateFaceMask_;
 void check(int e,const char*)const{if(e)throw std::runtime_error("native error");}
 static bool finiteMirrorScalar(double x){++finiteChecks;return std::isfinite(x)&&std::abs(x)<1e30;}
'''+h.block(src,'    void configureParticleStuckModel')+'};\n'


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_stuck_frontend_old_forwarding_exact_and_new_metadata(tmp_path,app):
    normal=r'''
void print(Resident&r){std::cout<<"DATA "<<nativeCalls<<" "<<r.particleWallHeatTransferEnabled_<<" "<<r.particleStuckWallHeatFluxFieldName_<<" "<<r.particleReflectedWallHeatFluxFieldName_;
 for(auto x:arguments){unsigned long long b;std::memcpy(&b,&x,8);std::cout<<" "<<b;}for(auto x:mask)std::cout<<" m"<<int(x);for(auto x:static_cast<std::vector<unsigned char>&>(r.particleStuckCandidateFaceMask_))std::cout<<" h"<<int(x);std::cout<<"\n";}
int main(){fvMesh mesh;Resident r;dictionary d;r.configureParticleStuckModel(mesh,d,true);print(r);
 d.set("gpuResidentStuckWallPatches",wordList{"a"});auto&m=d.children["gpuResidentStuckModel"];m.set("heatTransfer",true);m.set("wallInteractionModel",word("reboundContact"));m.set("contactAngleDegree",73.25);r.configureParticleStuckModel(mesh,d,true);print(r);
 m.children["wallInteractionModels"].set("a",word("coldWall2D"));r.configureParticleStuckModel(mesh,d,true);print(r);
 std::cout<<"COST "<<hostAllocations<<" "<<hostBytes<<" "<<finiteChecks<<"\n";
}
'''
    results={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();results[label]=h.run_cpp(d,stuck_header(app,rev)+normal)
    assert [x for x in results['old'].splitlines() if not x.startswith('COST')]==[x for x in results['current'].splitlines() if not x.startswith('COST')]
    assert results['current'].splitlines()[-1]=='COST 0 0 0'
    (tmp_path/'header-parity.json').write_text(json.dumps(results,indent=2))
    d=tmp_path/'admitted';d.mkdir()
    h.run_cpp(d,stuck_header(app)+r'''
int main(){fvMesh mesh;Resident r;dictionary d;d.set("gpuResidentStuckWallPatches",wordList{"a","a","b"});auto&m=d.children["gpuResidentStuckModel"];m.set("heatTransfer",true);m.set("contactAngleDegree",std::nan(""));m.set("maximumCoverage",-5.);m.set("thermalContactAreaFraction",.1);m.set("wallDensity",-2.);m.set("wallInteractionModel",word("unknown"));m.set("heatFluxField",word("same"));m.set("reflectionHeatFluxField",word("same"));
 r.configureParticleStuckModel(mesh,d,false);assert(nativeCalls==1&&mask[2]==0&&mask[4]==0&&arguments[2]==-5.);
 d.set("gpuResidentStuckWallPatches",wordList{"a"});m.children["wallInteractionModels"].set("b",word("coldWall2D"));r.configureParticleStuckModel(mesh,d,false);assert(mask[4]==4&&mask[5]==4);
 d.set("gpuResidentStuckWallPatches",wordList{"missing"});bool failed=false;try{r.configureParticleStuckModel(mesh,d,false);}catch(const std::runtime_error&){failed=true;}assert(failed&&nativeCalls==2);
 d.set("gpuResidentStuckWallPatches",wordList{"a"});nativeFault=1;failed=false;try{r.configureParticleStuckModel(mesh,d,false);}catch(const std::runtime_error&){failed=true;}assert(failed&&nativeCalls==3);
}
''')


def mirror_header(app,rev=None):
    src=source(f'applications/{app}/gpu/GpuResidentStrict.H',rev)
    names=['downloadSstToHostMirror','downloadNutToHostMirror','writeEpsGPrevRestartMirror','writeSourceResidualRestartMirror']
    methods='\n'.join(h.block(src,'    void '+n) for n in names)
    methods+='\n'+h.block(src,'    bool readEpsGPrevRestartMirrorIfPresent')+'\n'+h.block(src,'    label writeParticleRestartMirror')

    offset=0
    while True:
        pos=src.find('    void downloadToHostMirror',offset)
        if pos<0:break
        part=h.block(src[pos:],'    void downloadToHostMirror');methods+='\n'+part;offset=pos+len(part)
    for name in ['assignScalar','assignVector','finiteMirrorScalar','validateMirrorScalar','validateMirrorVector']:
        sig='    static '+('bool ' if name=='finiteMirrorScalar' else 'void ')+name
        if sig in src:methods+='\n'+h.block(src,sig)
    methods=methods.replace('return (value == value)', '++finiteChecks;return (value == value)')
    pre=h.CUDA+HEADER_PRELUDE.replace('struct Time{std::string path;bool write=true;bool writeTime()const{return write;}};', 'struct Time{std::string path;bool write=true;bool writeTime()const{return write;}const std::string&timePath()const{return path;}};')
    pre+=r'''
#include <filesystem>
#include <tuple>
#include "RESTART"
namespace gpuFshRestart=Foam::gpuFshRestart;
constexpr double GREAT=1e30;double mag(double x){return std::abs(x);}
struct vector{double data[3];vector(double x=0,double y=0,double z=0):data{x,y,z}{}};
using vectorField=Field<vector>;struct volVectorField{vectorField values=vectorField(2);vectorField&primitiveFieldRef(){return values;}};
void mkDir(const std::string&p){std::filesystem::create_directories(p);}
bool invalidValues=false;int boundaryCalls=0,savedVelocityCalls=0;std::vector<double>epsInput;
double value(int i){return invalidValues?(i==0?std::nan(""):-i-1.):i+.25;}
int ugkwpGpuResidentStrictDownloadSst(void*,double*k,double*w,double*n){++nativeCalls;for(int i=0;i<2;++i){k[i]=value(i);w[i]=value(i+2);n[i]=value(i+4);}return nativeFault;}
int ugkwpGpuResidentStrictDownloadNut(void*,double*n){++nativeCalls;for(int i=0;i<2;++i)n[i]=value(i);return nativeFault;}
template<class...P>int ugkwpGpuResidentStrictDownloadFields(void*,P...p){++nativeCalls;int index=0;auto fill=[&](double*d){d[0]=value(index++);d[1]=value(index++);};(fill(p),...);return nativeFault;}
int ugkwpGpuResidentStrictDownloadEpsGPrev(void*,double*p){++nativeCalls;p[0]=.25;p[1]=.75;return nativeFault;}
int ugkwpGpuResidentStrictUploadEpsGPrev(void*,const double*p){++nativeCalls;epsInput={p[0],p[1]};return nativeFault;}
int ugkwpGpuResidentStrictDownloadSourceResidualMass(void*,int*n,int cap,int*f,double*r){++nativeCalls;*n=2;if(cap){f[0]=2;f[1]=5;r[0]=.125;r[1]=-.25;}return nativeFault;}
template<class...P>int ugkwpGpuResidentStrictDownloadParticleRestartMirror(void*,int*n,int cap,P...p){++nativeCalls;*n=1;if(cap){int index=0;auto fill=[&](auto d){if constexpr(std::is_pointer_v<decltype(d)>){int width=index==18?gpuFshRestart::coldWallAxialNodeCount:index==19?gpuFshRestart::coldWallRadialRingCount:index==22?gpuFshRestart::coldWall2DNodeCount:index==23?gpuFshRestart::coldWall2DRadialNodeCount:1;for(int j=0;j<width;++j)d[j]=static_cast<std::remove_pointer_t<decltype(d)>>(index+j+1);}++index;};(fill(p),...);}return nativeFault;}
int ugkwpGpuResidentStrictDownloadParticleSavedVelocity(void*,int n,double*x,double*y,double*z){++savedVelocityCalls;for(int i=0;i<n;++i){x[i]=3.25;y[i]=4.5;z[i]=-6.75;}return nativeFault;}
struct Resident{void*handle_=this;int hostMirrorCells_=2;
 void requireInitialised(const char*)const{if(!handle_)throw std::runtime_error("missing handle");}void check(int e,const char*)const{if(e)throw std::runtime_error("native error");}
 static fileName particleRestartFile(const Time&t){return t.path+"/particles";}static fileName epsGPrevRestartFile(const Time&t){return t.path+"/eps";}static fileName sourceResidualRestartFile(const Time&t){return t.path+"/residual";}
 void downloadGasBoundaryToHostMirror(volScalarField&,volVectorField&,volScalarField&,volScalarField&){++boundaryCalls;}
METHODS
};
unsigned long long hash(const void*p,size_t n){auto*b=(const unsigned char*)p;unsigned long long v=1469598103934665603ULL;for(size_t i=0;i<n;++i)v=(v^b[i])*1099511628211ULL;return v;}
'''.replace('RESTART',f'applications/{app}/gpu/'+('GpuChtParticleRestart.H' if app=='CHT' else 'GpuFshParticleRestart.H')).replace('METHODS',methods)
    return pre


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_mirror_assignment_restart_bytes_and_off_time_errors(tmp_path,app):
    main=r'''
int main(){Resident r;Time t{DIR};std::filesystem::create_directories(t.path);t.write=WRITE;invalidValues=BAD;volScalarField k,w,n,rho,e,p,T,eps,rhoEs,rhoDs,rhoHp,theta,Tp,diam;volVectorField rhoU,U,rhoUs,Us;
 r.downloadSstToHostMirror(t,k,w,n);r.downloadNutToHostMirror(t,n);assert(n.boundaryCalls==1);
 r.downloadToHostMirror(t,rho,rhoU,e,U,p,T,eps,rhoUs,rhoEs,rhoDs,rhoHp,Us,theta,Tp,diam);
 auto print=[&](auto&f){const auto*bytes=reinterpret_cast<const unsigned char*>(f.values.data());const size_t n=f.values.size()*sizeof(f.values[0]);for(size_t i=0;i<n;++i)std::cout<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(bytes[i]);std::cout<<std::dec<<" ";};
 std::cout<<"DATA ";print(k);print(w);print(n);print(rho);print(rhoU);print(e);print(U);print(p);print(T);print(eps);print(rhoUs);print(rhoEs);print(rhoDs);print(rhoHp);print(Us);print(theta);print(Tp);print(diam);std::cout<<"\n";
 r.downloadToHostMirror(t,rho,rhoU,e,U,p,T);std::cout<<"GAS ";print(rho);print(rhoU);print(e);print(U);print(p);print(T);std::cout<<"\n";
 r.writeParticleRestartMirror(t);assert(savedVelocityCalls==1&&boundaryCalls==2);assert(std::filesystem::exists(t.path+"/particles")&&std::filesystem::exists(t.path+"/eps")&&std::filesystem::exists(t.path+"/residual"));
 std::cout<<"COST "<<nativeCalls<<" "<<finiteChecks<<"\n";
 if(BAD){ // Genuine missing bytes and native errors remain errors; foreign metadata is not one.
  {std::ofstream f(t.path+"/eps");f<<"-8 -2 3\n";}assert(r.readEpsGPrevRestartMirrorIfPresent(t)&&epsInput[0]==0&&epsInput[1]==1);
  {std::ofstream f(t.path+"/eps");f<<"2 0.5\n";}bool failed=false;try{r.readEpsGPrevRestartMirrorIfPresent(t);}catch(const std::runtime_error&){failed=true;}assert(failed);
  nativeFault=1;failed=false;try{r.downloadSstToHostMirror(t,k,w,n);}catch(const std::runtime_error&){failed=true;}assert(failed);nativeFault=0;
  std::filesystem::remove(t.path+"/particles");std::filesystem::create_directory(t.path+"/particles");failed=false;try{r.writeParticleRestartMirror(t);}catch(const std::runtime_error&){failed=true;}assert(failed); // actual rename fails against existing directory
 }
}
'''
    result={}
    for label,rev in [('old',BASE),('current',None),('admitted',None)]:
        d=tmp_path/label;d.mkdir();m=main.replace('DIR',json.dumps(str(d/'files'))).replace('WRITE','false' if label=='admitted' else 'true').replace('BAD','true' if label=='admitted' else 'false')
        result[label]=h.run_cpp(d,mirror_header(app,rev)+m,[ROOT/'common'])
    assert [x for x in result['old'].splitlines() if not x.startswith('COST')]==[x for x in result['current'].splitlines() if not x.startswith('COST')]
    assert result['current'].splitlines()[-1].endswith(' 0')
    for name in ['particles','eps','residual']:assert (tmp_path/'old/files'/name).read_bytes()==(tmp_path/'current/files'/name).read_bytes()
    (tmp_path/'mirror-parity.json').write_text(json.dumps(result,indent=2))


def test_actual_pending_ledger_wrappers_keep_addressing_and_noops(tmp_path):
    names=['peekPendingWallEnergyLedgers','restorePendingWallEnergyLedgers','resetGasWallEnergy','resetParticleWallHeatLedgers','peekParticleWallHeatLedgers']
    for label,rev in [('old',BASE),('current',None),('admitted',None)]:
        d=tmp_path/label;d.mkdir();src=source('applications/CHT/gpu/GpuResidentStrict.H',rev)
        methods='\n'.join(h.block(src,'    void '+n) for n in names)
        code=ledger.native(rev)+HEADER_PRELUDE+r'''
struct Resident{void*handle_;bool particleWallHeatTransferEnabled_=true;int hostMirrorFaces_=6;labelList gasWallEnergyFaceIds_,gasWallEnergyPatchSizes_;
 void requireInitialised(const char*)const{assert(handle_);}void check(int e,const char*)const{if(e)throw std::runtime_error("native error");}
METHODS
};
int main(){GpuWallEnergyState s,d;s.deviceState=&d;s.nFaces=6;s.nInternalFaces=2;double g[]={1,2,3,4,5,6},p[]={7,8,9,10,11,12},r[]={13,14,15,16,17,18};unsigned char mask[6]{};s.gasWallEnergy=g;s.gasWallEnergyMask=mask;s.particleWallDepositedEnergy=p;s.particleWallReflectedEnergy=r;s.particleWallHeatTransferEnabled=1;
 Resident owner{&s};owner.gasWallEnergyFaceIds_={2,3,5};owner.gasWallEnergyPatchSizes_={2,0,1};if(BAD)owner.gasWallEnergyFaceIds_={2,99,5,88,77};scalarField a,b,c;counters();owner.peekPendingWallEnergyLedgers(a,b,c);assert(a[0]==3&&a[1]==4&&a[2]==6&&b[0]==9&&c[2]==18&&d2h==6);
 a[0]=23;b[1]=29;c[2]=31;counters();owner.restorePendingWallEnergyLedgers(a,b,c);assert(g[2]==23&&p[3]==29&&r[5]==31&&h2d==6);
 owner.particleWallHeatTransferEnabled_=false;s.particleWallHeatTransferEnabled=0;counters();owner.peekParticleWallHeatLedgers(a,b);assert(a.empty()&&b.empty()&&copies==0);owner.resetParticleWallHeatLedgers();assert(memsets==0&&p[3]==29);
 owner.gasWallEnergyFaceIds_.clear();owner.resetGasWallEnergy();assert(memsets==0&&g[2]==23);
}
'''.replace('METHODS',methods).replace('BAD','true' if label=='admitted' else 'false')
        h.run_cpp(d,code)


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_native_old_success_bytes_costs_and_zero_cell_download(tmp_path,app):
    main=r'''
#include <iostream>
#include <iomanip>
void dump(const double*p,int n){for(int i=0;i<n;++i){unsigned long long b;std::memcpy(&b,p+i,8);std::cout<<" "<<b;}std::cout<<"\n";}
void cost(){std::cout<<"COST "<<copies<<" "<<h2d<<" "<<d2h<<" "<<memsets<<" "<<syncs<<" "<<copiedBytes<<" "<<zeroedBytes<<"\n";}
int main(){DeviceState s;double k[]={1.25,2.5},w[]={3.75,4.5},n[]={-0.,5.25},a[4],b[4],c[4];s.k=k;s.omega=w;s.nut=n;counters();assert(ugkwpGpuResidentStrictDownloadSst(&s,a,b,c)==0);dump(a,2);dump(b,2);dump(c,2);cost();
 s.nCells=0;s.k=s.omega=s.nut=nullptr;counters();assert(ugkwpGpuResidentStrictDownloadSst(&s,a,b,c)==0);cost();s.nCells=2;
 int faces[]={2,3},out[2],count=0;s.sourceFace=faces;s.sourceResidualMass=w;counters();assert(ugkwpGpuResidentStrictDownloadSourceResidualMass(&s,&count,2,out,a)==0);dump(a,2);cost();assert(out[0]==2&&out[1]==3);
 APP
}
'''
    cht=r'''
 double t[]={300,400},gas[4]={1,2,3,4},riemann[4]={5,6,7,8};s.gasBoundaryT=gas;s.riemannBoundaryT=riemann;counters();assert(ugkwpGpuResidentStrictUploadGasBoundaryTemperaturePatch(&s,2,2,t)==0);dump(gas,4);dump(riemann,4);cost();
 double area[]={1,2,3,4},scale[]={.5,1,.25,.125};s.particleWallRepresentedContactArea=area;s.particleWallContactAreaScale=scale;counters();assert(ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea(&s,4,a)==0);dump(a,4);cost();std::cout<<"PROOF "<<finiteChecks<<"\n";
'''
    fsh=r'''
 double dep[]={1,2,3,4},ref[]={5,6,7,8};s.particleWallDepositedEnergy=dep;s.particleWallReflectedEnergy=ref;counters();assert(ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers(&s,4,a,b)==0);dump(a,4);dump(b,4);dump(dep,4);dump(ref,4);cost();
'''
    result={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();result[label]=h.run_cpp(d,native(app,rev)+main.replace('APP',cht if app=='CHT' else fsh))
    assert [x for x in result['old'].splitlines() if not x.startswith('PROOF')]==[x for x in result['current'].splitlines() if not x.startswith('PROOF')]
    if app=='CHT':assert 'PROOF 10' in result['old'] and 'PROOF 0' in result['current']
    (tmp_path/'native-parity.json').write_text(json.dumps(result,indent=2))


def test_fsh_disabled_direct_ledger_uses_retained_payload_and_real_errors(tmp_path):
    h.run_cpp(tmp_path,native('FSH')+r'''
int main(){DeviceState s;s.particleWallHeatTransferEnabled=0;double dep[]={1,2,3,4},ref[]={5,6,7,8},a[4],b[4];s.particleWallDepositedEnergy=dep;s.particleWallReflectedEnergy=ref;
 counters();assert(ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers(&s,2,a,b)==0);assert(a[0]==1&&a[1]==2&&b[1]==6&&dep[0]==0&&dep[2]==3&&ref[2]==7&&d2h==2&&memsets==2&&syncs==1);
 for(int kind=0;kind<3;++kind)for(int f=1;f<=(kind==2?1:2);++f){for(int i=0;i<4;++i){dep[i]=i+1;ref[i]=i+5;}counters();if(kind==0)failCopy=f;if(kind==1)failMemset=f;if(kind==2)failSync=f;
 assert(ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers(&s,4,a,b)!=0);
 if(kind==0)assert(copies==f&&memsets==0&&syncs==0);if(kind==1)assert(copies==2&&memsets==f&&syncs==0);if(kind==2)assert(copies==2&&memsets==2&&syncs==1);
 }
}
''')


def test_native_ledger_old_success_exact_payload_and_operation_cost(tmp_path):
    main=r'''
#include <iostream>
void dump(const double*p,int n){for(int i=0;i<n;++i){unsigned long long b;std::memcpy(&b,p+i,8);std::cout<<b<<" ";}std::cout<<"\n";}
int main(){GpuWallEnergyState s,d;s.deviceState=&d;s.nFaces=4;s.nInternalFaces=1;double g[]={-0.,std::nan(""),std::numeric_limits<double>::infinity(),7},p[]={1,2,3,4},r[]={5,6,7,8},a[4],b[4],c[4];unsigned char mask[4]{};s.gasWallEnergy=g;s.gasWallEnergyMask=mask;s.particleWallDepositedEnergy=p;s.particleWallReflectedEnergy=r;s.particleWallHeatTransferEnabled=1;
 counters();assert(ugkwpGpuResidentStrictPeekGasWallEnergy(&s,4,a)==0);dump(a,4);
 assert(ugkwpGpuResidentStrictPeekWallEnergyLedgerRange(&s,1,2,a,b,c)==0);dump(a,2);dump(b,2);dump(c,2);
 s.particleWallHeatTransferEnabled=0;assert(ugkwpGpuResidentStrictPeekWallEnergyLedgerRange(&s,1,2,a,b,c)==0);dump(a,2);dump(b,2);dump(c,2);s.particleWallHeatTransferEnabled=1;
 assert(ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers(&s,4,b,c)==0);dump(b,4);dump(c,4);dump(p,4);dump(r,4);
 assert(ugkwpGpuResidentStrictDownloadAndResetGasWallEnergy(&s,4,a)==0);dump(a,4);dump(g,4);
 std::cout<<"COST "<<copies<<" "<<d2h<<" "<<h2d<<" "<<memsets<<" "<<syncs<<" "<<copiedBytes<<" "<<zeroedBytes<<"\n";}
'''
    results={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();results[label]=h.run_cpp(d,ledger.native(rev)+main)
    assert results['old']==results['current']
    (tmp_path/'ledger-parity.json').write_text(json.dumps(results,indent=2))
