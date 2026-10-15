"""Actual native API bodies with recording CUDA boundaries; no GPU claim."""
from pathlib import Path
import importlib.util
import json
import re
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
BASE='d7f49c0ae5d25de3a1d629d505725e4f98892315'
spec=importlib.util.spec_from_file_location('native_helpers',ROOT/'tests/test_native_cleanup.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def text(app,revision='working'):
    path='applications/'+app+('/gpu/' if app=='CHT' else '/private_backend/')+'GpuResidentStrict.cu'
    return (ROOT/path).read_text() if revision=='working' else subprocess.check_output(['git','show',revision+':'+path],cwd=ROOT,text=True)


STATE=r'''
#include <iostream>
#include <iomanip>
#include <string>
using GpuReal=double;
#define GPU_R(x) double(x)
struct DeviceState {
 int nFaces=5,nInternalFaces=1,nCells=2,nBoundarySources=3;
 int *sourceFace=nullptr;double *sourceResidualMass=nullptr;
 int nScheduledInletFaces=0,nPressureScheduleRows=0,nVolumeFractionScheduleRows=0;
 int *scheduledInletFaceMask=nullptr;double scheduledInletTemperature=0;
 double *pressureScheduleTimes=nullptr,*pressureScheduleValues=nullptr;
 double *volumeFractionScheduleTimes=nullptr,*volumeFractionScheduleValues=nullptr;
 bool particlesMayBePresent=false;DeviceState* deviceState=this;
 int reductionBlockThreads=32,particleCapacity=3,particleWorkGrid=1,particleBlockThreads=32;
 int csrCellLocalPathEnabled=1,preBaseDirectoryReady=1;
 double radiationMobileMass[2]={99,99},radiationMobileTemperatureMass[2]={99,99},radiationMobileDiameterMass[2]={99,99};
 double enthalpy[2]={99,99},epsHistory[2]={.25,.75};
};
char lastError[2048];int packedSums=0,atomicSums=0,sumClears=0,packedRefresh=0,enthalpyClears=0,atomicEnthalpy=0,recoverEnthalpy=0;
DeviceState* asState(void* p){return static_cast<DeviceState*>(p);}
int validateState(DeviceState* p,const char*){return !p;}
void setLastErrorText(const char* s){std::snprintf(lastError,sizeof(lastError),"%s",s);}
void setLastError(const char* s,int){setLastErrorText(s);}
template<class T>int allocate(T*& p,size_t n,const char*){if(!n){p=nullptr;return 0;}return cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T));}
template<class T>void release(T*& p){cudaFree(p);p=nullptr;}
template<class T>int copyToDevice(T* d,const T* s,size_t n,const char*){if(!n)return 0;if(!d||!s)return 1;return cudaMemcpy(d,s,n*sizeof(T),1);}
template<class T>int copyToHost(T* d,const T* s,size_t n,const char*){if(!n)return 0;if(!d||!s)return 1;return cudaMemcpy(d,s,n*sizeof(T),2);}
void clearMobileParticleRadiationSumsKernel(DeviceState*s){++sumClears;for(int i=0;i<2;++i)s->radiationMobileMass[i]=s->radiationMobileTemperatureMass[i]=s->radiationMobileDiameterMass[i]=0;}
// These are recording kernel boundaries, not substitute physics evidence.
void accumulatePackedMobileParticleRadiationSumsKernel(DeviceState*s){++packedSums;s->radiationMobileMass[0]=2;s->radiationMobileTemperatureMass[0]=600;s->radiationMobileDiameterMass[0]=.004;s->radiationMobileMass[1]=s->radiationMobileTemperatureMass[1]=s->radiationMobileDiameterMass[1]=0;}
void accumulateMobileParticleRadiationSumsAtomicKernel(DeviceState*s){++atomicSums;s->radiationMobileMass[0]+=2;s->radiationMobileTemperatureMass[0]+=600;s->radiationMobileDiameterMass[0]+=.004;}
void refreshPackedParticleEnthalpyKernel(DeviceState*s){++packedRefresh;s->enthalpy[0]=13;s->enthalpy[1]=17;}
void clearParticleEnthalpyMomentKernel(DeviceState*s){++enthalpyClears;s->enthalpy[0]=s->enthalpy[1]=0;}
void accumulateParticleEnthalpyMomentAtomicKernel(DeviceState*s){++atomicEnthalpy;s->enthalpy[0]+=13;s->enthalpy[1]+=17;}
void recoverParticleEnthalpyMomentAtomicKernel(DeviceState*){++recoverEnthalpy;}
void reset(){counters();packedSums=atomicSums=sumClears=packedRefresh=enthalpyClears=atomicEnthalpy=recoverEnthalpy=0;}
void releaseSchedule(DeviceState&s){release(s.scheduledInletFaceMask);release(s.pressureScheduleTimes);release(s.pressureScheduleValues);release(s.volumeFractionScheduleTimes);release(s.volumeFractionScheduleValues);}
'''


def code(app,revision='working',radiation=False):
    src=text(app,revision)
    names=['ugkwpGpuResidentStrictUploadSourceResidualMass','ugkwpGpuResidentStrictConfigureScheduledInlet']
    if radiation:names+=['ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums','ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked']
    publish=(ROOT/'common/operators/publishScheduledInletConfigurationKernel.cuh').read_text()
    publish=h.block(publish,'__global__ void publishScheduledInletConfigurationKernel')
    result=h.CUDA+STATE+'\n#define GPU_OPERATOR_REAL double\n#define GPU_OPERATOR_TIME double\n'+publish+'\n'
    for name in names:
        body=h.block(src,'extern "C" int '+name)
        result+=re.sub(r'(\w+)\s*<<<.*?>>>',r'(++launches, \1)',body,flags=re.S)+'\n'
    return result


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_residual_actual_copy_no_certificate(tmp_path,app):
    source=code(app)+r'''
int main(){DeviceState s;int faces[]={1,2,3};double storage[]={17,19,23};s.sourceFace=faces;s.sourceResidualMass=storage;
 int foreign[]={4,4};double values[]={-.25,std::numeric_limits<double>::infinity()};reset();
 assert(ugkwpGpuResidentStrictUploadSourceResidualMass(&s,2,foreign,values)==0);
 assert(storage[0]==-.25&&std::isinf(storage[1])&&storage[2]==23);assert(d2h==0&&h2d==1&&copies==1);
 reset();assert(ugkwpGpuResidentStrictUploadSourceResidualMass(&s,0,nullptr,nullptr)==0&&copies==0&&storage[0]==-.25);
 reset();assert(ugkwpGpuResidentStrictUploadSourceResidualMass(&s,1,nullptr,values)==0&&h2d==1);
 reset();assert(ugkwpGpuResidentStrictUploadSourceResidualMass(&s,1,faces,nullptr)!=0);
 reset();failCopy=1;assert(ugkwpGpuResidentStrictUploadSourceResidualMass(&s,2,foreign,values)!=0);
 std::cout<<"residual copies=1 d2h=0 h2d=1 prefix=2\n";}
'''
    h.run_cpp(tmp_path,source)


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_scheduled_actual_empty_and_supplied_tables(tmp_path,app):
    source=code(app)+r'''
int main(){DeviceState s,published;s.deviceState=&published;published.nPressureScheduleRows=99;published.scheduledInletTemperature=999;reset();
 assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&s,0,nullptr,-3,0,nullptr,nullptr,0,nullptr,nullptr)==0);
 assert(s.nScheduledInletFaces==0&&s.nPressureScheduleRows==0&&s.nVolumeFractionScheduleRows==0&&s.scheduledInletTemperature==-3);
 assert(published.nScheduledInletFaces==0&&published.nPressureScheduleRows==0&&published.nVolumeFractionScheduleRows==0&&published.scheduledInletTemperature==-3);
 assert(published.scheduledInletFaceMask==s.scheduledInletFaceMask&&published.pressureScheduleTimes==nullptr);
 assert(allocations==1&&h2d==1&&launches==1&&syncs==1);releaseSchedule(s);
 int faces[]={0,2,2};double pt[]={2,1,1},pv[]={-3,4,std::numeric_limits<double>::infinity()},vt[]={-2,1},vv[]={-1,2};reset();
 assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&s,3,faces,-7,3,pt,pv,2,vt,vv)==0);
 assert(s.scheduledInletFaceMask[0]==1&&s.scheduledInletFaceMask[2]==1&&s.nScheduledInletFaces==3);
 assert(std::memcmp(s.pressureScheduleTimes,pt,sizeof pt)==0&&std::memcmp(s.pressureScheduleValues,pv,sizeof pv)==0);
 assert(std::memcmp(s.volumeFractionScheduleValues,vv,sizeof vv)==0&&s.particlesMayBePresent);
 assert(published.nScheduledInletFaces==3&&published.nPressureScheduleRows==3&&published.nVolumeFractionScheduleRows==2&&published.scheduledInletTemperature==-7&&published.particlesMayBePresent);
 assert(published.scheduledInletFaceMask==s.scheduledInletFaceMask&&published.pressureScheduleTimes==s.pressureScheduleTimes&&published.pressureScheduleValues==s.pressureScheduleValues&&published.volumeFractionScheduleTimes==s.volumeFractionScheduleTimes&&published.volumeFractionScheduleValues==s.volumeFractionScheduleValues);
 assert(allocations==5&&h2d==5&&d2h==0&&launches==1&&syncs==1);
 // An old successful reconfiguration with no future particles cannot clear the
 // monotonic fact that particles may already exist.
 double goodTime[]={0,1},goodPressure[]={100000,120000},zeroVolume[]={0,0};int goodFaces[]={1,3};reset();
 assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&s,2,goodFaces,300,2,goodTime,goodPressure,2,goodTime,zeroVolume)==0);
 assert(s.particlesMayBePresent&&published.particlesMayBePresent&&published.nPressureScheduleRows==2&&published.scheduledInletTemperature==300&&s.scheduledInletFaceMask[1]==1&&s.scheduledInletFaceMask[3]==1);releaseSchedule(s);
 for(int fault=1;fault<=5;++fault){reset();failAlloc=fault;assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&s,3,faces,-7,3,pt,pv,2,vt,vv)!=0);releaseSchedule(s);}
 for(int fault=1;fault<=5;++fault){reset();failCopy=fault;assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&s,3,faces,-7,3,pt,pv,2,vt,vv)!=0);releaseSchedule(s);}
 reset();failLaunch=1;assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&s,3,faces,-7,3,pt,pv,2,vt,vv)!=0);releaseSchedule(s);
 reset();failSync=1;assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&s,3,faces,-7,3,pt,pv,2,vt,vv)!=0);releaseSchedule(s);
 reset();assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&s,1,nullptr,1,0,nullptr,nullptr,0,nullptr,nullptr)!=0);
 std::cout<<"schedule allocations=5 h2d=5 d2h=0 publication=1 sync=1\n";}
'''
    h.run_cpp(tmp_path,source)


def test_radiation_actual_dispatch_and_errors(tmp_path):
    source=code('CHT',radiation=True)+r'''
int main(){
 for(int enabled:{0,1})for(int ready:{0,1})for(int capacity:{0,3})for(int work:{0,1}){
  DeviceState s;s.csrCellLocalPathEnabled=enabled;s.preBaseDirectoryReady=ready;s.particleCapacity=capacity;s.particleWorkGrid=work;
  double mass[2],temp[2],diam[2];reset();
  assert(ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums(&s,19,mass,temp,diam)==0);
  bool packed=enabled&&ready&&capacity&&work;bool active=capacity&&work;
  assert(packedSums==int(packed)&&atomicSums==int(active&&!packed)&&sumClears==int(!packed)&&d2h==3);
  assert(mass[0]==(active?2:0)&&mass[1]==0&&temp[0]==(active?600:0));
  reset();assert(ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(&s)==0);
  assert(packedRefresh==int(enabled&&ready)&&enthalpyClears==int(!(enabled&&ready))&&recoverEnthalpy==int(!(enabled&&ready))&&syncs==1);
  assert(atomicEnthalpy==int(!(enabled&&ready)&&active)&&s.epsHistory[0]==.25&&s.epsHistory[1]==.75);
 }
 for(int ready:{0,1}){
  DeviceState s;s.preBaseDirectoryReady=ready;double m[2],t[2],d[2];
  for(int f=1;f<=(ready?1:2);++f){reset();failLaunch=f;assert(ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums(&s,2,m,t,d)!=0);}
  for(int f=1;f<=3;++f){reset();failCopy=f;assert(ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums(&s,2,m,t,d)!=0);}
  for(int f=1;f<=(ready?1:3);++f){reset();failLaunch=f;assert(ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(&s)!=0);}
  reset();failSync=1;assert(ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(&s)!=0);
 }
 std::cout<<"packed clear=0 atomic/no-work clear=1 downloads=3 refresh-sync=1\n";}
'''
    h.run_cpp(tmp_path,source)


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_old_success_exact_payload_and_work_counts(tmp_path,app):
    main=r'''
void data(const double* p,int n){std::cout<<"DATA"<<std::hexfloat;for(int i=0;i<n;++i)std::cout<<" "<<p[i];std::cout<<"\n";}
int main(){DeviceState s,published;s.deviceState=&published;int faces[]={1,2,3};double residual[]={.125,.5,1.25},storage[3]{};s.sourceFace=faces;s.sourceResidualMass=storage;
 reset();assert(ugkwpGpuResidentStrictUploadSourceResidualMass(&s,3,faces,residual)==0);data(storage,3);
 std::cout<<"COST residual "<<d2h<<" "<<h2d<<" "<<copiedBytes<<"\n";
 double pt[]={0,.125,1},pv[]={100000,120000,90000},vt[]={0,.5},vv[]={0,.125};reset();
 assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&s,3,faces,350,3,pt,pv,2,vt,vv)==0);
 data(s.pressureScheduleTimes,3);data(s.pressureScheduleValues,3);data(s.volumeFractionScheduleTimes,2);data(s.volumeFractionScheduleValues,2);data(&s.scheduledInletTemperature,1);
 std::cout<<"DATA "<<s.nScheduledInletFaces<<" "<<s.nPressureScheduleRows<<" "<<s.nVolumeFractionScheduleRows<<" "<<s.particlesMayBePresent;for(int i=0;i<s.nFaces;++i)std::cout<<" "<<s.scheduledInletFaceMask[i];std::cout<<"\n";
 std::cout<<"COST schedule "<<allocations<<" "<<h2d<<" "<<d2h<<" "<<launches<<" "<<syncs<<" "<<pressureProof<<" "<<volumeProof<<"\n";
 data(published.pressureScheduleValues,3);data(published.volumeFractionScheduleValues,2);data(&published.scheduledInletTemperature,1);
 assert(published.nScheduledInletFaces==3&&published.nPressureScheduleRows==3&&published.nVolumeFractionScheduleRows==2&&published.particlesMayBePresent);
 releaseSchedule(s);s.deviceState=&s;
#ifdef RADIATION
 for(int enabled:{0,1})for(int work:{0,1}){s.csrCellLocalPathEnabled=enabled;s.preBaseDirectoryReady=1;s.particleWorkGrid=work;double m[2],t[2],d[2];reset();
 assert(ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums(&s,2,m,t,d)==0);data(m,2);data(t,2);data(d,2);
 std::cout<<"COST sums "<<enabled<<" "<<work<<" "<<sumClears<<" "<<packedSums<<" "<<atomicSums<<" "<<d2h<<"\n";
 reset();assert(ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(&s)==0);data(s.enthalpy,2);data(s.epsHistory,2);
 std::cout<<"COST refresh "<<enabled<<" "<<work<<" "<<packedRefresh<<" "<<enthalpyClears<<" "<<atomicEnthalpy<<" "<<recoverEnthalpy<<" "<<syncs<<"\n";}
#endif
}
'''
    results={}
    for label,revision in [('old',BASE),('current','working')]:
        folder=tmp_path/label;folder.mkdir()
        src=code(app,revision,radiation=app=='CHT')
        src='int pressureProof=0,volumeProof=0;\n'+src
        src=src.replace('for (int i = 0; i < nPressureRows; ++i)\n    {',
                        'for (int i = 0; i < nPressureRows; ++i)\n    { ++pressureProof;')
        src=src.replace('!std::isfinite(volumeFractionTimes[i])',
                        '(++volumeProof,!std::isfinite(volumeFractionTimes[i]))')
        result=h.run_cpp(folder,src+ ('\n#define RADIATION\n' if app=='CHT' else '')+main)
        results[label]=result.splitlines()
    assert [x for x in results['old'] if x.startswith('DATA')]==[x for x in results['current'] if x.startswith('DATA')]
    assert 'COST residual 1 1 36' in results['old']
    assert 'COST residual 0 1 24' in results['current']
    assert 'COST schedule 5 5 0 1 1 3 2' in results['old']
    assert 'COST schedule 5 5 0 1 1 0 0' in results['current']
    if app=='CHT':
        assert 'COST sums 1 1 1 1 0 3' in results['old']
        assert 'COST sums 1 1 0 1 0 3' in results['current']
        for prefix in ['COST refresh','COST sums 0','COST sums 1 0']:
            assert [x for x in results['old'] if x.startswith(prefix)]==[x for x in results['current'] if x.startswith(prefix)]
    (tmp_path/'parity-cost.json').write_text(json.dumps(results,indent=2)+'\n')


@pytest.mark.parametrize('app',['CHT','FSH'])
def test_actual_host_readers_and_supplied_configuration(tmp_path,app):
    src=(ROOT/f'applications/{app}/gpu/GpuResidentStrict.H').read_text()
    methods='\n'.join(h.block(src,sig) for sig in ['    bool readSourceResidualRestartMirrorIfPresent','    void configureScheduledInlet'])
    if app=='CHT':methods+='\n'+h.block(src,'    void downloadMobileParticleRadiationMaterial')+'\n'+h.block(src,'    void refreshParticleEnthalpyAfterRadiation')
    prelude=r'''
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using label=int;using scalar=double;using string=std::string;using fileName=std::string;
template<class T>struct Field:std::vector<T>{using std::vector<T>::vector;T* begin(){return this->data();}const T* begin()const{return this->data();}void setSize(size_t n,T value=T{}){this->assign(n,value);}};
using labelList=Field<int>;using scalarField=Field<double>;struct Time{std::string path;};struct fvMesh{int nCells()const{return 2;}};
bool isFile(const fileName&p){return bool(std::ifstream(p));}
#define forAll(f,i) for(size_t i=0;i<(f).size();++i)
#define FatalErrorInFunction std::cerr
int throwingExit(){throw std::runtime_error("actual extraction failure");}
#define exit(x) throwingExit()
int calls=0;double copied=0;int fault=0;double radiationMass=2,radiationTemp=600;
int ugkwpGpuResidentStrictUploadSourceResidualMass(void*,int n,const int*,const double*r){++calls;if(n)copied=r[0];return fault;}
int ugkwpGpuResidentStrictConfigureScheduledInlet(void*,int n,const int*,double t,int np,const double*,const double*,int nv,const double*,const double*){++calls;assert(n==0&&np==0&&nv==0&&t==-3);return fault;}
int ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums(void*,int,double*m,double*t,double*d){++calls;m[0]=radiationMass;m[1]=0;t[0]=radiationTemp;t[1]=0;d[0]=.004;d[1]=0;return fault;}
int ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(void*){++calls;return fault;}
struct Resident {void* handle_=this;const fvMesh* residentMesh_=nullptr;bool particleStateCellLocalPacked_=false;double residentTpMin_=200;
 void requireInitialised(const char*)const{}void check(int e,const char*)const{if(e)throw std::runtime_error("actual backend error");}
 fileName sourceResidualRestartFile(const Time&t)const{return t.path;}
METHODS
};
int main(){Resident r;Time t{PATH};
 {std::ofstream f(t.path);f<<"foreignTag 1\n99 -0.125\n";}assert(r.readSourceResidualRestartMirrorIfPresent(t)&&copied==-.125&&calls==1);
 {std::ofstream f(t.path);f<<"foreignTag 1\n99\n";}bool failed=false;try{r.readSourceResidualRestartMirrorIfPresent(t);}catch(const std::runtime_error&){failed=true;}assert(failed&&calls==1);
 {std::ofstream f(t.path);f<<"foreignTag 1\n99 0.25\n";}fault=1;failed=false;try{r.readSourceResidualRestartMirrorIfPresent(t);}catch(const std::runtime_error&){failed=true;}assert(failed);fault=0;
 assert(!r.readSourceResidualRestartMirrorIfPresent(Time{t.path+".missing"}));
 r.configureScheduledInlet(labelList{},-3,scalarField{},scalarField{},scalarField{},scalarField{});
#ifdef RADIATION
 fvMesh mesh;scalarField m,temp,d;r.downloadMobileParticleRadiationMaterial(mesh,m,temp,d);assert(temp[0]==300&&temp[1]==200&&d[1]==0);
 for(double mass:{-2.,std::numeric_limits<double>::quiet_NaN()}){radiationMass=mass;r.downloadMobileParticleRadiationMaterial(mesh,m,temp,d);assert(temp[0]==200&&d[0]==0);}
 radiationMass=2;radiationTemp=std::numeric_limits<double>::infinity();r.downloadMobileParticleRadiationMaterial(mesh,m,temp,d);assert(std::isinf(temp[0]));
 r.refreshParticleEnthalpyAfterRadiation();fault=1;failed=false;try{r.refreshParticleEnthalpyAfterRadiation();}catch(const std::runtime_error&){failed=true;}assert(failed);
#endif
}
'''
    prelude=prelude.replace('METHODS',methods).replace('PATH',json.dumps(str(tmp_path/'actual-residual.txt')))
    if app=='CHT':prelude='#define RADIATION\n'+prelude
    h.run_cpp(tmp_path,prelude)


def test_fsh_actual_client_server_and_native_tails(tmp_path):
    native=code('FSH').replace('ugkwpGpuResidentStrictConfigureScheduledInlet','peerConfigureScheduledInlet').replace('ugkwpGpuResidentStrictUploadSourceResidualMass','peerUploadSourceResidualMass')
    server=native+r'''
#define ugkwpGpuResidentStrictConfigureScheduledInlet peerConfigureScheduledInlet
#define ugkwpGpuResidentStrictUploadSourceResidualMass peerUploadSourceResidualMass
#define ugkwpGpuResidentStrictLastError peerLastError
#define main unused_backend_main
#include "applications/FSH/private_backend/GpuBackendServer.C"
#undef main
extern "C" const char* ugkwpGpuResidentStrictLastError(){return "injected native failure";}
void serve(int fd){using namespace ugkwpGpuIpc;DeviceState device,published;device.deviceState=&published;int faces[3]={1,2,3};double residual[3]={11,13,17};device.sourceFace=faces;device.sourceResidualMass=residual;
 ServerState state;state.backend=&device;state.nFaces=5;state.nCells=2;
 for(int frame=0;frame<7;++frame){RequestHeader request{};assert(readObject(fd,request));readingFrame=true;frameBytes=frameRemaining=request.payloadBytes;reset();if(frame==5)failCopy=1;if(frame==6)failLaunch=1;
  if(frame<3||frame==6){assert(handleScheduledInletConfiguration(fd,request,state));if(frame==0)assert(device.nPressureScheduleRows==2&&device.pressureScheduleValues[1]==120000&&published.nPressureScheduleRows==2&&published.pressureScheduleValues==device.pressureScheduleValues);if(frame==1)assert(device.nScheduledInletFaces==0&&device.nPressureScheduleRows==0&&published.nScheduledInletFaces==0&&published.nPressureScheduleRows==0&&published.pressureScheduleTimes==nullptr);if(frame==2)assert(device.pressureScheduleTimes[0]==2&&device.pressureScheduleValues[0]==-3&&device.particlesMayBePresent&&published.particlesMayBePresent&&published.scheduledInletTemperature==-7);}
  else {assert(handleUploadResidual(fd,request,state));if(frame==3)assert(residual[0]==-.25&&std::isinf(residual[1])&&residual[2]==17);if(frame>=4)assert(residual[0]==-.25&&residual[2]==17);assert(d2h==0);}
  assert(discardRequest(fd)&&frameRemaining==0);
 }
 releaseSchedule(device);close(fd);
}
'''
    client=r'''
#include "applications/FSH/gpu/GpuBackendClient.C"
#include <cassert>
#include <limits>
#include <thread>
extern void serve(int);
int main(){int fd[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0);ClientState state;state.fd=fd[0];state.nFaces=5;std::thread worker(serve,fd[1]);
 int faces[]={1,3};double time[]={0,1},p[]={100000,120000},v[]={0,.125};
 assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&state,2,faces,300,2,time,p,2,time,v)==0);
 assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&state,0,nullptr,-7,0,nullptr,nullptr,0,nullptr,nullptr)==0);
 double unordered[]={2,1},signedP[]={-3,4},signedV[]={-1,2};assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&state,2,faces,-7,2,unordered,signedP,2,unordered,signedV)==0);
 int foreign[]={4,4};double mass[]={-.25,std::numeric_limits<double>::infinity()};
 assert(ugkwpGpuResidentStrictUploadSourceResidualMass(&state,2,foreign,mass)==0);
 assert(ugkwpGpuResidentStrictUploadSourceResidualMass(&state,0,nullptr,nullptr)==0);
 assert(ugkwpGpuResidentStrictUploadSourceResidualMass(&state,2,foreign,mass)!=0);
 assert(ugkwpGpuResidentStrictConfigureScheduledInlet(&state,2,faces,-7,2,unordered,signedP,2,unordered,signedV)!=0);
 worker.join();close(fd[0]);}
'''
    (tmp_path/'server.cpp').write_text(server);(tmp_path/'client.cpp').write_text(client);exe=tmp_path/'wire'
    command=['g++','-std=c++17','-O0','-pthread','-ffunction-sections','-fdata-sections','-Wl,--gc-sections','-I'+str(ROOT),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'applications/FSH/gpu'),str(tmp_path/'server.cpp'),str(tmp_path/'client.cpp'),'-o',str(exe)]
    result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True,timeout=30);assert result.returncode==0,result.stdout+result.stderr
    (tmp_path/'wire-command.json').write_text(json.dumps(command,indent=2))


def test_pinned_unready_directory_is_real_unfinished_red_case(tmp_path):
    source=code('CHT',BASE,radiation=True)+r'''
int main(){DeviceState s;s.preBaseDirectoryReady=0;double mass[2]={777,777},temperature[2]={777,777},diameter[2]={777,777};reset();
 assert(ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums(&s,2,mass,temperature,diameter)!=0);
 assert(sumClears==1&&packedSums==0&&atomicSums==0&&d2h==0&&mass[0]==777);
 reset();assert(ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(&s)!=0);assert(packedRefresh==0&&enthalpyClears==0&&atomicEnthalpy==0&&recoverEnthalpy==0&&s.enthalpy[0]==99);
 std::cout<<"pinned missing-directory refused before actual producer/download\n";}
'''
    h.run_cpp(tmp_path,source)


def test_actual_fsh_touched_handlers_surplus_and_short_input(tmp_path):
    source=code('FSH')+r'''
#include <sys/socket.h>
#define main unused_backend_main
#include "applications/FSH/private_backend/GpuBackendServer.C"
#undef main
extern "C" const char* ugkwpGpuResidentStrictLastError(){return "native error";}
int main(){using namespace ugkwpGpuIpc;
 for(bool scheduled:{false,true})for(bool truncated:{false,true}){
  int fd[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0);readingFrame=false;
  DeviceState host,published;host.deviceState=&published;double storage[3]={11,13,17};host.sourceResidualMass=storage;
  ServerState state;state.backend=&host;state.nFaces=5;state.nCells=2;
  CountArgs count{};count.count=1;ScheduledInletConfigArgs args{};args.count=1;args.inletTemperature=-7;args.nPressureRows=1;args.nVolumeFractionRows=1;
  const uint64_t required=scheduled?wireBytes(args)+sizeof(int)+4*sizeof(double):wireBytes(count)+sizeof(int)+sizeof(double);
  RequestHeader request{};request.payloadBytes=required+(truncated?0:3);
  if(scheduled)assert(writeObject(fd[0],args));else assert(writeObject(fd[0],count));
  int face=0;double value=-.25;assert(writeObject(fd[0],face));
  const int doubles=scheduled?4:1;
  for(int i=0;i<doubles-(truncated?1:0);++i)assert(writeObject(fd[0],value));
  if(truncated){assert(writeAll(fd[0],&value,sizeof(double)-1));shutdown(fd[0],SHUT_WR);}
  else {unsigned char extra[3]={7,8,9};assert(writeAll(fd[0],extra,3));CountArgs next{};next.count=37;assert(writeObject(fd[0],next));}
  readingFrame=true;frameBytes=frameRemaining=request.payloadBytes;reset();
  bool ok=scheduled?handleScheduledInletConfiguration(fd[1],request,state):handleUploadResidual(fd[1],request,state);
  if(truncated){assert(!ok&&copies==0&&launches==0&&allocations==0&&storage[0]==11);assert(!frameReadError.empty());}
  else {assert(ok&&frameRemaining==0&&!readingFrame);CountArgs next{};assert(readObject(fd[1],next)&&next.count==37);ResponseHeader response{};assert(readObject(fd[0],response)&&response.status==0);
   if(scheduled)assert(published.nPressureScheduleRows==1&&published.pressureScheduleValues[0]==value&&published.scheduledInletTemperature==-7&&h2d==5&&launches==1);
   else assert(storage[0]==value&&storage[1]==13&&d2h==0&&h2d==1);
  }
  readingFrame=false;releaseSchedule(host);close(fd[0]);close(fd[1]);
 }
}
'''
    path=tmp_path/'handlers.cpp';path.write_text(source);exe=tmp_path/'handlers'
    command=['g++','-std=c++17','-O0','-ffunction-sections','-fdata-sections','-Wl,--gc-sections','-I'+str(ROOT),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'applications/FSH/gpu'),str(path),'-o',str(exe)]
    result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True,timeout=30);assert result.returncode==0,result.stdout+result.stderr
    (tmp_path/'handler-command.json').write_text(json.dumps(command,indent=2))
