"""Real CHMT owner lifecycle through an instrumented host CUDA runtime."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest
from test_shared_device_storage import APP, SOURCE, STUB

COUNTERS = STUB.replace('#include <cstring>', '#include <cstring>\n#include <map>\n#include <cassert>')
COUNTERS = COUNTERS.replace('inline int cudaMalloc(void**p,size_t n){*p=std::malloc(n);return *p?0:2;}', '''
inline std::map<void*,size_t> live;
inline int allocs=0,frees=0,zeros=0,syncs=0,viewCopies=0,failAlloc=-1,failCopy=-1,failZero=-1;
inline size_t zeroBytes=0,h2d=0,d2h=0,d2d=0;inline void* viewTarget=nullptr;
inline int cudaMalloc(void**p,size_t n){if(failAlloc==0){failAlloc=-1;return 2;}if(failAlloc>0)--failAlloc;*p=std::malloc(n);if(*p){live[*p]=n;++allocs;}return *p?0:2;}
''')
COUNTERS = COUNTERS.replace('inline int cudaFree(void*p){std::free(p);return 0;}', 'inline int cudaFree(void*p){if(p){assert(live.erase(p)==1);++frees;std::free(p);}return 0;}')
COUNTERS = COUNTERS.replace('inline int cudaMemcpy(void*d,const void*s,size_t n,int){std::memcpy(d,s,n);return 0;}', '''inline int cudaMemcpy(void*d,const void*s,size_t n,int kind){
 if(d==viewTarget)++viewCopies;
 if(kind==1)h2d+=n;else if(kind==2)d2h+=n;else d2d+=n;
 if(failCopy==0){failCopy=-1;if(n)std::memcpy(d,s,n/2);return 3;}if(failCopy>0)--failCopy;
 std::memcpy(d,s,n);return 0;}
''')
COUNTERS = COUNTERS.replace('inline int cudaMemsetAsync(void*p,int value,size_t n,void*){std::memset(p,value,n);return 0;}', 'inline int cudaMemsetAsync(void*p,int value,size_t n,void*){++zeros;zeroBytes+=n;if(failZero==0){failZero=-1;return 3;}if(failZero>0)--failZero;std::memset(p,value,n);return 0;}')
COUNTERS = COUNTERS.replace('inline int cudaDeviceSynchronize(){return 0;}', 'inline int cudaDeviceSynchronize(){++syncs;return 0;}').replace('inline int cudaStreamSynchronize(void*){return 0;}', 'inline int cudaStreamSynchronize(void*){++syncs;return 0;}')

PREFIX = r'''
#include <vector>
struct Saved {void* pointer;std::vector<unsigned char> bytes;};
std::vector<Saved> saved;
template<class T>void seed(T&){ }
template<class T>void seed(T*& p){if(!p)return;const size_t n=live.at(p);std::memset(p,1+saved.size()%120,n);saved.push_back({p,std::vector<unsigned char>((unsigned char*)p,(unsigned char*)p+n)});}
void same(){for(const auto& s:saved){assert(live.count(s.pointer));assert(!std::memcmp(s.pointer,s.bytes.data(),s.bytes.size()));}}
void resetCounts(){allocs=frees=zeros=syncs=viewCopies=0;zeroBytes=h2d=d2h=d2d=0;}
'''
BODY = SOURCE[:SOURCE.index(' chmt::SharedGasDeviceStorage storage;')]
BODY = BODY.replace('int main(){', PREFIX+'\nint main(int argc,char**argv){assert(argc==2);int test=std::atoi(argv[1]);')
BODY += r'''
 model.physics.enableSst=true;h.sst={{4,8}};gas.speciesOrderHash=0;
 {chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));auto& v=storage.hostView();
 auto* device=storage.deviceView();viewTarget=device;
 #define UGKWP_GAS_STATE_FIELD(Type,Name) seed(v.Name);
 #include "common/gasTransport/GasStateFields.inc"
 #undef UGKWP_GAS_STATE_FIELD
 auto& sp=v.gasSpecies;
 #define SEED(name) seed(sp.name)
 SEED(rho);SEED(initial);SEED(flux);SEED(gradX);SEED(gradY);SEED(gradZ);SEED(limiter);SEED(boundaryMassFraction);SEED(positivityScale);SEED(compositionBoundaryFixed);SEED(soundSpeed);SEED(heatCapacity);SEED(gasConstant);SEED(cellStatus);SEED(faceStatus);
 #undef SEED
 // Only real resident solver channels have owners; removed audit arrays do not.
 const auto old=v;auto count=live.size();h.gas[0].mass=999;h.gas[0].energy=777;h.sst[0]={888,444};
 gas.thermoHash+=19;gas.speciesOrderHash=37;gas.coefficients[0]=1140;gas.diffusionCoefficients[0]=.03;
 resetCounts();
 if(test==1){assert(storage.configure(model,gas,{},h,error));same();assert(storage.deviceView()==device);assert(device->rho==old.rho&&device->gasSpecies.rho==old.gasSpecies.rho);assert(device->gasSpecies.thermo.speciesOrderHash==37);assert(device->gasSpecies.thermo.coefficients[0]==1140);assert(device->gasSpecies.diffusivity[0]==.03);
  assert(allocs==3&&frees==3&&zeros==0&&zeroBytes==0&&d2d==0&&d2h==0&&syncs==0&&viewCopies==1);
  const size_t expected=gas.species.size()*sizeof(ugkwp::SpeciesThermoData<double>)+gas.coefficients.size()*sizeof(double)+gas.diffusionCoefficients.size()*sizeof(double)+sizeof(v);assert(h2d==expected);
  std::printf("repeat alloc=%d free=%d zero=%zu H2D=%zu D2D=%zu D2H=%zu sync=%d view=%d\n",allocs,frees,zeroBytes,h2d,d2d,d2h,syncs,viewCopies);
  assert(storage.uploadState(h,error));assert(v.rho[0]==999./2&&v.rhoE[0]==777./2&&v.rhoK[0]==444&&v.rhoOmega[0]==222);
 }else if(test>=2&&test<=5){if(test==2)failAlloc=0;else failCopy=test==3?0:test==4?2:3;
  assert(!storage.configure(model,gas,{},h,error));same();assert(live.size()==count);assert(storage.deviceView()==device);assert(device->gasSpecies.thermo.coefficients==old.gasSpecies.thermo.coefficients);assert(v.gasSpecies.thermo.coefficients==old.gasSpecies.thermo.coefficients);
  failAlloc=failCopy=-1;assert(storage.configure(model,gas,{},h,error));same();
 }else if(test==6){m.volumes={2,4};m.cellCentres={{0,0,0},{2,0,0}};m.cellFaceOffsets={0,1,2};m.cellFaces={0,1};m.owner={0,1};h.gas.resize(2,h.gas[0]);h.gas[1].mass=888;h.gas[1].energy=666;h.gas[1].species[0]=111;h.gas[1].species[1]=777;h.sst.push_back({400,800});
  assert(storage.configure(model,gas,{},h,error));assert(v.nCells==2&&v.rho[0]==999./2&&v.rho[1]==222&&v.rhoE[1]==166.5);assert(v.gasSpecies.rho[1]==111./4&&v.gasSpecies.rho[3]==777./4&&v.rhoK[1]==100);
  assert(storage.deviceView()->rho==v.rho&&storage.deviceView()->nCells==2);
 }
 }assert(live.empty());
}
'''

@pytest.fixture(scope='module')
def lifecycle_probe(tmp_path_factory):
    p=tmp_path_factory.mktemp('shared_reconfigure')
    (p/'cuda_runtime.h').write_text(COUNTERS)
    (p/'probe.cpp').write_text(BODY)
    cmd=['g++','-std=c++17','-O1','-I'+str(p),'-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str(APP.parents[1]),'-I'+str(APP.parents[1]/'common'),str(p/'probe.cpp'),'-o',str(p/'probe')]
    build=subprocess.run(cmd,capture_output=True,text=True);(p/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0,build.stderr
    return p/'probe'

@pytest.mark.parametrize('case',[1,2,3,4,5,6])
def test_reconfigure_retains_or_imports_actual_owned_state(lifecycle_probe,case):
    run=subprocess.run([str(lifecycle_probe),str(case)],capture_output=True,text=True)
    (lifecycle_probe.parent/('case-%s.log'%case)).write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr


def _run_source(tmp_path, source, stub=COUNTERS):
    (tmp_path/'cuda_runtime.h').write_text(stub)
    (tmp_path/'probe.cpp').write_text(source)
    cmd=['g++','-std=c++17','-O1','-I'+str(tmp_path),'-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str(APP.parents[1]),'-I'+str(APP.parents[1]/'common'),str(tmp_path/'probe.cpp'),'-o',str(tmp_path/'probe')]
    b=subprocess.run(cmd,capture_output=True,text=True);(tmp_path/'build.log').write_text(b.stdout+b.stderr)
    assert b.returncode==0,b.stderr
    r=subprocess.run([str(tmp_path/'probe')],capture_output=True,text=True);(tmp_path/'run.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,r.stdout+r.stderr


def test_reconfigure_preserves_completed_wall_then_consumes_new_tables(tmp_path):
    from test_reacting_wall_storage import storage_source
    source=storage_source().split(' // A moving-geometry rebuild')[0]+r'''
 auto* device=storage.deviceView();viewTarget=device;
 auto* input=v.gasBoundaryLayerModel.input;auto* output=v.gasBoundaryLayerModel.output;auto* exchange=v.gasBoundaryLayer.exchange;
 const auto before=*output;const auto budget=*exchange;const auto sampled=*input;
 h.gas[0].mass=888;canonical.coefficients[0]=1140;canonical.thermoHash+=21;
 allocs=frees=zeros=syncs=viewCopies=0;zeroBytes=h2d=d2h=d2d=0;
 assert(storage.configure(model,canonical,{},h,error));assert(storage.deviceView()==device&&v.rho[0]==1);
 assert(input==v.gasBoundaryLayerModel.input&&output==v.gasBoundaryLayerModel.output&&exchange==v.gasBoundaryLayer.exchange);
 assert(!std::memcmp(&before,output,sizeof(before))&&!std::memcmp(&budget,exchange,sizeof(budget))&&!std::memcmp(&sampled,input,sizeof(sampled)));
 assert(allocs==3&&frees==3&&zeros==0&&d2h==0&&d2d==0&&syncs==0&&viewCopies==1);
 assert(ugkwp::evaluateGasBoundaryLayerSlot(*device,0,0));
 assert(input->model.thermo.coefficients==device->gasSpecies.thermo.coefficients&&input->model.thermo.coefficients[0]==1140);
 assert(exchange->ready&&input->matching.temperature==900);
 std::printf("wall repeat alloc=%d free=%d zero=%zu H2D=%zu D2H=%zu D2D=%zu sync=%d view=%d\n",allocs,frees,zeroBytes,h2d,d2h,d2d,syncs,viewCopies);
}
'''
    _run_source(tmp_path,source)


def test_optional_channels_retain_inventory_across_disable_and_enable(tmp_path):
    source=SOURCE[:SOURCE.index(' chmt::SharedGasDeviceStorage storage;')]+r'''
 auto mixture=gas;gas.mode=ugkwp::GasMode::SingleLegacy;model.physics.singleGasSpecies=0;
 {chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));auto& v=storage.hostView();
 assert(!v.gasSpecies.rho);v.rho[0]=31;v.rhoE[0]=41;h.gas[0].mass=222;h.sst={{12,18}};
 model.physics.enableSst=true;assert(storage.configure(model,mixture,{},h,error));
 assert(v.rho[0]==31&&v.rhoE[0]==41&&v.gasSpecies.rho[0]==.25&&v.gasSpecies.rho[1]==.75&&v.rhoK[0]==6&&v.rhoOmega[0]==9);
 auto* rho=v.gasSpecies.rho;auto* rhoK=v.rhoK;auto* rhoOmega=v.rhoOmega;rho[0]=7;v.rhoK[0]=19;v.rhoOmega[0]=23;
 gas=mixture;gas.mode=ugkwp::GasMode::MixtureChemistry;assert(storage.configure(model,gas,{},h,error));
 auto* status=v.gasSpecies.chemistryStatus;assert(!status);
 gas.mode=ugkwp::GasMode::SingleLegacy;model.physics.enableSst=false;assert(storage.configure(model,gas,{},h,error));
 assert(!v.sstConfigured&&v.gasSpecies.rho==rho&&v.rhoK==rhoK&&v.rhoOmega==rhoOmega&&!v.gasSpecies.thermo.species&&!v.gasSpecies.mechanism.reactions);
 gas=mixture;gas.mode=ugkwp::GasMode::MixtureChemistry;model.physics.enableSst=true;h.sst[0]={999,999};
 assert(storage.configure(model,gas,{},h,error));assert(v.rho[0]==31&&v.rhoK[0]==19&&v.rhoOmega[0]==23&&v.gasSpecies.rho==rho&&rho[0]==7);
 assert(v.rhoK==rhoK&&v.rhoOmega==rhoOmega&&v.sstConfigured&&v.gasSpecies.chemistryStatus==status);
 }assert(live.empty());
}
'''
    _run_source(tmp_path,source)


def test_real_geometry_rebinding_preserves_resident_inventory(tmp_path):
    source=SOURCE[:SOURCE.index(' chmt::SharedGasDeviceStorage storage;')]+r'''
 {chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));auto& v=storage.hostView();
 auto* rho=v.rho;auto* partial=v.gasSpecies.rho;rho[0]=13;partial[0]=5;partial[1]=8;h.gas[0].mass=777;
 ++m.geometryVersion;m.volumes[0]=3;m.cellCentres[0].x=.2;
 assert(storage.configure(model,gas,{},h,error));assert(v.rho==rho&&rho[0]==13&&partial==v.gasSpecies.rho&&partial[0]==5&&v.V[0]==3&&v.Cx[0]==.2);
 assert(storage.deviceView()->V==v.V&&storage.deviceView()->Cx==v.Cx);
 chmt::HostStageGeometry stage;stage.oldVolume={3,9};stage.newVolume={3.1,10};stage.sweptVolume={0,.1,0};stage.interval=.1;
 assert(storage.bindStageGeometry(stage));assert(storage.deviceView()->gasGeometry.oldVolume[1]==9);
 stage.oldVolume.clear();assert(!storage.bindStageGeometry(stage));
 // Extra supplied geometric cells have no gas inventory source: geometry
 // ownership grows, while the real resident transport extent remains one.
 m.volumes={3,4};m.cellCentres.push_back({2,0,0});m.cellFaceOffsets={0,1,2};m.cellFaces={0,1};m.owner[1]=1;
 assert(storage.uploadGeometry(m));assert(v.nCells==1&&v.rho==rho&&rho[0]==13&&live.at(v.V)==2*sizeof(double));
 assert(storage.deviceView()->V==v.V&&storage.deviceView()->rho==rho);
 }assert(live.empty());
}
'''
    _run_source(tmp_path,source)


def test_real_view_rollback_failure_retains_all_exposed_owners_until_destroy(tmp_path):
    stub=COUNTERS.replace('if(d==viewTarget)++viewCopies;', 'if(d==viewTarget){++viewCopies;if(failViewCopies){--failViewCopies;if(n)std::memcpy(d,s,n/2);return 3;}}')
    stub=stub.replace('inline std::map<void*,size_t> live;', 'inline int failViewCopies=0;inline std::map<void*,size_t> live;')
    source=SOURCE[:SOURCE.index(' chmt::SharedGasDeviceStorage storage;')]+r'''
 {chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));auto old=storage.hostView();viewTarget=storage.deviceView();const auto count=live.size();
 failViewCopies=2;assert(!storage.configure(model,gas,{},h,error));assert(live.size()==count+3);assert(live.count((void*)old.gasSpecies.thermo.species));
 const auto attempts=viewCopies;assert(!storage.configure(model,gas,{},h,error));assert(viewCopies==attempts&&live.size()==count+3);
 }assert(live.empty());
}
'''
    _run_source(tmp_path,source,stub)


@pytest.mark.parametrize('failure',['zero','publish'])
def test_optional_first_enable_failure_does_not_destroy_retained_owners(tmp_path,failure):
    source=SOURCE[:SOURCE.index(' chmt::SharedGasDeviceStorage storage;')]+r'''
 auto mixture=gas;gas.mode=ugkwp::GasMode::SingleLegacy;model.physics.singleGasSpecies=0;
 {chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));auto& v=storage.hostView();
 const auto count=live.size();auto* rhoK=v.rhoK;v.rhoK[0]=97;v.rhoOmega[0]=101;v.rho[0]=13;h.sst={{22,44}};model.physics.enableSst=true;
 INJECT
 assert(!storage.configure(model,mixture,{},h,error));assert(v.rhoK==rhoK&&v.rhoK[0]==97&&v.rhoOmega[0]==101&&v.rho[0]==13&&!v.gasSpecies.rho&&live.size()==count);
 failZero=failCopy=-1;assert(storage.configure(model,mixture,{},h,error));assert(v.rhoK[0]==11&&v.rhoOmega[0]==22&&v.rho[0]==13&&v.gasSpecies.rho[0]==.25);
 }assert(live.empty());
}
'''

    source=source.replace('INJECT', 'failZero=0;' if failure=='zero' else 'failCopy=6;')
    _run_source(tmp_path,source)


def test_enable_wall_from_existing_resident_uses_actual_face_area(tmp_path):
    from test_reacting_wall_storage import storage_source
    source=storage_source().replace(' chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,canonical,{},h,error));auto&v=storage.hostView();', '''
 p.wallModel.family=WallFamily::LowRe;
 chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,canonical,{},h,error));auto&v=storage.hostView();
 auto* inventory=v.rho;v.rho[0]=17;
 p.wallModel.family=WallFamily::BoundaryLayer;assert(storage.configure(model,canonical,{},h,error));assert(v.rho==inventory&&v.rho[0]==17);
''')
    _run_source(tmp_path,source)


def test_committed_view_survives_reported_old_owner_free_error(tmp_path):
    stub=COUNTERS.replace('inline std::map<void*,size_t> live;', 'inline bool failFree=false;inline std::map<void*,size_t> live;')
    stub=stub.replace('++frees;std::free(p);}return 0;', '++frees;std::free(p);if(failFree){failFree=false;return 3;}}return 0;')
    source=SOURCE[:SOURCE.index(' chmt::SharedGasDeviceStorage storage;')]+r'''
 {chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));auto* rho=storage.hostView().rho;rho[0]=17;
 gas.coefficients[0]=1140;failFree=true;assert(!storage.configure(model,gas,{},h,error));assert(error.find("free CUDA buffer")!=std::string::npos);
 auto& v=storage.hostView();assert(v.rho==rho&&rho[0]==17&&v.gasSpecies.thermo.coefficients[0]==1140&&storage.deviceView()->gasSpecies.thermo.coefficients==v.gasSpecies.thermo.coefficients);
 assert(storage.configure(model,gas,{},h,error));assert(rho[0]==17);
 }assert(live.empty());
}
'''
    _run_source(tmp_path,source,stub)


def test_rebound_fault_owner_reports_download_and_sync_failures_without_partial_host_publish(tmp_path):
    source=SOURCE[:SOURCE.index(' chmt::SharedGasDeviceStorage storage;')]+r'''
 model.physics.enableSst=true;h.sst={{4,8}};
 {chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));assert(storage.configure(model,gas,{},h,error));
 for(int failure=0;failure<8;++failure){auto output=h;output.gas[0].mass=987;output.sst[0].rhoK=654;failCopy=failure;
  assert(!storage.downloadState(output,error)&&!error.empty()&&output.gas[0].mass==987&&output.sst[0].rhoK==654);
  failCopy=-1;assert(storage.downloadState(output,error)&&output.gas[0].mass==h.gas[0].mass&&output.sst[0].rhoK==4);
 }
 for(int failure=0;failure<2;++failure){failCopy=failure;assert(!storage.checkStatus(error)&&!error.empty());failCopy=-1;assert(storage.checkStatus(error));}
 failSync=0;assert(!storage.checkStatus(error)&&!error.empty());failSync=-1;assert(storage.checkStatus(error));
 const int beforeSync=syncs;assert(storage.checkStatus(error));assert(syncs==beforeSync+1);
 }assert(live.empty());std::puts("8 D2H failure positions preserve host candidate; 2 status D2H failures and 1 device sync failure retain report and retry; no redundant download sync");}
'''
    stub=COUNTERS.replace('inline int allocs=0,','inline int failSync=-1;inline int allocs=0,').replace('++syncs;return 0;', '++syncs;if(failSync==0){failSync=-1;return 3;}if(failSync>0)--failSync;return 0;')
    _run_source(tmp_path,source,stub)


def test_geometry_only_face_growth_does_not_masquerade_as_resident_flux_layout(tmp_path):
    source=SOURCE[:SOURCE.index(' chmt::SharedGasDeviceStorage storage;')]+r'''
 {chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));auto& v=storage.hostView();v.rho[0]=13;h.gas[0].mass=777;
 m.owner.push_back(0);m.neighbour.push_back(-1);m.periodicPartner.push_back(-1);m.areaVectors.push_back(m.areaVectors[1]);m.faceCentres.push_back(m.faceCentres[1]);m.boundaryKind.push_back(m.boundaryKind[1]);m.boundaryPrimitive.push_back(m.boundaryPrimitive[1]);
 assert(storage.uploadGeometry(m)&&v.nFaces==2&&v.rho[0]==13&&live.at(v.gasPhiRho)==2*sizeof(double));
 assert(storage.configure(model,gas,{},h,error)&&v.nFaces==3&&v.rho[0]==777./2&&live.at(v.gasPhiRho)==3*sizeof(double));
 assert(storage.deviceView()->gasPhiRho==v.gasPhiRho&&storage.deviceView()->rho==v.rho);
 }assert(live.empty());}
'''
    _run_source(tmp_path,source)
