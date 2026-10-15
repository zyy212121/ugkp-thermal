import subprocess
import os
from pathlib import Path
import re
import pytest
from test_shared_device_storage import APP, SOURCE, STUB


def _probe():
    stub=STUB.replace('inline int cudaMemcpy(void*d,const void*s,size_t n,int){std::memcpy(d,s,n);return 0;}',r'''#include <atomic>
inline std::atomic<size_t> allocationCount{0};
inline size_t lastAllocationCount=0,temperatureAllocations=0,temperatureCopies=0;
inline const void* trackedTemperature=nullptr;
inline const void* failRead=nullptr;
inline const void* readOrder[16]{};
inline size_t reads=0;
inline int cudaMemcpy(void*d,const void*s,size_t n,int kind){
 if(kind==cudaMemcpyDeviceToHost){
  if(reads<16)readOrder[reads]=s;
  ++reads;
  if(s==trackedTemperature){temperatureAllocations=allocationCount.load()-lastAllocationCount;++temperatureCopies;}
  lastAllocationCount=allocationCount.load();
  if(s==failRead){if(n)std::memcpy(d,s,sizeof(double));return 2;}
 }
 std::memcpy(d,s,n);return 0;
}''')
    source=SOURCE[:SOURCE.index(' chmt::SharedGasDeviceStorage storage;')]
    source=source.replace('int main(){',r'''void* operator new(std::size_t n){++allocationCount;if(void*p=std::malloc(n))return p;throw std::bad_alloc();}
void operator delete(void*p)noexcept{std::free(p);}
void operator delete(void*p,std::size_t)noexcept{std::free(p);}
int main(int argc,char**argv){assert(argc==2);const bool allocations=std::string(argv[1])=="allocation";''')
    source+=r'''
 model.physics.enableSst=true;h.sst={{4,8}};
 chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));
 auto&v=storage.hostView();trackedTemperature=v.Tgas;
 auto download=[&](){reads=0;return storage.downloadState(h,error);};
 auto verifyOrder=[&](){
  const void* expected[]={v.rho,v.rhoUx,v.rhoUy,v.rhoUz,v.rhoE,v.Tgas,v.gasSpecies.rho,v.rhoK,v.rhoOmega};
  assert(reads==9);for(size_t j=0;j<9;++j)assert(readOrder[j]==expected[j]);
 };
 for(int j=0;j<3;++j){v.Tgas[0]=601.25+j;assert(download());verifyOrder();}
 if(allocations){
  for(int j=0;j<12;++j){v.Tgas[0]=701.25+j;assert(download());verifyOrder();assert(h.gasTemperature[0]==701.25+j);
   std::printf("cells=1 sample=%d temperature_allocations=%zu copies=%zu\n",j,temperatureAllocations,temperatureCopies);
   assert(temperatureAllocations==0);
  }
 }else{
  for(const void* failed:{static_cast<const void*>(v.Tgas),static_cast<const void*>(v.rhoK),static_cast<const void*>(v.rhoOmega)}){
   auto before=h;v.Tgas[0]=811.25;v.rho[0]=3;v.rhoE[0]=9e5;failRead=failed;
   assert(!download());assert(!error.empty());assert(h.gas==before.gas);
   assert(std::memcmp(h.gasTemperature.data(),before.gasTemperature.data(),sizeof(double))==0);
   assert(h.sst[0].rhoK==before.sst[0].rhoK&&h.sst[0].rhoOmega==before.sst[0].rhoOmega);
   failRead=nullptr;storage.beginOperation();assert(download());verifyOrder();
   assert(h.gas[0].mass==6&&h.gas[0].energy==18e5&&h.gasTemperature[0]==811.25);
  }
 }
 m.volumes={.4,1.6};m.cellCentres={{.2,0,0},{1.4,0,0}};
 m.owner={0,1};m.neighbour={-1,-1};m.periodicPartner={1,0};m.cellFaceOffsets={0,1,2};m.cellFaces={0,1};
 m.faceCentres={{0,0,0},{2,0,0}};m.boundaryKind={chmt::BoundaryKind::Periodic,chmt::BoundaryKind::Periodic};
 h.gas.resize(2,h.gas[0]);h.sst.resize(2,h.sst[0]);h.gasTemperature.resize(2,h.gasTemperature[0]);
 assert(storage.configure(model,gas,{},h,error));trackedTemperature=v.Tgas;
 for(int j=0;j<3;++j){v.Tgas[0]=901.25+j;v.Tgas[1]=1001.25+j;assert(download());verifyOrder();}
 for(int j=0;j<6;++j){v.Tgas[0]=1101.25+j;v.Tgas[1]=1201.25+j;assert(download());verifyOrder();
  assert(h.gasTemperature[0]==1101.25+j&&h.gasTemperature[1]==1201.25+j);
  if(allocations){std::printf("cells=2 sample=%d temperature_allocations=%zu copies=%zu\n",j,temperatureAllocations,temperatureCopies);assert(temperatureAllocations==0);}
 }
 std::printf("mode=%s total_temperature_copies=%zu\n",argv[1],temperatureCopies);
}
'''
    source=source.replace('assert(h.gas==before.gas);','assert(std::memcmp(h.gas.data(),before.gas.data(),h.gas.size()*sizeof(chmt::GasQ))==0);')
    return stub,source


@pytest.mark.parametrize('mode',['transaction','allocation'])
def test_temperature_staging_preserves_snapshot_and_reuses_capacity(tmp_path,mode):
    stub,source=_probe()
    (tmp_path/'cuda_runtime.h').write_text(stub)
    (tmp_path/'probe.cpp').write_text(source)
    binary=tmp_path/'probe'
    command=['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic','-I'+str(tmp_path),'-I'+str(APP),'-I'+str(APP.parents[1]/'common'),str(tmp_path/'probe.cpp'),'-o',str(binary)]
    build=subprocess.run(command,capture_output=True,text=True)
    (tmp_path/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary),mode],capture_output=True,text=True)
    (tmp_path/'run.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr


@pytest.mark.parametrize('mode',['transaction','allocation'])
def test_temperature_staging_with_actual_cuda(tmp_path,mode):
    if os.environ.get('CHMT_STORAGE_GPU_TESTS')!='1':
        pytest.skip('Set CHMT_STORAGE_GPU_TESTS=1 on a real GPU')
    cuda=Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))
    if not (cuda/'include/cuda_runtime.h').is_file():
        pytest.skip('CUDA headers unavailable')
    _,source=_probe()
    source=source.replace('for(const void* failed:{static_cast<const void*>(v.Tgas),static_cast<const void*>(v.rhoK),static_cast<const void*>(v.rhoOmega)}){','for(double** failed:{&v.Tgas,&v.rhoK,&v.rhoOmega}){')
    source=source.replace('failRead=failed;','double* original=*failed;*failed=nullptr;')
    source=source.replace('failRead=nullptr;storage.beginOperation();','*failed=original;cudaGetLastError();storage.beginOperation();')
    source=re.sub(r'v\.(Tgas|rho|rhoE)\[(\d)\]=([^;]+);',r'set(v.\1,\2,\3);',source)
    (tmp_path/'probe.cpp').write_text(NATIVE_RUNTIME+source)
    binary=tmp_path/'probe'
    command=['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic','-I'+str(cuda/'include'),'-I'+str(APP),'-I'+str(APP.parents[1]/'common'),str(tmp_path/'probe.cpp'),'-L'+str(cuda/'lib64'),'-Wl,-rpath,'+str(cuda/'lib64'),'-lcudart','-o',str(binary)]
    build=subprocess.run(command,capture_output=True,text=True)
    (tmp_path/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary),mode],capture_output=True,text=True)
    (tmp_path/'run.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
    if mode=='transaction':
        assert run.stderr.count('actual cudaMemcpy error=1 invalid argument')==3


NATIVE_RUNTIME=r'''

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstddef>
#include <atomic>
inline std::atomic<size_t> allocationCount{0};
inline size_t lastAllocationCount=0,temperatureAllocations=0,temperatureCopies=0;
inline const void* trackedTemperature=nullptr;
inline const void* readOrder[16]{};
inline size_t reads=0;
cudaError_t observeMemcpy(void*d,const void*s,size_t n,cudaMemcpyKind kind){
 if(kind==cudaMemcpyDeviceToHost){
  if(reads<16)readOrder[reads]=s;
  ++reads;
  if(s==trackedTemperature){temperatureAllocations=allocationCount.load()-lastAllocationCount;++temperatureCopies;}
  lastAllocationCount=allocationCount.load();
 }
 const auto code=cudaMemcpy(d,s,n,kind);
 if(code!=cudaSuccess)std::fprintf(stderr,"actual cudaMemcpy error=%d %s bytes=%zu kind=%d\n",int(code),cudaGetErrorString(code),n,int(kind));
 return code;
}
void set(double*p,size_t index,double value){assert(cudaMemcpy(p+index,&value,sizeof(value),cudaMemcpyHostToDevice)==cudaSuccess);}
#define cudaMemcpy observeMemcpy

'''
