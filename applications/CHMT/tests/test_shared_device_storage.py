"""Allocation/marshaling test with a host CUDA API shim; never native CUDA."""
from pathlib import Path
import subprocess
APP=Path(__file__).resolve().parents[1]
STUB=r'''
#pragma once
#include <cstdlib>
#include <cstring>
using cudaError_t=int; using cudaStream_t=void*;
constexpr int cudaSuccess=0,cudaErrorMemoryAllocation=2,cudaStreamNonBlocking=1;
constexpr int cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2,cudaMemcpyDeviceToDevice=3;
inline int cudaMalloc(void**p,size_t n){*p=std::malloc(n);return *p?0:2;}
inline int cudaFree(void*p){std::free(p);return 0;}
inline int cudaMemcpy(void*d,const void*s,size_t n,int){std::memcpy(d,s,n);return 0;}
inline int cudaMemcpyAsync(void*d,const void*s,size_t n,int kind,void*){return cudaMemcpy(d,s,n,kind);}
inline int cudaMemsetAsync(void*p,int value,size_t n,void*){std::memset(p,value,n);return 0;}
inline int cudaDeviceSynchronize(){return 0;}
inline int cudaStreamSynchronize(void*){return 0;}
inline int cudaStreamCreateWithFlags(void**p,int){*p=reinterpret_cast<void*>(1);return 0;}
inline int cudaStreamDestroy(void*){return 0;}
inline const char* cudaGetErrorString(int){return "shim error";}
inline int cudaPeekAtLastError(){return 0;} inline int cudaGetLastError(){return 0;}
'''
SOURCE=r'''
#include "gpu/SharedGasDeviceStorage.cuh"
#include "configuration/SharedGasModel.H"
#include <cassert>
int main(){
 auto gas=ugkwp::parseGasModelProperties(R"(gasMode mixtureFrozen;species (A B);speciesThermo{
 A{model linearCp;molarMass 0.028;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}
 B{model linearCp;molarMass 0.032;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}}
 diffusion{model constant;coefficients(0.01 0.02);})");
 chmt::ModelConfig model;std::string error;assert(chmt::bindSharedGasModel(gas,model,error));model.physics.tolerances.absoluteGeometry=1e-7;model.physics.tolerances.relativeGeometry=2e-6;
 chmt::HostState h;h.gas.resize(1);h.gas[0].mass=2;h.gas[0].species[0]=.5;h.gas[0].species[1]=1.5;h.gas[0].energy=1e6;
 auto&m=h.gasMesh;m.volumes={2};m.cellCentres={{0,0,0}};m.owner={0,0};m.neighbour={-1,-1};m.periodicPartner={-1,-1};m.cellFaceOffsets={0,2};m.cellFaces={0,1};
 m.faceCentres={{-1,0,0},{1,0,0}};m.areaVectors={{-1,0,0},{1,0,0}};m.boundaryKind={chmt::BoundaryKind::Slip,chmt::BoundaryKind::Slip};m.boundaryPrimitive.resize(2);
 chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));auto&v=storage.hostView();
 assert(v.nut && v.gasSpecies.rho && v.gasSpecies.flux && v.gradUxX && v.rhoNext && v.cellFaceId);
 assert(v.gasSpecies.rho[0]==.25&&v.gasSpecies.rho[1]==.75&&v.rho[0]==1);
 assert(v.gasSpecies.thermo.species!=gas.species.data());assert(v.gasSpecies.thermo.speciesOrderHash==gas.speciesOrderHash);
 assert(v.gasSpecies.thermo.coefficients!=gas.coefficients.data());
 assert(storage.deviceView()!=&v);v.gasLimiter=2;assert(storage.refreshView());assert(storage.deviceView()->gasLimiter==2);
 v.gasSpecies.rho[0]=.5;v.gasSpecies.rho[1]=.5;assert(storage.downloadState(h,error));assert(h.gas[0].species[0]==1&&h.gas[0].species[1]==1);
 chmt::HostStageGeometry stage;stage.interval=.1;stage.oldVolume={2};stage.newVolume={2.1};stage.sweptVolume={0,.1};
 assert(storage.bindStageGeometry(stage));assert(v.gasGeometry.absoluteGeometryTolerance==1e-7&&v.gasGeometry.relativeGeometryTolerance==2e-6);
 assert(v.gasGeometry.oldVolume!=stage.oldVolume.data());assert(storage.refreshView());assert(storage.deviceView()->gasGeometry.relativeGeometryTolerance==2e-6);
 assert(storage.clearGeometry());
 assert(storage.checkStatus(error));v.gasSpecies.faceStatus[1]=7;assert(!storage.checkStatus(error));
 assert(storage.clearTrialStatus());assert(storage.checkStatus(error));
}
'''
def test_shared_device_ownership_with_host_runtime_shim(tmp_path):
    (tmp_path/'cuda_runtime.h').write_text(STUB)
    (tmp_path/'probe.cpp').write_text(SOURCE)
    binary=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-pedantic','-I'+str(tmp_path),'-I'+str(APP),'-I'+str(APP.parents[1]/'common'),str(tmp_path/'probe.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_graded_periodic_face_uses_mapped_neighbor_distance(tmp_path):
    (tmp_path/'cuda_runtime.h').write_text(STUB)
    source=SOURCE[:SOURCE.index(' chmt::SharedGasDeviceStorage storage;')]+r'''
 m.volumes={.4,1.6};m.cellCentres={{.2,0,0},{1.4,0,0}};
 m.owner={0,1};m.neighbour={-1,-1};m.periodicPartner={1,0};m.cellFaceOffsets={0,1,2};m.cellFaces={0,1};
 m.faceCentres={{0,0,0},{2,0,0}};m.boundaryKind={chmt::BoundaryKind::Periodic,chmt::BoundaryKind::Periodic};
 h.gas.resize(2,h.gas[0]);
 chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));const auto&v=storage.hostView();
 assert(std::abs(v.faceWeight[0]-.75)<1e-12);assert(std::abs(v.faceWeight[1]-.25)<1e-12);
 assert(std::abs(v.deltaCoeffs[0]-1.25)<1e-12);
}
'''
    (tmp_path/'probe.cpp').write_text(source)
    binary=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-I'+str(tmp_path),'-I'+str(APP),'-I'+str(APP.parents[1]/'common'),str(tmp_path/'probe.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_sst_storage_preserves_integrals_and_configured_coefficients(tmp_path):
    (tmp_path/'cuda_runtime.h').write_text(STUB)
    source=SOURCE[:SOURCE.index(' chmt::SharedGasDeviceStorage storage;')]+r'''
 model.physics.enableSst=true;model.physics.sst.sigmaK1=.91;model.physics.sst.minimumK=1e-8;
 model.physics.sst.minimumOmega=2e-8;h.sst={{4,8}};m.boundarySst.resize(2);m.boundarySst[0].k=.5;m.boundarySst[0].omega=9;
 chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,gas,{},h,error));auto&v=storage.hostView();
 assert(v.rhoK[0]==2&&v.rhoOmega[0]==4);assert(v.sstCoefficients.alphaK1==.91);
 assert(v.sstKMin==1e-8&&v.sstOmegaMin==2e-8);assert(v.sstBoundaryK[0]==.5&&v.sstBoundaryOmega[0]==9);
 v.rhoK[0]=4;v.rhoOmega[0]=6;assert(storage.downloadState(h,error));assert(h.sst[0].rhoK==8&&h.sst[0].rhoOmega==12);
 v.rhoK[0]=-1;assert(!storage.downloadState(h,error));assert(h.sst[0].rhoK==8);
}
'''
    (tmp_path/'probe.cpp').write_text(source)
    binary=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-I'+str(tmp_path),'-I'+str(APP),'-I'+str(APP.parents[1]/'common'),str(tmp_path/'probe.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
