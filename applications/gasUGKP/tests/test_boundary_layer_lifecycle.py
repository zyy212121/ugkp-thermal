"""Execute real native configuration bodies with host-memory CUDA plumbing."""
from pathlib import Path
import re
import subprocess
ROOT=Path(__file__).resolve().parents[3]
PREAMBLE=r'''
#include "common/gasTransport/GasCapabilities.H"

#include "common/gasTransport/GasStateView.H"
#include "common/gasWall/WallModelTypes.H"
#include <cassert>
#include <cstring>
#include <cstdio>
using cudaError_t=int;constexpr int cudaSuccess=0;
int cudaGetLastError(){return 0;}int cudaDeviceSynchronize(){return 0;}
struct DeviceState {
 ugkwp::GasSpeciesState<double,2> gasSpecies;ugkwp::GasBoundaryLayerState<double> gasBoundaryLayer;
 ugkwp::SstCoefficients sstCoefficients;DeviceState*deviceState=this;
 int hostTurbulenceModel=3,nCells=2,nFaces=2,fixedCellBlockThreads=128,hostGasFluxScheme=1,hostGasReconstruction=0,hostGasLimiter=0,hostGasTimeIntegrator=1,particleCapacity=0;
 int sstWallTreatment=2,sstConfigured=1;
 double sstKMin=1e-12,sstOmegaMin=1e-12,sstMaxSourceNumber=.25,sstWallKappa=.41,sstWallE=9.8,sstWallCmu=.09;
 double k[2]{},omega[2]{},sstWallDistance[2]{},sstBoundaryK[2]{},sstBoundaryOmega[2]{};int sstBoundaryKMode[2]{},sstBoundaryOmegaMode[2]{};
};
DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}int validateState(DeviceState*,const char*){return 0;}
void setLastErrorText(const char*s){std::puts(s);}void setLastError(const char*,int){}
int copies=0;
template<class T>int copyToDevice(T*d,const T*s,std::size_t n,const char*){++copies;std::memcpy(d,s,n*sizeof(T));return 0;}
int syncSstConfiguration(DeviceState*,const char*){return 0;}void initialiseSstConservativeStateKernel(DeviceState*){}
'''

def test_sst_configuration_precedes_boundary_layer_without_changing_legacy_reconfiguration(tmp_path):
    backend=(ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    start=backend.index('extern "C" int ugkwpGpuResidentStrictConfigureSst')
    stop=backend.index('extern "C" int ugkwpGpuResidentStrictComputeGasCourant',start)
    body=re.sub(r'<<<[\s\S]*?>>>','',backend[start:stop])
    adapter=(ROOT/'applications/gasUGKP/private_backend/SharedGasAdapter.cuh').read_text()
    helper=adapter[adapter.index('inline ugkwp::GasCapabilityRequest sharedGasCapabilityRequest'):adapter.index('inline bool sharedGasIdentityMatches')]
    main=r'''int main(){DeviceState s;s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;
 double k[]={1,1},omega[]={2,2},distance[]={.1,.1},boundary[]={0,0};int mode[]={0,0};
 auto configure=[&](int family){return ugkwpGpuResidentStrictConfigureSst(&s,1,1,1,1,.075,.083,.09,.55,.44,.31,1,10,1e-12,1e-12,.25,family,.41,9.8,.09,k,omega,distance,mode,mode,boundary,boundary);};
 assert(configure(2)==0&&s.sstWallTreatment==2&&s.k[0]==1); // legal startup SST setup
 s.gasBoundaryLayer.enabled=true;s.k[0]=7;const int before=copies;
 for(int family:{0,1,2}){assert(configure(family)!=0);assert(s.sstWallTreatment==2&&s.k[0]==7&&copies==before);}
 // A separate legacy resident retains its historical repeat-configuration path.
 DeviceState legacy;s=legacy;s.deviceState=&s;s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;
 assert(configure(1)==0&&s.sstWallTreatment==1);k[0]=3;
 assert(configure(0)==0&&s.sstWallTreatment==0&&s.k[0]==3);
}'''
    source=tmp_path/'lifecycle.cpp';source.write_text(PREAMBLE+helper+body+main)
    binary=tmp_path/'lifecycle'
    build=subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
