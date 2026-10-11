"""Native adapter ownership tests use a host-memory CUDA allocation shim.

This executes the real adapter's loading/identity/allocation paths. It does not
claim a GPU kernel, CUDA build or native transport result.
"""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]

ADAPTER_PROBE = r'''
#include "applications/gasUGKP/gpu/GpuBackendApi.H"
#include "common/gasTransport/GasBuildConfig.H"
#include "common/gasTransport/GasStateView.H"
#include "common/gasTransport/GasCapabilities.H"
#include "common/gasTransport/GasMechanismIO.H"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
static int allocations=0,failAfter=-1,copyFailAfter=-1,syncFailAfter=-1,hostCopyFailAfter=-1,syncFailures=0;
using cudaError_t=int;
const int cudaSuccess=0,cudaMemcpyHostToDevice=1;
int cudaFree(void*p){if(p){--allocations;std::free(p);}return 0;}
int cudaMemcpy(void*d,const void*s,std::size_t n,int){if(syncFailures){if(syncFailures--==2 && n)std::memcpy(d,s,n-1);return 1;}if(syncFailAfter==0){syncFailAfter=-1;if(n)std::memcpy(d,s,n/2);return 1;}if(syncFailAfter>0)--syncFailAfter;std::memcpy(d,s,n);return 0;}
int cudaMemset(void*d,int v,std::size_t n){std::memset(d,v,n);return 0;}
std::string last;
void setLastErrorText(const char*s){last=s;}
void setLastError(const char*s,int){last=s;}
template<class T> int allocate(T*&p,std::size_t n,const char*){p=nullptr;if(!n)return 0;if(failAfter==0)return 1;if(failAfter>0)--failAfter;p=static_cast<T*>(std::malloc(n*sizeof(T)));if(!p)return 1;++allocations;return 0;}
template<class T> int copyToDevice(T*d,const T*s,std::size_t n,const char*){if(copyFailAfter==0){copyFailAfter=-1;if(n)std::memcpy(d,s,sizeof(T));return 1;}if(copyFailAfter>0)--copyFailAfter;if(n)std::memcpy(d,s,n*sizeof(T));return 0;}
template<class T> int copyToHost(T*d,const T*s,std::size_t n,const char*){if(hostCopyFailAfter==0){hostCopyFailAfter=-1;if(n)std::memcpy(d,s,sizeof(T));return 1;}if(hostCopyFailAfter>0)--hostCopyFailAfter;if(n)std::memcpy(d,s,n*sizeof(T));return 0;}
struct DeviceState {
 ugkwp::GasSpeciesState<double,ugkwp::compiledGasSpecies> gasSpecies,gasRejectedView;
 bool gasModelPoisoned=false;
 DeviceState* deviceState=nullptr;
 int nCells=2,nFaces=2,particleCapacity=0;
 int gasFluxScheme=1,gasReconstruction=0,gasLimiter=0,gasTimeIntegrator=1,turbulenceModel=0,sstWallTreatment=0;
 int hostGasFluxScheme=1,hostGasReconstruction=0,hostGasLimiter=0,hostGasTimeIntegrator=1,hostTurbulenceModel=0;
 bool gasInitialFieldsUploaded=false,gasSpeciesUploaded=false;
 double* rho=nullptr;
};
DeviceState* asState(void*p){return static_cast<DeviceState*>(p);}
int validateState(DeviceState*s,const char*){return s && !s->gasModelPoisoned?0:1;}
#include "applications/gasUGKP/private_backend/SharedGasStorage.cuh"
#include "applications/gasUGKP/private_backend/SharedGasAdapter.cuh"
int main(){
 DeviceState state,device;state.deviceState=&device;
 const std::string text=R"(gasMode mixtureFrozen;species(A B);speciesThermo{
 A{model linearCp;molarMass .028;minTemperature 100;maxTemperature 4000;coefficients(1040 0 0);}
 B{model linearCp;molarMass .028;minTemperature 100;maxTemperature 4000;coefficients(1040 0 0);}}
 diffusion{model constant;coefficients(.02 .02);turbulentSchmidt .7;})";
 auto model=ugkwp::parseGasModelProperties(text);
 ugkwpGpuIpc::GasModelConfigureArgsV1 args{1,1,2,0,model.speciesOrderHash,model.thermoHash,0,text.size(),0};
 state.turbulenceModel=state.hostTurbulenceModel=3;state.sstWallTreatment=1;
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)!=0);
 assert(allocations==0 && state.gasSpecies.mode==ugkwp::GasMode::SingleLegacy);
 state.turbulenceModel=state.hostTurbulenceModel=0;state.sstWallTreatment=0;
 auto bad=args;++bad.thermoHash;
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&bad,text.data(),nullptr)!=0);
 assert(allocations==0 && state.gasSpecies.mode==ugkwp::GasMode::SingleLegacy);
 failAfter=3;
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)!=0);
 assert(allocations==0 && state.gasSpecies.mode==ugkwp::GasMode::SingleLegacy);
 failAfter=-1;
 syncFailAfter=0;
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)!=0);
 assert(allocations==0 && state.gasSpecies.mode==ugkwp::GasMode::SingleLegacy);
 assert(device.gasSpecies.mode==ugkwp::GasMode::SingleLegacy && !device.gasSpecies.rho);
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)==0);
 assert(allocations>0 && device.gasSpecies.mode==ugkwp::GasMode::MixtureFrozen);
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)!=0);
 double density[]={1,1};state.rho=density;state.gasInitialFieldsUploaded=true;
 ugkwpGpuIpc::GasSpeciesIdentityV1 identity{1,2,model.speciesOrderHash,model.thermoHash,0};
 double values[]={-.1,.4,1.1,.6};
 assert(ugkwpGpuResidentStrictUploadSpeciesV1(&state,&identity,values)!=0);
 assert(!state.gasSpeciesUploaded);
 values[0]=.4;values[2]=.6;
 const int configuredAllocations=allocations;
 copyFailAfter=0;
 assert(ugkwpGpuResidentStrictUploadSpeciesV1(&state,&identity,values)!=0);
 assert(!state.gasSpeciesUploaded && allocations==configuredAllocations);
 for(int i=0;i<4;++i)assert(state.gasSpecies.rho[i]==0);
 double* acceptedDensity=state.gasSpecies.rho;
 syncFailAfter=0;
 assert(ugkwpGpuResidentStrictUploadSpeciesV1(&state,&identity,values)!=0);
 assert(!state.gasSpeciesUploaded && allocations==configuredAllocations);
 assert(state.gasSpecies.rho==acceptedDensity && device.gasSpecies.rho==acceptedDensity);
 for(int i=0;i<4;++i)assert(acceptedDensity[i]==0);
 assert(ugkwpGpuResidentStrictUploadSpeciesV1(&state,&identity,values)==0);
 double output[4]={};
 hostCopyFailAfter=0;
 assert(ugkwpGpuResidentStrictDownloadSpeciesV1(&state,&identity,output)!=0);
 for(int i=0;i<4;++i)assert(output[i]==0);
 assert(ugkwpGpuResidentStrictDownloadSpeciesV1(&state,&identity,output)==0);
 for(int i=0;i<4;++i)assert(output[i]==values[i]);
 auto wrong=identity;++wrong.speciesOrderHash;
 assert(ugkwpGpuResidentStrictDownloadSpeciesV1(&state,&wrong,output)!=0);
 assert(ugkwpGpuResidentStrictUploadSpeciesV1(&state,&identity,values)!=0);
 int fixed[]={1,1};double boundary[]={.3,.4,.7,.6};
 copyFailAfter=1;
 assert(ugkwpGpuResidentStrictUploadSpeciesBoundaryV1(&state,&identity,fixed,boundary)!=0);
 assert(allocations==configuredAllocations);
 for(int i=0;i<2;++i)assert(state.gasSpecies.compositionBoundaryFixed[i]==0);
 for(int i=0;i<4;++i)assert(state.gasSpecies.boundaryMassFraction[i]==0);
 int* acceptedFixed=state.gasSpecies.compositionBoundaryFixed;
 double* acceptedBoundary=state.gasSpecies.boundaryMassFraction;
 syncFailAfter=0;
 assert(ugkwpGpuResidentStrictUploadSpeciesBoundaryV1(&state,&identity,fixed,boundary)!=0);
 assert(allocations==configuredAllocations);
 assert(state.gasSpecies.compositionBoundaryFixed==acceptedFixed && device.gasSpecies.compositionBoundaryFixed==acceptedFixed);
 assert(state.gasSpecies.boundaryMassFraction==acceptedBoundary && device.gasSpecies.boundaryMassFraction==acceptedBoundary);
 for(int i=0;i<2;++i)assert(acceptedFixed[i]==0);
 for(int i=0;i<4;++i)assert(acceptedBoundary[i]==0);
 assert(ugkwpGpuResidentStrictUploadSpeciesBoundaryV1(&state,&identity,fixed,boundary)==0);
 assert(device.gasSpecies.compositionBoundaryFixed==state.gasSpecies.compositionBoundaryFixed);
 assert(device.gasSpecies.boundaryMassFraction==state.gasSpecies.boundaryMassFraction);
 for(int i=0;i<4;++i)assert(state.gasSpecies.boundaryMassFraction[i]==boundary[i]);
 assert(device.gasSpecies.rho==state.gasSpecies.rho);
 // Field upload deliberately scrubs host copies of device calculation scalars.
 state.gasFluxScheme=state.gasTimeIntegrator=state.turbulenceModel=0;
 state.hostGasFluxScheme=2;state.hostGasTimeIntegrator=3;state.hostTurbulenceModel=3;
 state.hostGasReconstruction=1;state.hostGasLimiter=2;
 const auto retained=sharedGasCapabilityRequest(&state,state.gasSpecies.mode,0);
 assert(ugkwp::validateGasCapabilities(retained));
 assert(retained.fluxScheme==2 && retained.timeIntegrator==3 && retained.turbulenceModel==3);
 assert(retained.reconstruction==1 && retained.limiter==2);
 releaseSharedGasSpecies(state.gasSpecies);
 assert(allocations==0);
}
'''

def test_model_configuration_is_transactional_and_identity_immutable(tmp_path):
    source=tmp_path/"adapter.cpp"
    source.write_text(ADAPTER_PROBE)
    binary=tmp_path/"adapter"
    build=subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror","-I",str(ROOT),str(source),"-o",str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr

def test_actual_resident_owns_optional_storage_and_calls_shared_adapter():
    source=(ROOT/"applications/gasUGKP/private_backend/GpuResidentStrict.cu").read_text()
    assert '#include "SharedGasAdapter.cuh"' in source
    assert '#include "SharedGasStorage.cuh"' in source
    assert 'ugkwp::GasSpeciesState<double,ugkwp::compiledGasSpecies> gasSpecies;' in source
    assert 'releaseSharedGasSpecies(s->gasSpecies);' in source
    assert 'releaseSharedGasSpecies(s->gasRejectedView);' in source
    validation=source[source.index('int validateState('):source.index('void scrubHostCalculationScalars(')]
    assert 'if (s->gasModelPoisoned)' in validation
    assert 's->gasInitialFieldsUploaded = true;' in source


def test_reacting_adapter_allocates_exact_ns10_tables_and_failure_is_transactional(tmp_path):
    source=tmp_path/"reacting_adapter.cpp"
    harness=ADAPTER_PROBE.split("int main(){",1)[0]
    source.write_text(harness+r'''
int main(int argc,char**argv){
 assert(argc==2);
 DeviceState state,device;state.deviceState=&device;
 const std::string directory=argv[1];
 const auto model=ugkwp::readGasModelProperties(directory+"/h2o2.gasModelProperties");
 std::ifstream modelFile(directory+"/h2o2.gasModelProperties");
 std::string text((std::istreambuf_iterator<char>(modelFile)),{});
 std::ifstream mechanismFile(directory+"/h2o2.mechanism");
 std::string chemistry((std::istreambuf_iterator<char>(mechanismFile)),{});
 const auto mechanism=ugkwp::parseGasMechanismProperties<10>(chemistry,model.thermoView<10>(),model.phase);
 ugkwpGpuIpc::GasModelConfigureArgsV1 args{1,2,10,0,model.speciesOrderHash,model.thermoHash,mechanism.mechanismHash,text.size(),chemistry.size()};
 ugkwpGpuIpc::GasModelCapabilitiesV1 capabilities{};
 assert(ugkwpGpuResidentStrictQueryGasModelCapabilitiesV1(&state,&capabilities)==0);
 assert(capabilities.compiledSpecies==10 && (capabilities.modeMask&4));
 failAfter=20;
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),chemistry.data())!=0);
 assert(allocations==0 && state.gasSpecies.mode==ugkwp::GasMode::SingleLegacy);
 failAfter=-1;
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),chemistry.data())==0);
 assert(state.gasSpecies.mode==ugkwp::GasMode::MixtureChemistry);
 assert(state.gasSpecies.chemistryAudit && state.gasSpecies.chemistryStatus);
 assert(state.gasSpecies.mechanism.reactionCount==29);
 assert(state.gasSpecies.mechanism.mechanismHash==mechanism.mechanismHash);
 assert(device.gasSpecies.thermo.species==state.gasSpecies.thermo.species);
 assert(device.gasSpecies.mechanism.reactions==state.gasSpecies.mechanism.reactions);
 releaseSharedGasSpecies(state.gasSpecies);
 assert(allocations==0);
}
''')
    binary=tmp_path/"reacting_adapter"
    build=subprocess.run(["g++","-std=c++17","-DUGKWP_GAS_SPECIES=10","-Wall","-Wextra","-Werror","-I",str(ROOT),str(source),"-o",str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary),str(ROOT/"common/chemistry/mechanisms")],capture_output=True,text=True)
    assert run.returncode==0,run.stderr


def test_late_sst_configuration_checks_shared_capability_before_mutation():
    source=(ROOT/"applications/gasUGKP/private_backend/GpuResidentStrict.cu").read_text()
    body=source[source.index('extern "C" int ugkwpGpuResidentStrictConfigureSst'):]
    assert body.index("sharedGasCapabilityRequest(s,s->gasSpecies.mode,wallTreatment)") < body.index("s->sstCoefficients =")


def test_double_view_copy_failure_quarantines_live_storage_and_poisoned_resident(tmp_path):
    source=tmp_path/"poison.cpp"
    harness=ADAPTER_PROBE.split("int main(){",1)[0]
    setup=ADAPTER_PROBE[ADAPTER_PROBE.index(" const std::string text="):ADAPTER_PROBE.index(" state.turbulenceModel=state.hostTurbulenceModel=3;")]
    source.write_text(harness+"int main(){for(int operation=0;operation<3;++operation){DeviceState state,device;state.deviceState=&device;"+setup+r'''
 if(operation==0)syncFailures=2;
 const int configured=ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr);
 assert((configured!=0)==(operation==0));
 double density[]={1,1};state.rho=density;state.gasInitialFieldsUploaded=true;
 ugkwpGpuIpc::GasSpeciesIdentityV1 identity{1,2,model.speciesOrderHash,model.thermoHash,0};
 double values[]={.4,.4,.6,.6};int fixed[]={1,1};double boundary[]={.3,.4,.7,.6};
 if(operation>0){
  double* accepted=state.gasSpecies.rho;
  if(operation==1)syncFailures=2;
  const int uploaded=ugkwpGpuResidentStrictUploadSpeciesV1(&state,&identity,values);
  assert((uploaded!=0)==(operation==1));
  if(operation==1){assert(state.gasSpecies.rho==accepted);for(int i=0;i<4;++i)assert(accepted[i]==0);}
 }
 if(operation==2){
  int* accepted=state.gasSpecies.compositionBoundaryFixed;
  syncFailures=2;
  assert(ugkwpGpuResidentStrictUploadSpeciesBoundaryV1(&state,&identity,fixed,boundary)!=0);
  assert(state.gasSpecies.compositionBoundaryFixed==accepted);
  for(int i=0;i<2;++i)assert(accepted[i]==0);
 }
 assert(state.gasModelPoisoned);
 if(operation<2){assert(state.gasRejectedView.rho);assert(device.gasSpecies.rho==state.gasRejectedView.rho);}
 else {assert(state.gasRejectedView.compositionBoundaryFixed);assert(device.gasSpecies.compositionBoundaryFixed==state.gasRejectedView.compositionBoundaryFixed);}
 ugkwpGpuIpc::GasModelCapabilitiesV1 caps{};
 assert(ugkwpGpuResidentStrictQueryGasModelCapabilitiesV1(&state,&caps)!=0);
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&state,&args,text.data(),nullptr)!=0);
 assert(ugkwpGpuResidentStrictUploadSpeciesV1(&state,&identity,values)!=0);
 assert(ugkwpGpuResidentStrictDownloadSpeciesV1(&state,&identity,values)!=0);
 assert(ugkwpGpuResidentStrictUploadSpeciesBoundaryV1(&state,&identity,fixed,boundary)!=0);
 assert(allocations>0);
 releaseSharedGasSpecies(state.gasSpecies);releaseSharedGasSpecies(state.gasRejectedView);
 assert(allocations==0);
}}
''')
    binary=tmp_path/"poison"
    build=subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror","-I",str(ROOT),str(source),"-o",str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
