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


def _native_function(source, signature):
    start = source.index(signature)
    brace = source.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def _publication_probe():
    from test_shared_gas_adapter import ADAPTER_PROBE
    pre = ADAPTER_PROBE.split('int main(){', 1)[0]
    pre = pre.replace('#include <cassert>', '''#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include "applications/gasUGKP/private_backend/SharedGasTrialFields.H"
''')
    pre = pre.replace(' double* rho=nullptr;', ''' double* rho=nullptr;
 ugkwp::GasBoundaryLayerState<double> gasBoundaryLayer;
 ugkwp::GasBoundaryLayerModelState<double,ugkwp::compiledGasSpecies> gasBoundaryLayerModel;
 ugkwp::GasSstAuditState<double> gasSstAudit;SharedGasTrialStorage gasTrial;
 double *gasWallQuadratureDistance=nullptr,*gasWallQuadratureWeight=nullptr;
 double gasBoundaryLayerStageTime=0,gasThermalConductivity=-1;
 int *faceOwner=nullptr,*faceNeighbour=nullptr,*gasBoundaryKind=nullptr,*riemannBoundaryKind=nullptr;
 int *riemannBoundaryTFix=nullptr,*riemannBoundaryUFix=nullptr;
 double *V=nullptr,*magSf=nullptr,*p=nullptr,*Tgas=nullptr,*Ux=nullptr,*Uy=nullptr,*Uz=nullptr,*k=nullptr,*omega=nullptr;
 double *riemannBoundaryT=nullptr,*riemannBoundaryUx=nullptr,*riemannBoundaryUy=nullptr,*riemannBoundaryUz=nullptr;
''')
    backend = (ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    scrub = _native_function(backend, 'void scrubHostCalculationScalars(')
    sync = _native_function(backend, 'int syncDeviceState(')
    declaration = pre[pre.index('struct DeviceState {'):pre.index('DeviceState* asState')]
    # Import every scrubbed field, rather than testing only mu/Pr and overlooking
    # future additions to the real resident's deliberately device-owned scalars.
    missing = [name for name in re.findall(r's->(\w+)\s*=', scrub)
               if not re.search(r'\b'+name+r'\b', declaration)]
    fields = ''.join(('ugkwp::SstCoefficients ' if name=='sstCoefficients' else 'double ')
                     + name + '{};\n' for name in missing)
    pre = pre.replace(' double* rho=nullptr;', ' double* rho=nullptr;\n'+fields)
    old_copy = _native_function(pre, 'int cudaMemcpy(')
    pre = pre.replace(old_copy, r'''
void* residentTarget=nullptr;std::size_t residentBytes=0;
int publicationAttempts=0,failPublication=-1;
int cudaMemcpy(void* dst,const void* src,std::size_t n,int){
 const auto at=reinterpret_cast<std::uintptr_t>(dst),base=reinterpret_cast<std::uintptr_t>(residentTarget);
 if(residentTarget&&at>=base&&at+n<=base+residentBytes){
  if(publicationAttempts++==failPublication){if(n)std::memcpy(dst,src,n/2);return 1;}
 }
 std::memcpy(dst,src,n);return cudaSuccess;
}
''')
    return pre+sync+'\n'+scrub+r'''
const int cudaDevAttrMultiProcessorCount=1;
int cudaGetDevice(int*p){*p=0;return 0;}
int cudaDeviceGetAttribute(int*p,int,int){*p=80;return 0;}
int cudaMemGetInfo(std::size_t*f,std::size_t*t){*f=*t=std::size_t(8)<<30;return 0;}
#include "applications/gasUGKP/private_backend/BoundaryLayerStorage.cuh"
#include "applications/gasUGKP/private_backend/BoundaryLayerAdapter.cuh"
void checkUnrelated(const DeviceState& before,const DeviceState& after){
 bool changed[sizeof(DeviceState)]{};
#define ALLOWED(field) for(std::size_t i=offsetof(DeviceState,field);i<offsetof(DeviceState,field)+sizeof(before.field);++i)changed[i]=true
 ALLOWED(gasBoundaryLayer);ALLOWED(gasBoundaryLayerModel);ALLOWED(gasSstAudit);
 ALLOWED(gasWallQuadratureDistance);ALLOWED(gasWallQuadratureWeight);ALLOWED(sstWallTreatment);
#undef ALLOWED
 const auto*a=reinterpret_cast<const unsigned char*>(&before);const auto*b=reinterpret_cast<const unsigned char*>(&after);
 for(std::size_t i=0;i<sizeof(DeviceState);++i)assert(changed[i]||a[i]==b[i]);
}
int main(int argc,char**argv){assert(argc==4);const int model=std::atoi(argv[1]);failPublication=std::atoi(argv[2]);const int audit=std::atoi(argv[3]);
 DeviceState state,device;state.deviceState=&device;
 int owner[]={0,1},neighbour[]={-1,-1},kind[]={2,0},fixed[]={1,0},zeroFlags[]={0,0},cellStatus[]={0,0},faceStatus[]={0,0};
 double volume[]={4e-7,4e-7},area[]={1,1},diffusion[]={1e-4,1e-4},rho[]={.6824451,.6824451};
 double p[]={101325,101325},temperature[]={500,500},u[]={0,30},zero[]={0,0},k[]={.5,.5},omega[]={200,200},partial[]={.7*rho[0],.7*rho[1],.3*rho[0],.3*rho[1]};
 state.faceOwner=owner;state.faceNeighbour=neighbour;state.gasBoundaryKind=state.riemannBoundaryKind=kind;state.V=volume;state.magSf=area;
 state.rho=rho;state.p=p;state.Tgas=temperature;state.Ux=u;state.Uy=state.Uz=zero;state.k=k;state.omega=omega;
 state.riemannBoundaryTFix=fixed;state.riemannBoundaryUFix=zeroFlags;state.riemannBoundaryT=temperature;
 state.riemannBoundaryUx=state.riemannBoundaryUy=state.riemannBoundaryUz=zero;
 state.gasSpecies.mode=model?ugkwp::GasMode::MixtureChemistry:ugkwp::GasMode::MixtureFrozen;
 state.gasSpecies.diffusivity=diffusion;state.gasSpecies.rho=partial;state.gasSpecies.cellStatus=cellStatus;state.gasSpecies.faceStatus=faceStatus;state.gasSpeciesUploaded=true;
 ugkwp::SpeciesThermoData<double> metadata[2];double coefficients[]={1000,0,0,1000,0,0},elements[]={1,1},basis[]={-1,1};
 for(int i=0;i<2;++i){metadata[i].coefficientOffset=3*i;metadata[i].molarMass=.028;metadata[i].minTemperature=200;metadata[i].maxTemperature=4000;metadata[i].referencePressure=101325;}
 auto&t=state.gasSpecies.thermo;t.species=metadata;t.coefficients=coefficients;t.coefficientCount=6;t.elementComposition=elements;t.elementCount=1;t.speciesOrderHash=1;
 ugkwp::GasReactionData<double> reaction;ugkwp::GasStoichTerm<double> reactant,product;
 reaction.reactantCount=reaction.productCount=1;reaction.highRate.preExponential=2;reactant.species=0;reactant.coefficient=1;product.species=1;product.coefficient=1;
 auto&mech=state.gasSpecies.mechanism;mech.reactions=&reaction;mech.reactionCount=1;mech.reactants=&reactant;mech.reactantTermCount=1;mech.products=&product;mech.productTermCount=1;mech.referencePressure=101325;mech.speciesOrderHash=1;mech.stoichiometricBasis=basis;mech.independentRank=1;
 state.gasMu=2e-5;state.gasPr=state.gasPrClamped=.5;state.gasFluxScheme=2;state.gasReconstruction=1;state.gasTimeIntegrator=3;
 state.gammaGas=1.4;state.Rgas=287;state.gasCp=1000;state.maxDiffusionNumber=.125;state.rhoMin=1e-12;state.TgasMin=1e-9;
 state.sstCoefficients=ugkwp::defaultSstCoefficients();state.sstKMin=state.sstOmegaMin=1e-12;state.sstMaxSourceNumber=.25;
 state.hostTurbulenceModel=model?3:0;device=state;const DeviceState before=device;
 // Actual production lifecycle: upload valid device state, scrub the host copy,
 // then call the public wall configuration ABI. No restoration of scrubbed data.
 scrubHostCalculationScalars(&state);assert(state.gasMu==0&&state.gasPrClamped==0&&state.gasFluxScheme==0);
 residentTarget=&device;residentBytes=sizeof(device);if(failPublication==-2)state.deviceState=nullptr;
 ugkwpGpuIpc::BoundaryLayerConfigV1 cfg;cfg.budgetAudit=audit;cfg.model=model;cfg.nodes=32;cfg.speciesCount=2;cfg.wallCount=1;cfg.quadratureCount=1;cfg.matchingCount=1;cfg.workspaceSlots=1;
 int faces[]={0},offset[]={0,1},match[]={1};double geometry[]={0,1,0,.004,.01,4e-7,1.6e-9},qd[]={.004},qw[]={4e-7},mw[]={1},flux[]={0,0};
 auto configure=[&](){return ugkwpGpuResidentStrictConfigureBoundaryLayerV1(&state,&cfg,faces,offset,offset,match,geometry,qd,qw,mw,flux);};
 const int rc=configure();checkUnrelated(before,device);
 if(failPublication!=-1){assert(rc!=0&&state.gasModelPoisoned);assert(publicationAttempts==(failPublication==-2?0:failPublication+1));
  const int calls=publicationAttempts,live=allocations;assert(configure()!=0&&publicationAttempts==calls&&allocations==live);
 }else{assert(rc==0&&!state.gasModelPoisoned&&device.gasBoundaryLayer.enabled&&publicationAttempts==6);
  assert(device.gasSstAudit.enabled==bool(audit));
  assert(bool(device.gasSstAudit.transportK)==bool(audit));
  assert(device.gasMu==2e-5&&device.gasPrClamped==.5&&device.gasFluxScheme==2);
  assert(ugkwp::evaluateGasBoundaryLayerSlot(device,0));
  assert(device.gasBoundaryLayer.exchange[0].ready&&std::isfinite(device.gasBoundaryLayerModel.output[0].conductiveHeatFlux));
  if(model)assert(device.gasBoundaryLayerModel.output[0].reactionIntegral[0]<-1e-8);
  assert(state.gasMu==0&&state.gasPrClamped==0&&state.gasFluxScheme==0);
  const int calls=publicationAttempts,live=allocations;
  assert(configure()!=0&&publicationAttempts==calls&&allocations==live);
 }
 releaseBoundaryLayerStorage(state);assert(allocations==0);
 assert(!state.gasBoundaryLayer.enabled&&!state.gasBoundaryLayerModel.input&&!state.gasSstAudit.enabled);
 assert(!state.gasSstAudit.transportK&&!state.gasSstAudit.volume);
 releaseBoundaryLayerStorage(state);assert(allocations==0);
}
'''


import pytest


@pytest.fixture(scope='module')
def publication_probe(tmp_path_factory):
    path=tmp_path_factory.mktemp('wall_publication')
    source=path/'publication.cpp';source.write_text(_publication_probe())
    binary=path/'publication'
    built=subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    return binary


@pytest.mark.parametrize('model,audit',[(0,0),(1,0),(1,1)])
@pytest.mark.parametrize('failed_publication',[-2,-1,0,1,2,3,4,5])
def test_wall_publication_preserves_scrubbed_device_state_and_poisons_partial_copy(publication_probe,model,audit,failed_publication):
    run=subprocess.run([str(publication_probe),str(model),str(failed_publication),str(audit)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
