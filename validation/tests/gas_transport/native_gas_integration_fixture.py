"""Native selected entry bodies + real numerical kernels, serial CUDA shims only."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import re
from pathlib import Path
from test_gas_advance_view import PREAMBLE
from test_legacy_ale_transport import fixture
from validation.solvers.gasUGKP.tests.test_shared_gas_adapter import ADAPTER_PROBE
from test_current_gas_courant import launches
ROOT=SOURCE_ROOT

def function(source,marker):
    start=source.index(marker);brace=source.index('{',start);level=1;i=brace+1
    while level:
        level+=(source[i]=='{')-(source[i]=='}');i+=1
    return source[start:i]

def native_fixture(source_root=None, include_combined=True):
    root=ROOT if source_root is None else source_root
    native=(root/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    h=ADAPTER_PROBE.split('int main(){',1)[0]
    a=h.index('struct DeviceState {');b=h.index('DeviceState* asState',a)
    host=r'''
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include "applications/gasUGKP/private_backend/SharedGasTrialFields.H"
struct DeviceState:State{
 ugkwp::GasSpeciesState<double,2> gasRejectedView;
 ugkwp::GasBoundaryLayerModelState<double,2> gasBoundaryLayerModel;
 SharedGasTrialStorage gasTrial;
 bool gasModelPoisoned=false,gasInitialFieldsUploaded=false,gasSpeciesUploaded=false,hostGravityActive=false,particlesMayBePresent=false;
 DeviceState*deviceState=this;int particleCapacity=0,gasTimeIntegrator=1,nBoundarySources=0;
 int fixedCellBlockThreads=4,fixedFaceBlockThreads=4,hostGasTimeIntegrator=1,hostTurbulenceModel=0,hostGasFluxScheme=1,hostGasReconstruction=0,hostGasLimiter=0,hasPeriodicFaces=0;
 double gasBoundaryLayerStageTime=0,gravityX=0,gravityY=0,gravityZ=0;
 int*particleCountDevice=nullptr,*pCellId=nullptr,*pStatus=nullptr;double*px=nullptr,*py=nullptr,*pz=nullptr,*pux=nullptr,*puy=nullptr,*puz=nullptr,*puxOld=nullptr,*puyOld=nullptr,*puzOld=nullptr,*pT=nullptr,*pTheta=nullptr,*pd=nullptr,*pm=nullptr;
 cudaStream_t gasCaptureStream=nullptr;cudaGraph_t gasGraph=nullptr;cudaGraphExec_t gasGraphExec=nullptr;double gasGraphDt=-1;
 std::vector<cudaGraphNode_t>gasGraphTimeNodes;std::vector<cudaKernelNodeParams>gasGraphTimeParams;
};
'''
    # Match the selected revision's real startup callback interface. The older
    # immutable oracle revision predates it and retains its original host shape.
    callbacks='int (*gasBoundaryLayerLaunch)(DeviceState*)'
    if callbacks in native:
        declarations="""
struct DeviceState;
template<bool Capture> int launchBoundaryLayer(DeviceState*);
namespace sharedGasTrialDetail {
template<bool Capture,class State> int snapshotWallRecords(State*,SharedGasTrialSnapshot&,bool);
}
"""
        fields="""
 int (*gasBoundaryLayerLaunch)(DeviceState*)=&launchBoundaryLayer<false>;
 int (*gasBoundaryLayerSnapshot)(DeviceState*,SharedGasTrialSnapshot&,bool)=
     &sharedGasTrialDetail::snapshotWallRecords<false,DeviceState>;
"""
        host=host.replace('struct DeviceState:State{',declarations+'struct DeviceState:State{'+fields)
    h=h[:a]+host+h[b:]
    graph=PREAMBLE.split('#ifdef LEGACY_REFERENCE',1)[0].replace('using cudaError_t=int;','').replace('cudaSuccess=0,','')
    graph=re.sub(r'void setLastError(?:Text)?\([^\n]+\n','',graph)
    h=h.replace('struct DeviceState:State{', 'struct DeviceState:State{')
    start=h.index('#include "common/gasTransport/GasBoundaryLayerModelState.H"')
    h=h[:start]+graph+h[start:]
    state=fixture().replace('s.gasSpecies.cellStatus=new int[2]{};s.gasSpecies.faceStatus=new int[3]{};','')
    api=r'''
int cudaDeviceSynchronize(){return 0;}
template<class F>void emulateLaunch(F f,int grid,int block){blockDim.x=block;for(blockIdx.x=0;blockIdx.x<grid;++blockIdx.x)for(threadIdx.x=0;threadIdx.x<block;++threadIdx.x)f();}
int tuneFixedWorkBlockThreads(DeviceState*,double,double){return 0;}
SQR3_FUNCTION
#include "GpuGravityUpdate.cuh"
#include "operators/advanceGasChemistryKernel.cuh"
template<class S,class T>void advanceGasChemistryKernel(S*s,T dt,const Real*v){int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<s->nCells)ugkwp::advanceGasChemistryCell(*s,c,dt,v[c]);}
#include "operators/evaluateGasBoundaryLayerKernel.cuh"
'''
    if 'nativeRefinement' in (root/'common/operators/advanceGasChemistryKernel.cuh').read_text():
        api=api.replace('const Real*v){','const Real*v,bool nativeRefinement=false){').replace('(*s,c,dt,v[c]);','(*s,c,dt,v[c],nativeRefinement);')
    api=api.replace("SQR3_FUNCTION",function((root/"common/operators/publishScheduledInletConfigurationKernel.cuh").read_text(),"__device__ GPU_OPERATOR_REAL sqr3"))
    if callbacks in native:
        if 'evaluateGasBoundaryLayerTeamKernel' in native:
            # Host CUDA shims execute the real scalar math once per CTA owner;
            # cooperative math/barriers have separate threaded oracle tests.
            api+=r'''
template<bool Capture,class GasState> void evaluateGasBoundaryLayerTeamKernel(GasState* state){
 if(threadIdx.x)return;
 const int worker=blockIdx.x,workers=state->gasBoundaryLayerModel.workspaceCount;
 if(worker>=workers || !ugkwp::gasBoundaryLayerEnabled(*state))return;
 for(int slot=worker;slot<state->gasBoundaryLayer.count;slot+=workers)
  ugkwp::evaluateGasBoundaryLayerSlot<Capture>(*state,slot,worker);
}
'''
        api+=launches(function(native,'template<bool Capture> int launchBoundaryLayer(DeviceState* s)'))

    gravity=function(native,'__global__ void applyGasGravitySourceKernel')+'\n'+function(native,'int applyGasGravitySource\n')
    policy='#include "GpuGasHostPolicy.cuh"\n'+function(native,'struct GasHostPolicy:')+';\n'
    sst=function(native,'int syncSstConfiguration')+'\n'+function(native,'extern "C" int ugkwpGpuResidentStrictConfigureSst')
    gas=function(native,'extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly\n').split('#ifndef UGKP_DEVELOPMENT_PROBES',1)[0]+'std::abort();}\n'
    combined=""
    if include_combined:
        combined=function(native,'extern "C" int ugkwpGpuResidentStrictAdvance\n').split('    if (tuneFixedWorkBlockThreads',1)[0]

        combined=combined[:combined.index('    if (s->gasSpecies.mode == ugkwp::GasMode::SingleLegacy')] if '    if (s->gasSpecies.mode == ugkwp::GasMode::SingleLegacy)'in combined else combined
        end=combined.index('    if(s->gasSpecies.mode!=ugkwp::GasMode::SingleLegacy)')
        branch=function(combined[end:],'if(s->gasSpecies.mode!=ugkwp::GasMode::SingleLegacy)')
        combined=combined[:end]+branch+'\nstd::abort();}\n'
    fixture_root = VALIDATION_ROOT/'tests/gas_transport' if root == ROOT else root/'tests/gas_transport'
    pre=(fixture_root/'gas_state_probe.cpp').read_text().split('void hashBytes',1)[0]
    return pre+state+h+api+launches(gravity)+'\n#include "applications/gasUGKP/private_backend/SharedGasTrialPolicy.cuh"\n'+launches(policy)+launches((root/'common/GpuGasAdvance.cuh').read_text())+launches(sst)+gas+combined

MODEL_SETUP=r'''
const std::string modelText=R"(gasMode mixtureFrozen;species(A B);speciesThermo{
A{model linearCp;molarMass .028;minTemperature 100;maxTemperature 4000;coefficients(1040 0 0);}
B{model linearCp;molarMass .032;minTemperature 100;maxTemperature 4000;coefficients(1100 0 2000);}}
diffusion{model constant;coefficients(.02 .03);turbulentSchmidt .7;})";
void setupNative(DeviceState&s,int capacity,int turbulence){initialise(s);s.particleCapacity=capacity;s.turbulenceModel=s.hostTurbulenceModel=turbulence;
 const auto model=ugkwp::parseGasModelProperties(modelText);ugkwpGpuIpc::GasModelConfigureArgsV1 args{1,1,2,0,model.speciesOrderHash,model.thermoHash,0,modelText.size(),0};
 assert(ugkwpGpuResidentStrictConfigureGasModelV1(&s,&args,modelText.data(),nullptr)==0);
 s.rho[0]=s.rho[1]=1;s.rhoUx[0]=2;s.rhoUx[1]=4;
 double species[4]={.7,.2,.3,.8};assert(ugkwpGpuResidentStrictUploadSpeciesV1(&s,nullptr,species)==0);
 for(int c=0;c<2;++c){double partial[2]={species[c],species[2+c]};s.rhoE[c]=ugkwp::mixtureEnergy(partial,700.+100*c,s.gasSpecies.thermo)+.5*s.rhoUx[c]*s.rhoUx[c];}
 const auto a=ugkwp::defaultSstCoefficients();double k[2]={.1,.2},omega[2]={3,4},distance[2]={.5,.5},boundary[3]={0,0,0};int modes[3]={0,0,0};
 assert(ugkwpGpuResidentStrictConfigureSst(&s,a.alphaK1,a.alphaK2,a.alphaOmega1,a.alphaOmega2,a.beta1,a.beta2,a.betaStar,a.gamma1,a.gamma2,a.a1,a.b1,a.c1,1e-12,1e-9,.25,0,.41,9.8,.09,k,omega,distance,modes,modes,boundary,boundary)==0);
}
'''


COMPARE_SPECIES=r"""
void compareSpeciesFields(const DeviceState&a,const DeviceState&b){
 const double*left[]={a.gasSpecies.rho,a.gasSpecies.initial,a.gasSpecies.flux,a.gasSpecies.gradX,a.gasSpecies.gradY,a.gasSpecies.gradZ,a.gasSpecies.limiter,a.gasSpecies.boundaryMassFraction,a.gasSpecies.positivityScale,a.gasSpecies.soundSpeed,a.gasSpecies.heatCapacity,a.gasSpecies.gasConstant};
 const double*right[]={b.gasSpecies.rho,b.gasSpecies.initial,b.gasSpecies.flux,b.gasSpecies.gradX,b.gasSpecies.gradY,b.gasSpecies.gradZ,b.gasSpecies.limiter,b.gasSpecies.boundaryMassFraction,b.gasSpecies.positivityScale,b.gasSpecies.soundSpeed,b.gasSpecies.heatCapacity,b.gasSpecies.gasConstant};
 const int count[]={4,4,6,4,4,4,2,6,4,2,2,2};
 for(int k=0;k<12;++k)assert(std::memcmp(left[k],right[k],count[k]*sizeof(double))==0);
 assert(std::memcmp(a.gasSpecies.compositionBoundaryFixed,b.gasSpecies.compositionBoundaryFixed,3*sizeof(int))==0);
}
"""
