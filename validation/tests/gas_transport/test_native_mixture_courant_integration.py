import subprocess

from native_gas_integration_fixture import COMPARE_SPECIES, function
from test_current_gas_courant import CUDA, launches
from test_mixture_transport import fixture
from tool.precision.materialize_sources import materialize
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT


def test_actual_native_mixture_courant_complete_outputs_and_errors(tmp_path):
    here = VALIDATION_ROOT / 'tests/gas_transport'
    fp32 = materialize(SOURCE_ROOT, 32)
    backend = 'applications/gasUGKP/private_backend'
    native = (fp32 / 'common/gasTransport/ResidentCourant.inl').read_text()
    wrapper = function(native, 'extern "C" int ugkwpGpuResidentStrictComputeGasCourant')
    policy = (SOURCE_ROOT / backend / 'SharedGasTrialPolicy.cuh').read_text()
    validate = function(policy, 'static int validate(')
    kernel_calls = launches((SOURCE_ROOT / 'common/GpuGasCourant.cuh').read_text())
    reduction = (fp32 / 'common/GpuGasCourantReduction.cuh').read_text()
    reduction = '\n'.join(line for line in reduction.splitlines() if not line.startswith('#include'))
    probe = (here / 'gas_state_probe.cpp').read_text().split('void hashBytes', 1)[0]
    host = r"""
#include <cassert>
#include <stdexcept>
struct Observer {
    ugkp::diagnostics::DeviceContext context{};
    int reads=0;
    int observeCompletedWork(cudaStream_t=nullptr){++reads;return 0;}
    int dispatchPending(){return ugkp::diagnostics::hasFatal(&context);}
};
struct DeviceState : Host {
    Observer observer;
    Observer* diagnosticOwner=&observer;
    DeviceState(){diagnostics=&observer.context;}
    unsigned long long* gasStatusFirstUnavailable=nullptr;
    cudaStream_t gasCaptureStream=nullptr;
    bool gasModelPoisoned=false,advanceRecording=false;
};
DeviceState* asState(void* p){return static_cast<DeviceState*>(p);}
int validateState(DeviceState* p,const char*){return p?0:1;}
int publishStartupWorkPlan(DeviceState*,double,double){return 0;}
int cudaStreamSynchronize(cudaStream_t){return 0;}
namespace ugkwp {
int enqueueGasStatusCheck(const int* cells,int n,const int* faces,int m,ugkp::diagnostics::DeviceContext* context,
    unsigned long long*,unsigned,unsigned,cudaStream_t){
    const int missing=int(GasTransportCode::InvalidStorage);
    if(std::find(cells,cells+n,missing)!=cells+n||std::find(faces,faces+m,missing)!=faces+m)
        ugkp::diagnostics::recordFatal(context,ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::MissingOperator));
    return 0;
}
}
namespace thrust {
namespace cuda {struct Execution {int on(cudaStream_t){return 0;}} par;}
template<class T> const T* device_pointer_cast(const T* p){return p;}
template<class T,class F> double reduce(int,const T* begin,const T* end,double v,F f){
    ++reductions;
    for(;begin!=end;++begin)v=f(v,*begin);
    return v;
}
}
template<class T>void sameField(const T& left,const T& right){
    if constexpr(std::is_pointer_v<T>)assert(std::memcmp(left,right,8*sizeof(*left))==0);
}
"""
    main = r"""
int main(){
    static_assert(sizeof(Real)==4);
    for(bool moving:{false,true})for(bool sst:{false,true})
    for(bool diffusion:{false,true})for(bool noFaces:{false,true}){
        DeviceState manual,api;
        GpuGeometry oldVolume[]={1,1},newVolume[]={.99,1.01},sweep[]={.01,0,0};
        Real diffusivity[]={Real(.1),Real(.2)};
        for(auto* s:{&manual,&api}){
            initialise(*s);
            s->turbulenceModel=s->hostTurbulenceModel=sst?3:0;
            s->sstConfigured=sst;s->gasMu=diffusion?Real(1e5):0;
            s->gasSpecies.diffusivity=diffusion?diffusivity:nullptr;
            s->gasGeometry.enabled=moving;
            s->gasGeometry.oldVolume=oldVolume;s->gasGeometry.newVolume=newVolume;
            s->gasGeometry.faceSweptVolume=sweep;s->gasGeometry.interval=.001;
            if(noFaces){
                s->nFaces=s->nInternalFaces=0;
                s->cellPlaneCount[0]=s->cellPlaneCount[1]=0;
            }
        }
        GasCourantMaxima expected;
        reductions=0;
        assert(computeGasCurrentCourant(&manual,.001,2.,4,4,nullptr,expected,Maximum{})==0);
        assert(reductions==1);
        double actual=-777;
        assert(ugkwpGpuResidentStrictComputeGasCourant(&api,.001,2.,&actual)==0);
        assert(reductions==2&&actual==expected.maximum&&api.observer.reads==1);
        assert(noFaces?actual==0:actual>0);
#define UGKWP_GAS_STATE_FIELD(Type,Name) sameField(manual.Name,api.Name);
#include "gasTransport/GasStateFields.inc"
#undef UGKWP_GAS_STATE_FIELD
        compareSpeciesFields(manual,api);
        assert(std::memcmp(manual.gasSpecies.cellStatus,api.gasSpecies.cellStatus,2*sizeof(int))==0);
        assert(std::memcmp(manual.gasSpecies.faceStatus,api.gasSpecies.faceStatus,3*sizeof(int))==0);
    }
    DeviceState s;
    initialise(s);
    ugkp::boundary::Face faces[3];
    using ugkp::boundary::Term;
    Term terms[]={Term::table(0,100000),Term::table(2,200000)};
    ugkp::boundary::Function table;
    table.kind=ugkp::boundary::FunctionKind::Table;table.count=2;
    faces[1].function[ugkp::boundary::Pressure]=0;
    s.gasBoundaryProgram={faces,&table,terms,3};
    s.gasBoundaryPFix[1]=s.riemannBoundaryPFix[1]=1;
    double out=-777;
    for(double sample:{0.,.75,2.}){
        assert(ugkwpGpuResidentStrictComputeGasCourant(&s,.001,sample,&out)==0);
        const Real expected=Real(100000+50000*sample);
        assert(s.gasBoundaryP[1]==expected&&s.riemannBoundaryP[1]==expected);
    }
    s.gasBoundaryProgram={};
    out=-777;
    assert(ugkwpGpuResidentStrictComputeGasCourant(nullptr,.001,2.,&out)!=0&&out==-777);
    assert(ugkwpGpuResidentStrictComputeGasCourant(&s,.001,2.,nullptr)!=0);
    const auto* coefficients=s.gasSpecies.thermo.coefficients;
    s.gasSpecies.thermo.coefficients=nullptr;
    assert(ugkwpGpuResidentStrictComputeGasCourant(&s,.001,2.,&out)!=0&&out==-777);
    assert(s.gasSpecies.cellStatus[0]!=0&&s.observer.context.firstFatal.entityKind==ugkp::diagnostics::EntityKind::Cell);
    s.gasSpecies.thermo.coefficients=coefficients;
    s.observer.context={};
    std::fill_n(s.gasSpecies.cellStatus,2,0);std::fill_n(s.gasSpecies.faceStatus,3,0);
    assert(ugkwpGpuResidentStrictComputeGasCourant(&s,.001,2.,&out)==0&&out>0);
    s.gasGeometry.enabled=true;s.gasGeometry.faceSweptVolume=nullptr;out=-777;
    assert(ugkwpGpuResidentStrictComputeGasCourant(&s,.001,2.,&out)!=0&&out==-777);
    assert(s.gasSpecies.faceStatus[0]==int(ugkwp::GasTransportCode::InvalidStorage));
    assert(s.observer.context.firstFatal.entityKind==ugkp::diagnostics::EntityKind::Face);
    assert(s.observer.context.firstFatal.code==ugkp::diagnostics::Code::MissingOperator);
    s.gasGeometry.enabled=false;s.observer.context={};
    std::fill_n(s.gasSpecies.cellStatus,2,0);std::fill_n(s.gasSpecies.faceStatus,3,0);
    failLaunch=launches+1;
    assert(ugkwpGpuResidentStrictComputeGasCourant(&s,.001,2.,&out)!=0&&out==-777);
    failLaunch=0;
    assert(ugkwpGpuResidentStrictComputeGasCourant(&s,1e16,2.,&out)==0&&out>.5*OfGreat);
}
"""
    compare_species = COMPARE_SPECIES.replace('double', 'Real')
    source = tmp_path / 'courant.cpp'
    gas_fixture = fixture().replace('GasStateView<Real,Time,ugkwp::SstCoefficients>',
                                    'GasStateView<Real,Time,ugkwp::SstCoefficients,GpuEnergy,GpuGeometry,GpuWallEnergy>')
    source.write_text('\n'.join((probe, gas_fixture, CUDA, host, reduction, kernel_calls,
        'struct ResidentCourantTrialPolicy {', validate, '};', wrapper, compare_species, main)))
    binary = tmp_path / 'courant'
    includes = (fp32 / 'validation/tests/gas_transport', fp32 / 'common', fp32 / 'common/gasNumerics')
    subprocess.run(['g++', '-std=c++17', '-O0', *(f'-I{path}' for path in includes),
                    str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
