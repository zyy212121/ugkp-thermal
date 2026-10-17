"""Regression tests for actual caloric code and extracted resident readback bodies.

CUDA storage/copy is host-emulated; these tests make no GPU runtime claim.
"""
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]


def compile_run(tmp_path, code):
    (tmp_path/'cuda_runtime.h').write_text(r"""#pragma once
#include <cstdlib>
#include <cstring>
typedef int cudaError_t;constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
inline int cudaMalloc(void** p,size_t n){*p=std::malloc(n);return *p?0:1;}
inline int cudaFree(void* p){std::free(p);return 0;}
inline int cudaMemcpy(void*d,const void*s,size_t n,int){std::memcpy(d,s,n);return 0;}
""")
    source=tmp_path/'probe.cpp'; source.write_text(code)
    binary=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-I'+str(ROOT),'-I'+str(tmp_path),str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)


def body(path, name):
    text=(ROOT/path).read_text(); start=text.index('extern "C" int '+name)
    opening=text.index('{',start); depth=1; end=opening+1
    while depth:
        depth += (text[end]=='{')-(text[end]=='}'); end+=1
    return text[start:end]


@pytest.mark.parametrize('mode', [0,1])
def test_nonfinite_caloric_intermediate_is_fatal(tmp_path, mode):
    compile_run(tmp_path,r'''
#include "common/gasTransport/MixtureThermo.H"
#include <cassert>
using namespace ugkwp; using namespace ugkp::diagnostics;
struct Evaluator { ThermoStatus operator()(double,double&e,double&c)const {
 e=1e308;c=MODE==0?1.:1e-308;return ThermoStatus::Success;} };
int main(){ DeviceContext ctx{};double T=400;bool evaluated=true;
 ThermoInversionControls<double> controls;controls.maximumIterations=1;
 auto status=invertCaloricEnergy(Evaluator{},MODE==0?-1e308:0.,100.,3000.,controls,T,&evaluated,&ctx);
 assert(status==ThermoStatus::NonFiniteInput);assert(!evaluated);assert(hasFatal(&ctx));
 assert(ctx.warnings[1].count==0);assert(T==400);
}
'''.replace('MODE',str(mode)))


@pytest.mark.parametrize('application',['gasUGKP','FSH'])
@pytest.mark.parametrize('name',['DownloadSourceResidualMass','DownloadGasBoundaryFields','DownloadSst'])
def test_empty_download_has_no_completion_copy(tmp_path,application,name):
    function=body(f'applications/{application}/private_backend/GpuResidentStrict.cu','ugkwpGpuResidentStrict'+name)
    args={'DownloadSourceResidualMass':'&s,&count,1,&face,&x',
          'DownloadGasBoundaryFields':'&s,&x,&x,&x,&x,&x,&x',
          'DownloadSst':'&s,&x,&x,&x'}[name]
    compile_run(tmp_path,r'''
#include <cstddef>
#include <cassert>
using std::size_t;
int copies=0;
struct Owner{int completeReadback(int status=0,bool completed=true){if(completed)++copies;return status;}};
struct DeviceState{int nBoundarySources=0,nFaces=0,nCells=0,hostTurbulenceModel=3,sstConfigured=1;
 int*sourceFace=nullptr; double *sourceResidualMass=nullptr,*gasBoundaryRho=nullptr,*gasBoundaryUx=nullptr,
 *gasBoundaryUy=nullptr,*gasBoundaryUz=nullptr,*gasBoundaryP=nullptr,*gasBoundaryT=nullptr,*k=nullptr,*omega=nullptr,*nut=nullptr;Owner*diagnosticOwner;};
DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}int validateState(DeviceState*,const char*){return 0;}
void setLastErrorText(const char*){}
template<class T> int copyToHost(T*,const T*,size_t n,const char*){if(n)++copies;return 0;}
'''+function+f'\nint main(){{Owner owner;DeviceState s;s.diagnosticOwner=&owner;int count=0,face=0;double x=0;assert(ugkwpGpuResidentStrict{name}({args})==0);assert(copies==0);}}')


def test_profile_readiness_drains_fatal_first(tmp_path):
    function=body('applications/gasUGKP/private_backend/BoundaryLayerProfileAdapter.cuh','ugkwpGpuResidentStrictDownloadBoundaryLayerProfilesV1')
    compile_run(tmp_path,r'''
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
namespace ugkwp{constexpr int compiledGasSpecies=1;template<class T>struct GasBoundaryLayerExchange{bool ready=false;};}
#include "common/GpuDiagnosticsOwner.cuh"
using Owner=ugkp::diagnostics::GpuOwner;
bool ordinaryError=false;
struct DeviceState{struct{double*profile;struct{int nodes=2;}config;}gasBoundaryLayerModel;
 struct{int count=1;ugkwp::GasBoundaryLayerExchange<double>*exchange;}gasBoundaryLayer;
 double gasBoundaryLayerStageTime=0;Owner*diagnosticOwner;};
DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}int validateState(DeviceState*,const char*){return 0;}
void setLastErrorText(const char*){ordinaryError=true;}
template<class T>int copyToHost(T*d,const T*s,std::size_t n,const char*){if(n)std::memcpy(d,s,n*sizeof(T));return 0;}
'''+function+r'''
int main(){Owner owner;assert(owner.initialise()==0);DeviceState s;ugkwp::GasBoundaryLayerExchange<double> exchange;double profiles[16]{},time=0;
using namespace ugkp::diagnostics;
recordFatal(owner.device(),makeRecord(Code::NonFinite,30600));
recordFatal(owner.device(),makeRecord(Code::InvalidIndex,30601));
s.diagnosticOwner=&owner;s.gasBoundaryLayer.exchange=&exchange;s.gasBoundaryLayerModel.profile=profiles;
assert(ugkwpGpuResidentStrictDownloadBoundaryLayerProfilesV1(&s,1,2,1,profiles,&time)==1);
Snapshot snapshot;owner.finish(snapshot);assert(snapshot.hasFatal);assert(snapshot.fatal.sourceSiteId==30600);assert(!ordinaryError);}
''')


def test_wall_readback_observation_precedes_interpretation():
    text=(ROOT/'applications/gasUGKP/private_backend/BoundaryLayerAdapter.cuh').read_text()
    config=text[:text.index('extern "C" int ugkwpGpuResidentStrictDownloadBoundaryLayerV1')]
    assert config.index('completeReadback') < config.index('faceSlot[f]=slot')
    download=text[len(config):]
    assert download.index('completeReadback') < download.index('if(!exchange[slot].ready)')


def test_warning_sample_is_labelled_historical():
    text=(ROOT/'common/DiagnosticsFoam.H').read_text()
    assert 'historical representative sample' in text


def test_empty_species_download_has_no_completion_copy(tmp_path):
    function=body('applications/gasUGKP/private_backend/SharedGasAdapter.cuh','ugkwpGpuResidentStrictDownloadSpeciesV1')
    compile_run(tmp_path,r'''
#include <vector>
#include <algorithm>
#include <cassert>
namespace ugkwp{constexpr int compiledGasSpecies=1;}
namespace ugkwpGpuIpc{struct GasSpeciesIdentityV1{};}
int copies=0;
struct Owner{int completeReadback(int status=0,bool completed=true){if(completed)++copies;return status;}};
struct DeviceState{int nCells=0;struct{double*rho;}gasSpecies;Owner*diagnosticOwner;};
DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}int validateState(DeviceState*,const char*){return 0;}
void setLastErrorText(const char*){}
template<class T>int copyToHost(T*,const T*,std::size_t n,const char*){if(n)++copies;return 0;}
'''+function+r'''
int main(){Owner owner;DeviceState s;double value=0;s.gasSpecies.rho=&value;s.diagnosticOwner=&owner;
assert(ugkwpGpuResidentStrictDownloadSpeciesV1(&s,nullptr,&value)==0);assert(copies==0);}
''')


def test_finite_caloric_residual_still_returns_warning(tmp_path):
    compile_run(tmp_path,r'''
#include "common/gasTransport/MixtureThermo.H"
#include <cassert>
using namespace ugkwp; using namespace ugkp::diagnostics;
struct Evaluator{ThermoStatus operator()(double T,double&e,double&c)const{e=T;c=1;return ThermoStatus::Success;}};
int main(){DeviceContext ctx{};double T=400;bool evaluated=false;
ThermoInversionControls<double> controls;controls.maximumIterations=1;
auto status=invertCaloricEnergy(Evaluator{},900.,100.,3000.,controls,T,&evaluated,&ctx);
assert(status==ThermoStatus::Success&&evaluated&&!hasFatal(&ctx));
assert(ctx.warnings[1].count==1&&ctx.warnings[1].sample.observed==-500);assert(T==400);}
''')


def test_warning_count_is_new_but_sample_is_lifetime_first(tmp_path):
    compile_run(tmp_path,r'''
#include "common/GpuDiagnosticsOwner.cuh"
#include <cassert>
using namespace ugkp::diagnostics;
int main(){GpuOwner owner;assert(owner.initialise()==0);
recordWarning(owner.device(),withOperation(makeRecord(Code::AccuracyResidual,2303,0,0,EntityKind::Cell,5,-1,3),1,10,.1));
assert(owner.completeReadback()==0);Snapshot first;owner.finish(first);
assert(first.warnings.size()==1&&first.warnings[0].count==1);
recordWarning(owner.device(),withOperation(makeRecord(Code::AccuracyResidual,2422,0,0,EntityKind::Face,9,-1,99),2,20,.2));
assert(owner.completeReadback()==0);Snapshot second;owner.finish(second);
assert(second.warnings.size()==1&&second.warnings[0].count==1);
assert(second.warnings[0].sample.sourceSiteId==2303&&second.warnings[0].sample.operationSequence==1);
assert(second.warnings[0].sample.observed==3);
assert(owner.completeReadback()==0);owner.finish(second);assert(second.warnings.empty());}
''')
