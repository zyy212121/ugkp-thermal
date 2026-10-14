"""Revision-correct native entry/kernel/policy execution with serial CUDA shims.

No GPU arithmetic/timing claim. The original immutable tree supplies all native
and common production dependencies, not only the wrapper body.
"""
from pathlib import Path
import io
import re
import subprocess
import tarfile
import pytest
from native_gas_integration_fixture import ROOT, native_fixture, MODEL_SETUP

ORIGINAL='d6110c02745a8ad726208a426c4bb3a28233c1fd'

@pytest.fixture(scope='module')
def original_native_root(tmp_path_factory):
    root=tmp_path_factory.mktemp('original-native-tree')
    archive=subprocess.check_output(['git','archive',ORIGINAL,'common','applications/gasUGKP','tests/gas_transport'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:source.extractall(root,filter='data')
    return root


def source(root):
    setup=MODEL_SETUP.replace('double species[4]={.7,.2,.3,.8};',
        's.gasInitialFieldsUploaded=true;ugkwpGpuIpc::GasSpeciesIdentityV1 identity{1,2,model.speciesOrderHash,model.thermoHash,0};double species[4]={.7,.2,.3,.8};')
    setup=setup.replace('UploadSpeciesV1(&s,nullptr,species)','UploadSpeciesV1(&s,&identity,species)')
    setup=setup.replace('assert(ugkwpGpuResidentStrictConfigureSst','if(turbulence==3)assert(ugkwpGpuResidentStrictConfigureSst')
    body=native_fixture(root,include_combined=False)
    body=body.replace('    if (TrialPolicy::begin(s) != 0)',
        '    std::printf("TRIAL %a %a\\n",double(dt),double(simulationTime));\n    if (TrialPolicy::begin(s) != 0)')
    fields=re.findall(r'    Real\* (\w+)\{\};',(ROOT/'tests/gas_transport/legacy_gas_fixture.hpp').read_text())
    dump='\n'.join('std::printf("'+f+' ");for(int i=0;i<8;++i)std::printf("%a ",s.'+f+'[i]);std::puts("");' for f in fields)
    return body+setup+r'''
int main(){for(double dt:{.0001,.002,.01}){DeviceState s;setupNative(s,0,0);trace.clear();
 const int rc=ugkwpGpuResidentStrictAdvanceGasOnly(&s,dt,2.);assert(rc==0);
 std::printf("CASE %a %d %s\n",dt,rc,last.c_str());
 DUMP_FIELDS
 const double*species[]={s.gasSpecies.rho,s.gasSpecies.initial,s.gasSpecies.flux,s.gasSpecies.gradX,s.gasSpecies.gradY,s.gasSpecies.gradZ,s.gasSpecies.limiter,s.gasSpecies.boundaryMassFraction,s.gasSpecies.positivityScale,s.gasSpecies.soundSpeed,s.gasSpecies.heatCapacity,s.gasSpecies.gasConstant};
 const int n[]={4,4,6,4,4,4,2,6,4,2,2,2};for(int k=0;k<12;++k){for(int j=0;j<n[k];++j)std::printf("%a ",species[k][j]);std::puts("");}
 for(int c=0;c<2;++c)std::printf("%d ",s.gasSpecies.cellStatus[c]);for(int i=0;i<3;++i)std::printf("%d ",s.gasSpecies.faceStatus[i]);std::puts("");
 releaseSharedGasTrialStorage(s.gasTrial);releaseSharedGasSpecies(s.gasSpecies);assert(allocations==0);
}}
'''.replace('DUMP_FIELDS',dump)


def build_run(folder,label,root,body):
    src=folder/(label+'.cpp');exe=folder/label;src.write_text(body)
    cmd=['g++','-std=c++17','-O0','-ffp-contract=off','-I'+str(root),'-I'+str(root/'common'),'-I'+str(root/'common/gasNumerics'),'-I'+str(root/'tests/gas_transport'),str(src),'-o',str(exe)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    (folder/(label+'_build.log')).write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True,timeout=180)
    (folder/(label+'.log')).write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout


def test_native_requested_interval_matches_original_fields_and_trial_times(tmp_path,original_native_root):
    before=build_run(tmp_path,'original',original_native_root,source(original_native_root))
    after=build_run(tmp_path,'corrected',ROOT,source(ROOT))
    assert after==before
    counts=[];count=0
    for line in before.splitlines():
        if line.startswith('TRIAL '):count+=1
        if line.startswith('CASE '):counts.append(count);count=0
    assert counts==[1,3,17]


def courant_source(root):
    from native_gas_integration_fixture import function
    from test_current_gas_courant import launches
    body=source(root).split('int main(){for(double dt:')[0]
    native=(root/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    body+=r'''
#include <algorithm>
char lastError[1024]{};
namespace thrust{
template<class T>using device_ptr=T*;int device=0;
template<class T>T*device_pointer_cast(T*p){return p;}
namespace cuda{struct Par{int on(cudaStream_t){return 0;}}par;}
template<class T>T*max_element(int,T*a,T*b){return std::max_element(a,b);}
template<class T,class F>T reduce(int,const T*a,const T*b,T value,F f){for(;a!=b;++a)value=f(value,*a);return value;}
}
'''
    if 'struct GasCourantMaximum' in native:
        body+=function(native,'struct GasCourantMaximum')+';\n'
        body+=launches((root/'common/GpuGasCourant.cuh').read_text())
    body+=launches(function(native,'extern "C" int ugkwpGpuResidentStrictComputeGasCourant'))
    fields=re.findall(r'    Real\* (\w+)\{\};',(ROOT/'tests/gas_transport/legacy_gas_fixture.hpp').read_text())
    dump='\n'.join('std::printf("'+f+' ");for(int i=0;i<8;++i)std::printf("%a ",s.'+f+'[i]);std::puts("");' for f in fields)
    return body+r'''
int main(){for(int moving:{0,1})for(int turbulence:{0,3})for(int visc:{0,1})for(int zero:{0,1}){
 DeviceState s;setupNative(s,0,turbulence);Real oldVolume[2]={1,1},newVolume[2]={1,1},sweep[3]={0x1p-10,0,0};
 s.gasMu=visc?1e5:0;if(!visc){std::free(const_cast<double*>(s.gasSpecies.diffusivity));--allocations;s.gasSpecies.diffusivity=nullptr;}
 s.gasGeometry.enabled=moving;s.gasGeometry.oldVolume=oldVolume;s.gasGeometry.newVolume=newVolume;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.001;s.gasGeometry.absoluteGeometryTolerance=s.gasGeometry.relativeGeometryTolerance=16*std::numeric_limits<double>::epsilon();
 if(zero){s.nFaces=0;for(int i=0;i<2;++i)s.cellPlaneCount[i]=0;}
 // Supply an original-valid moving mesh: volume change equals signed sweep.
 for(int c=0;c<2;++c)for(int i=0;i<s.cellPlaneCount[c];++i){const int f=s.cellFaceId[s.cellPlaneStart[c]+i];newVolume[c]+=(s.faceOwner[f]==c?1:-1)*sweep[f];}
 double value=-777;int rc=ugkwpGpuResidentStrictComputeGasCourant(&s,.001,.5,2.,&value);
 assert(rc==0);std::printf("CASE %d %d %d %d %d %a\n",moving,turbulence,visc,zero,rc,value);
 DUMP_FIELDS
 const double*species[]={s.gasSpecies.rho,s.gasSpecies.initial,s.gasSpecies.flux,s.gasSpecies.gradX,s.gasSpecies.gradY,s.gasSpecies.gradZ,s.gasSpecies.limiter,s.gasSpecies.boundaryMassFraction,s.gasSpecies.positivityScale,s.gasSpecies.soundSpeed,s.gasSpecies.heatCapacity,s.gasSpecies.gasConstant};
 const int n[]={4,4,6,4,4,4,2,6,4,2,2,2};for(int k=0;k<12;++k){for(int j=0;j<n[k];++j)std::printf("%a ",species[k][j]);std::puts("");}
 for(int c=0;c<2;++c)std::printf("%d ",s.gasSpecies.cellStatus[c]);for(int f=0;f<3;++f)std::printf("%d ",s.gasSpecies.faceStatus[f]);std::puts("");
 releaseSharedGasSpecies(s.gasSpecies);assert(allocations==0);
}}
'''.replace('DUMP_FIELDS',dump)


def test_native_courant_wrapper_revision_correct_complete_arrays(tmp_path,original_native_root):
    before=build_run(tmp_path,'original',original_native_root,courant_source(original_native_root))
    after=build_run(tmp_path,'corrected',ROOT,courant_source(ROOT))
    assert before==after
    assert sum(line.startswith('CASE ') for line in before.splitlines())==16


def test_native_terminal_effort_switch_preserves_actual_public_calculation(tmp_path):
    base=source(ROOT).split('int main(){for(double dt:')[0]
    body=base+r'''
int main(){for(int limit:{0,1}){DeviceState native,direct;setupNative(native,0,0);setupNative(direct,0,0);
 GasRequestedIntervalControls<double> controls;if(limit==0)controls.maximumAttempts=1;else controls.minimumSubstep=.002;
 assert(advanceGasRequestedInterval<SharedGasTrialPolicy>(&native,.002,2.,controls)==0);
 assert(advanceGasTrial<SharedGasTrialPolicy>(&direct,.002,2.)==0);
 const double* a[]={native.rho,native.rhoUx,native.rhoUy,native.rhoUz,native.rhoE,native.gasSpecies.rho};
 const double* b[]={direct.rho,direct.rhoUx,direct.rhoUy,direct.rhoUz,direct.rhoE,direct.gasSpecies.rho};
 for(int k=0;k<6;++k)assert(std::memcmp(a[k],b[k],(k==5?4:2)*sizeof(double))==0);
 releaseSharedGasTrialStorage(native.gasTrial);releaseSharedGasTrialStorage(direct.gasTrial);releaseSharedGasSpecies(native.gasSpecies);releaseSharedGasSpecies(direct.gasSpecies);assert(allocations==0);
}}
'''
    build_run(tmp_path,'terminal',ROOT,body)






def test_native_sst_interval_matches_original_physical_fields(tmp_path, original_native_root):
    def sst_source(root):
        return source(root).replace('setupNative(s,0,0);trace.clear();', 'setupNative(s,0,3);trace.clear();')
    before = build_run(tmp_path, 'original_sst', original_native_root, sst_source(original_native_root))
    after = build_run(tmp_path, 'current_sst', ROOT, sst_source(ROOT))
    assert before == after
    assert sum(line.startswith('CASE ') for line in after.splitlines()) == 3
