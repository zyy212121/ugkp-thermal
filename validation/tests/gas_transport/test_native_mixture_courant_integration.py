"""Actual configured mixture helper/native wrapper and required-error consumer."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import re
import subprocess
from native_gas_integration_fixture import ROOT,native_fixture,MODEL_SETUP,COMPARE_SPECIES,function

def tagged_launches(source):
    pattern=r'([\w:]+(?:<[^;{}]*?>)?)\s*<<<([\s\S]*?)>>>\s*\(([^;]*?)\);'
    def convert(m):
        grid,block,*_=m[2].split(',')
        return 'trace.push_back("'+m[1]+'");emulateLaunch([&](){'+m[1]+'('+m[3]+');},'+grid+','+block+');'
    return re.sub(pattern,convert,source)

def courant_probe(reference=''):
    native=(ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    code=native_fixture().replace('SharedGasTrialStorage gasTrial;', 'struct:SharedGasTrialStorage{double targetMaxCo=0;}gasTrial;')
    code=code.replace('template<class T> int copyToHost', 'int hostReads=0;\ntemplate<class T> int copyToHost').replace('if(hostCopyFailAfter==0)', '++hostReads;if(hostCopyFailAfter==0)')
    code+=r'''
int reductionCount=0;
namespace thrust{
template<class T>using device_ptr=T*;int device=0;
template<class T>T*device_pointer_cast(T*p){return p;}
namespace cuda{struct Par{int on(cudaStream_t){return 0;}}par;}
template<class T>T*max_element(int,T*a,T*b){++reductionCount;return std::max_element(a,b);}
template<class T,class F>T reduce(int,const T*a,const T*b,T value,F f){++reductionCount;for(;a!=b;++a)value=f(value,*a);return value;}
}
'''
    code+=function(native,'struct GasCourantMaximum')+';\n'
    code+=tagged_launches((ROOT/'common/GpuGasCourant.cuh').read_text())
    code+=function(native,'extern "C" int ugkwpGpuResidentStrictComputeGasCourant')
    code+=tagged_launches(reference)+MODEL_SETUP+COMPARE_SPECIES
    fields=re.findall(r'    Real\* (\w+)\{\};',(VALIDATION_ROOT / 'tests/gas_transport/legacy_gas_fixture.hpp').read_text())
    comparisons='\n'.join(f'assert(std::memcmp(a.{f},b.{f},8*sizeof(double))==0&&std::memcmp(a.{f},c.{f},8*sizeof(double))==0);'for f in fields)
    code+=r'''
int helper(DeviceState*s,double dt,double&out){GasCourantMaxima result;auto reduce=[](const double*f,int n,cudaStream_t,double&v){++reductionCount;v=-INFINITY;for(int i=0;i<n;++i)v=GasCourantMaximum{}(v,f[i]);return 0;};int rc=computeGasCurrentCourant(s,dt,.5,2.,4,4,nullptr,result,reduce);if(!rc)out=result.maximum;return rc;}
int main(){int cases=0;
for(int moving:{0,1})for(int turbulence:{0,3})for(int visc:{0,1})for(int zero:{0,1}){
 DeviceState a,b,c;setupNative(a,0,turbulence);setupNative(b,0,turbulence);setupNative(c,0,turbulence);
 Real oldVolume[2]={1,1},newVolume[2]={1,1},sweep[3]={.001,0,0};
 for(auto*s:{&a,&b,&c}){s->gasMu=visc?1e5:0;if(!visc){std::free(const_cast<double*>(s->gasSpecies.diffusivity));--allocations;s->gasSpecies.diffusivity=nullptr;}s->gasGeometry.enabled=moving;s->gasGeometry.oldVolume=oldVolume;s->gasGeometry.newVolume=newVolume;s->gasGeometry.faceSweptVolume=sweep;s->gasGeometry.interval=.001;if(zero){s->nFaces=0;for(int i=0;i<2;++i)s->cellPlaneCount[i]=0;}}
 double old=-777,direct=-777,current=-777;trace.clear();reductionCount=0;
 REFERENCE_CALL
 const auto expectedTrace=trace;assert(reductionCount==(turbulence==3?3:2));
 trace.clear();reductionCount=0;assert(helper(&b,.001,direct)==0);assert(trace==expectedTrace&&reductionCount==(turbulence==3?3:2));
 trace.clear();reductionCount=0;assert(ugkwpGpuResidentStrictComputeGasCourant(&c,.001,.5,2.,&current)==0);assert(trace==expectedTrace&&reductionCount==(turbulence==3?3:2));
 assert(std::memcmp(&old,&direct,sizeof old)==0&&std::memcmp(&old,&current,sizeof old)==0);
 compareSpeciesFields(a,b);compareSpeciesFields(a,c);
 COMPARE_ALL_WRITTEN_FIELDS
 auto al=sharedGasTrialDetail::cellFields(&a),bl=sharedGasTrialDetail::cellFields(&b),cl=sharedGasTrialDetail::cellFields(&c);
 for(std::size_t k=0;k<al.size();++k)if(al[k].pointer)assert(std::memcmp(al[k].pointer,bl[k].pointer,al[k].count*sizeof(double))==0&&std::memcmp(al[k].pointer,cl[k].pointer,al[k].count*sizeof(double))==0);
 // All status arrays remain complete; all three face slots are checked even in zero-incidence cases.
 for(int i=0;i<2;++i)assert(a.gasSpecies.cellStatus[i]==0&&b.gasSpecies.cellStatus[i]==0&&c.gasSpecies.cellStatus[i]==0);
 for(int i=0;i<3;++i)assert(a.gasSpecies.faceStatus[i]==0&&b.gasSpecies.faceStatus[i]==0&&c.gasSpecies.faceStatus[i]==0);
 std::printf("moving=%d turbulence=%d diffusion=%d zeroIncidence=%d maximum=%.17g gasArrays=%d speciesArrays=12 BIT_IDENTICAL\n",moving,turbulence,visc,zero,current,FIELD_COUNT);++cases;
 releaseSharedGasSpecies(a.gasSpecies);releaseSharedGasSpecies(b.gasSpecies);releaseSharedGasSpecies(c.gasSpecies);assert(allocations==0);
}
DeviceState s;setupNative(s,0,0);double result=-777;assert(ugkwpGpuResidentStrictComputeGasCourant(&s,.001,.5,2.,&result)==0);
// Recovery reports a real missing table while old finite caches remain readable.
// The real native policy must read the status and refuse to publish that Co.
auto*coeff=s.gasSpecies.thermo.coefficients;s.gasSpecies.thermo.coefficients=nullptr;hostReads=0;result=-777;
assert(ugkwpGpuResidentStrictComputeGasCourant(&s,.001,.5,2.,&result)!=0&&result==-777&&s.gasSpecies.cellStatus[0]!=0&&hostReads==2);
s.gasSpecies.thermo.coefficients=coeff;assert(ugkwpGpuResidentStrictComputeGasCourant(&s,.001,.5,2.,&result)==0);
// Actual absent consumed moving-face backing and launch failures keep output unpublished.
s.gasGeometry.enabled=true;s.gasGeometry.faceSweptVolume=nullptr;result=-777;assert(ugkwpGpuResidentStrictComputeGasCourant(&s,.001,.5,2.,&result)!=0&&result==-777&&s.gasSpecies.faceStatus[2]!=0);s.gasGeometry.enabled=false;
failLaunch=launchCount+1;assert(ugkwpGpuResidentStrictComputeGasCourant(&s,.001,.5,2.,&result)!=0&&result==-777);failLaunch=0;
assert(ugkwpGpuResidentStrictComputeGasCourant(&s,1e16,.5,2.,&result)==0&&result>.5*OfGreat);
releaseSharedGasSpecies(s.gasSpecies);assert(allocations==0&&cases==16);
}
'''.replace('REFERENCE_CALL','assert(referenceCourant(&a,.001,.5,2.,&old)==0);'if reference else'assert(helper(&a,.001,old)==0);').replace('COMPARE_ALL_WRITTEN_FIELDS',comparisons).replace('FIELD_COUNT',str(len(fields)))
    return code

def build_run(out,reference=''):
    source=out/'courant.cpp';source.write_text(courant_probe(reference));binary=out/'courant'
    cmd=['g++','-std=c++17','-O0','-I'+str(ROOT),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),'-I'+str(VALIDATION_ROOT / 'tests/gas_transport'),str(source),'-o',str(binary)]
    b=subprocess.run(cmd,capture_output=True,text=True);assert b.returncode==0,b.stderr
    r=subprocess.run([str(binary)],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
    return r.stdout

def test_actual_native_mixture_courant_complete_outputs_and_errors(tmp_path):
    assert len(build_run(tmp_path).splitlines())==16
