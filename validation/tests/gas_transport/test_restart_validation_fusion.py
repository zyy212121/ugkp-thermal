from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import re
import subprocess
import pytest

ROOT=SOURCE_ROOT

def braced(source,start):
    begin=source.index('{',start);end=begin+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]

@pytest.mark.parametrize('app',['CHT','FSH','gasUGKP'])
def test_restart_upload_consumes_only_available_particle_storage(tmp_path,app):
    backend='gpu' if app=='CHT' else 'private_backend'
    source=(ROOT/f'applications/{app}/{backend}/GpuResidentStrict.cu').read_text()
    upload=braced(source,source.index('extern "C" int ugkwpGpuResidentStrictUploadParticleRestartMirror'))
    parameters=re.findall(r'const ([\w ]+)\* (\w+)',upload[:upload.index('{')])
    fields='\n'.join(f'{ty} {name}[192]{{}};' for ty,name in parameters)
    members={name for _,name in parameters}
    if 'pContactAge' not in members:fields+='\ndouble pContactAge[192]{};'
    upload=re.sub(r'<<<.*?>>>','',upload)
    arguments=','.join(name for _,name in parameters)
    declarations='\n'.join(f'{ty} data_{name}[192]{{}};const {ty}* {name}=data_{name};' for ty,name in parameters)
    value_checks='\n'.join(f'assert(std::memcmp(s.{name},data_{name},sizeof({ty})*3)==0);' for ty,name in parameters if name not in ['pTheta'])
    null_cases=[]
    for ty,name in parameters:
        null_cases.append('{const auto saved='+name+';'+name+'=nullptr;calls=0;assert(call(3)==1&&calls==0);'+name+'=saved;assert(call(3)==0);}')
    code=r"""
#include <cassert>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
using GpuReal=double;using GpuTime=double;using cudaError_t=int;
const int cudaSuccess=0;int calls=0,failCopy=0;std::string error;
namespace Foam {namespace gpuThermal {constexpr int coldWallAxialNodeCount=8,coldWallRadialRingCount=8,coldWall2DNodeCount=64,coldWall2DRadialNodeCount=8;}}
struct DeviceState {FIELDS
 int particleCapacity=3,count=0,wallCount=0,*particleCountDevice=&count,*wallBoundParticleCountDevice=&wallCount;
 int preBaseDirectoryReady=1,splitPreDirectoryActive=1,coldWallSolidificationEnabled=1,coldWall2DEnabled=1;
 bool particlesMayBePresent=false;DeviceState* deviceState=this;
};
DeviceState* asState(void* p){return static_cast<DeviceState*>(p);}
int validateState(DeviceState* p,const char*){return p?0:1;}
void setLastError(const char* p,int){error=p;}
void setLastErrorText(const char* p){error=p;}
int cudaMemset(void* p,int v,size_t n){++calls;std::memset(p,v,n);return 0;}
int cudaGetLastError(){return 0;}
template<class T>int copyToDevice(T* p,const T* v,size_t n,const char*){++calls;if(failCopy&&calls==failCopy)return 1;if(n>192||!v)return 1;std::memcpy(p,v,n*sizeof(T));return 0;}
void unpackParticleRestartTheta(size_t n,const unsigned char*,const double* v,double* t,double* age){for(size_t i=0;i<n;++i){t[i]=v[i];age[i]=0;}}
void rebuildWallBoundParticleDirectoryKernel(DeviceState*){}
int rebuildResidentParticleMomentsFromParticles(DeviceState*,int){++calls;return 0;}
UPLOAD
int main(){DeviceState s;DECLARATIONS
 auto call=[&](int n){return ugkwpGpuResidentStrictUploadParticleRestartMirror(&s,n,ARGUMENTS);};
 for(int n:{-1,4}){calls=0;const auto before=s;assert(call(n)==1&&calls==0&&s.count==before.count&&s.preBaseDirectoryReady==before.preBaseDirectoryReady);}
 NULL_CASES
 data_pT[0]=std::nan("");data_pm[1]=-1;data_px[2]=INFINITY;
 assert(call(3)==0&&s.count==3);VALUE_CHECKS
 calls=0;failCopy=8;assert(call(3)==1);failCopy=0;assert(call(3)==0);VALUE_CHECKS
 assert(call(0)==0&&s.count==0);
}
""".replace('FIELDS',fields).replace('UPLOAD',upload).replace('DECLARATIONS',declarations).replace('ARGUMENTS',arguments).replace('NULL_CASES','\n'.join(null_cases)).replace('VALUE_CHECKS',value_checks)
    path=tmp_path/'upload.cpp';path.write_text(code);binary=tmp_path/'upload'
    result=subprocess.run(['g++','-std=c++17','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer',str(path),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
