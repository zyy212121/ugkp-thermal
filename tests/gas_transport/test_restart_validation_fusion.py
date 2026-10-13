"""The existing upload validation scan also validates authoritative carry."""
from pathlib import Path
import json
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def braced(source, start):
    begin = source.index('{', start)
    depth, end = 1, begin + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def test_carry_check_shares_actual_upload_scan_and_precedes_array_copy(tmp_path):
    source = (ROOT/'applications/CHT/gpu/GpuResidentStrict.cu').read_text()
    start = source.index('extern "C" int ugkwpGpuResidentStrictUploadParticleRestartMirror')
    upload = braced(source, start)
    start = upload.index('for (int i = 0; i < nParticles; ++i)')
    loop = braced(upload, start)
    assert 'std::isfinite(pThermalEnthalpyResidual[i])' in loop, 'carry has a redundant particle scan'
    assert 'for (size_t i = 0; i < n; ++i)' not in upload
    assert start + len(loop) < upload.index('copyToDevice(s->px')
    fields = json.loads((ROOT/'common/ParticleFieldManifest.json').read_text())['fields']
    arrays = '\n'.join(f"{field['host']['CHT']} {field['name']}[{3*field['width']}]={{}};"
                       for field in fields if 'CHT' in field['host'])
    code = r'''
#include <cmath>
#include <limits>
#include <string>
using GpuReal=REAL;
using GpuTime=double;
#define GPU_R(x) static_cast<GpuReal>(x)
namespace Foam { namespace gpuThermal {
constexpr int particleWallTransientDeposit=3,particleWallDeposited=1;
constexpr int particleWallTransientRebound=2,coldWallAxialNodeCount=8;
constexpr int coldWallRadialRingCount=8,coldWall2DNodeCount=64,coldWall2DRadialNodeCount=8;
}}
std::string error;
void setLastErrorText(const char* text){error=text;}
int copies=0;
int validate(int badIndex,double badCarry,bool badMass){
 const int nParticles=3;
 struct State {int nFaces=1;} state;State* s=&state;
 ARRAYS
 for(int i=0;i<nParticles;++i){pm[i]=1;pStuckFaceId[i]=-1;pThermalEnthalpyResidual[i]=0.0003;}
 if(badIndex>=0)pThermalEnthalpyResidual[badIndex]=badCarry;
 if(badMass)pm[2]=0;
 LOOP
 ++copies;return 0;
}
int main(){
 if(validate(-1,0,false)||copies!=1)return 1;
 for(int i=0;i<3;++i)for(double x:{std::nan(""),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}){
  error.clear();if(validate(i,x,false)!=1||copies!=1||error!="particle restart thermal enthalpy residual is nonfinite")return 2;
 }
 error.clear();if(validate(-1,0,true)!=1||copies!=1||error!="particle restart stuck state/face/area is invalid")return 3;
}
'''.replace('REAL', 'double').replace('ARRAYS', arrays).replace('LOOP', loop)
    path=tmp_path/'probe.cpp';path.write_text(code)
    binary=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-O2',str(path),'-o',str(binary)],check=True,capture_output=True,text=True)
    subprocess.run([str(binary)],check=True)
