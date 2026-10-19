"""Boundary-local thread indices must reproduce the full-face operator exactly."""
import re
import subprocess
from validation.tools.paths import SOURCE_ROOT

def test_boundary_local_indices_preserve_all_three_operators(tmp_path):
    mirror=(SOURCE_ROOT/'common/operators/updateLegacyGasBoundaryMirrorKernel.cuh').read_text().split('#include "reconstructGasMixtureFace.cuh"')[0]
    wave=(SOURCE_ROOT/'common/operators/updateWaveTransmissivePressureBoundaryKernel.cuh').read_text()
    current=mirror+wave
    # Reference index contract only; all boundary arithmetic is the real header.
    reference=current.replace('s.nInternalFaces + blockIdx.x*blockDim.x + threadIdx.x','blockIdx.x*blockDim.x + threadIdx.x')
    reference=reference.replace('if (f >= s.nFaces','if (f < s.nInternalFaces || f >= s.nFaces')
    names=['updateLegacyGasBoundaryMirrorKernel','updateRiemannBoundaryMirrorKernel','updateWaveTransmissivePressureBoundaryKernel']
    for name in names:reference=reference.replace(name,'reference_'+name)
    scalar={'nInternalFaces','nFaces','nCells','scheduledInletTemperature','TgasMin','rhoMin','Rgas'}
    fields=sorted(set(re.findall(r's\.(\w+)',current))-scalar)
    integers={x for x in fields if x.endswith(('Fix','Kind')) or x in {'faceOwner','gasBoundaryPWave','riemannBoundaryPWave'}}
    declarations='\n'.join(('int' if x in integers else 'double')+' '+x+'[101]{};' for x in fields)
    initialization='\n'.join('a.'+x+'[i]='+ ('(i+trial)%6' if x.endswith('Kind') else '(i+trial)%3' if x.endswith('Fix') else 'i%7' if x=='faceOwner' else 'i%2' if x in integers else '1.+.01*i')+';' for x in fields)
    cpp=tmp_path/'boundary.cpp'
    cpp.write_text(r"""
#include <cmath>
#include <cstring>
#include <cassert>
#define __global__
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) (x)
#define GPU_OPERATOR_TINY(x) (x)
struct Index{int x=0;}blockIdx,blockDim,threadIdx;
constexpr double OfSmall=1e-15,OfVSmall=1e-300;
double clampMin(double a,double b){return a<b?b:a;}
double finiteOr(double a,double b){return std::isfinite(a)?a:b;}
struct State{int nInternalFaces=0,nFaces=0,nCells=7;double scheduledInletTemperature=300,TgasMin=1,rhoMin=1e-9,Rgas=287;
"""+declarations+r"""};
bool scheduledInletFaceDevice(const State&,int f){return f%11==0;}
double scheduledPressureDevice(const State&,double t){return 100000+t;}
int coupledFaceNeighbour(const State&,int f){return (f+1)%7;}
"""+reference+current+r"""
int main(){blockDim.x=1;
 for(int n:{0,1,17,101})for(int internal:{0,n/2,n})for(int trial=0;trial<6;++trial){
  State a;a.nFaces=n;a.nInternalFaces=internal;for(int i=0;i<101;++i){
"""+initialization+r"""}
  State b=a;
  for(int f=0;f<n;++f){threadIdx.x=f;reference_updateLegacyGasBoundaryMirrorKernel(&a,.3);}
  for(int f=0;f<n-internal;++f){threadIdx.x=f;updateLegacyGasBoundaryMirrorKernel(&b,.3);}
  assert(std::memcmp(&a,&b,sizeof(a))==0);
  for(int f=0;f<n;++f){threadIdx.x=f;reference_updateRiemannBoundaryMirrorKernel(&a);}
  for(int f=0;f<n-internal;++f){threadIdx.x=f;updateRiemannBoundaryMirrorKernel(&b);}
  assert(std::memcmp(&a,&b,sizeof(a))==0);
  for(int f=0;f<n;++f){threadIdx.x=f;reference_updateWaveTransmissivePressureBoundaryKernel(&a,.01);}
  for(int f=0;f<n-internal;++f){threadIdx.x=f;updateWaveTransmissivePressureBoundaryKernel(&b,.01);}
  assert(std::memcmp(&a,&b,sizeof(a))==0);
 }
}
""")
    # initializer_list is required by the range-based test cases.
    cpp.write_text('#include <initializer_list>\n'+cpp.read_text())
    exe=tmp_path/'boundary'
    q=subprocess.run(['g++','-std=c++17','-O2',str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
    q=subprocess.run([str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
