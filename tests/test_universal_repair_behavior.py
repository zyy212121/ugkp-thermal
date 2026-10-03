"""Behavioral regression for demand-driven probabilities and shared scheduling."""
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
def function(text,name):
    import re
    m=re.search(r'^(?:__\w+__\s+)*(?:inline\s+)?(?:void|int|double) '+name+r'\s*\(',text,re.M)
    assert m,name
    start=m.start(); end=text.index('{',start)+1; depth=1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}'); end+=1
    return text[start:end]

def test_clear_computes_probability_only_for_consuming_heavy_cells(tmp_path):
    source=(ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    helper=(ROOT/'common/GpuCollisionProbability.cuh').read_text() if (ROOT/'common/GpuCollisionProbability.cuh').exists() else function(source,'poissonCollisionProbabilityForCell')
    clear=function(source,'clearPoissonThermalPoolKernel')
    # Compile the actual production body, with an observable tau accessor.
    # A return to all-cell preparation makes the explicit call-count checks fail.
    if 'CacheHindrance' in clear: clear='template<bool CacheHindrance=false>\n'+clear
    text=r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#define __device__
#define __global__
#define __forceinline__ inline
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) (x)
struct Dim{int x=0;} blockIdx,threadIdx,blockDim;
template<class T>T clampRange(T x,T a,T b){return std::min(std::max(x,a),b);}
const double OfGreat=1e30,OfSmall=1e-15;
struct DeviceState {
 int nCells=7,csrHeavyReductionEnabled=1,particleCapacity=1,particleCount=1;
 int* particleCountDevice=&particleCount;
 int csrCellTaskCount[7]={0,1,3,0,2,1,0},calls[7]={};
 int poolThermalCount[7]={},poissonPoolSampleTargetCount[7]={};
 double poissonCellCollisionProbability[7]={-99,-99,-99,-99,-99,-99,-99};
 double poolThermalSumUx[7]={},poolThermalSumUy[7]={},poolThermalSumUz[7]={},poolThermalSumU2[7]={};
 double poissonPoolMass[7]={},poissonPoolMomX[7]={},poissonPoolMomY[7]={},poissonPoolMomZ[7]={};
 double poissonPoolEnergy[7]={},poissonPoolDiameter[7]={},poissonPoolDiameter2[7]={},dragHindranceCache[7]={};
 double tau=2;
};
double granularCollisionTauFromCellDevice(DeviceState& s,int c){++s.calls[c];return s.tau;}
double solidEpsFromMomentDevice(DeviceState&,int){return 0;}
"""+helper+'\n'+clear+r"""
int main(){
 DeviceState s;blockDim.x=1;
 for(int c=0;c<7;++c){blockIdx.x=c;clearPoissonThermalPoolKernel(&s,0.2);}
 for(int c=0;c<7;++c){
  assert(s.calls[c]==((c==2||c==4)?1:0));
  if(c==2||c==4)assert(std::abs(s.poissonCellCollisionProbability[c]-(1-std::exp(-0.1)))<1e-15);
  else assert(s.poissonCellCollisionProbability[c]==-99);
 }
 // A reused directory must not reuse a previous time/state's probability.
 s.tau=4;
 for(int c=0;c<7;++c){blockIdx.x=c;clearPoissonThermalPoolKernel(&s,0.8);}
 assert(std::abs(s.poissonCellCollisionProbability[2]-(1-std::exp(-0.2)))<1e-15);
 s.csrHeavyReductionEnabled=0;
 for(int c=0;c<7;++c){blockIdx.x=c;clearPoissonThermalPoolKernel(&s,1.0);}
 for(int c=0;c<7;++c)assert(s.calls[c]==((c==2||c==4)?2:0));
}
"""
    cpp=tmp_path/'probability.cpp';cpp.write_text(text);exe=tmp_path/'probability'
    result=subprocess.run(['g++','-std=c++17','-O2',str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
