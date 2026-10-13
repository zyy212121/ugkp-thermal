"""Publish actual reduced bounds, including nonfinite values, without a veto."""
from pathlib import Path
import re
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def test_actual_courant_helper_publishes_all_computed_maxima(tmp_path):
    source=(ROOT/'common/GpuGasCourant.cuh').read_text()
    kernels=sorted(set(re.findall(r'(\w+)\s*<<<',source)))
    source=re.sub(r'<<<[\s\S]*?>>>','',source)
    shim=r'''
#include <cmath>
#include <cassert>
#include <limits>
using cudaStream_t=void*;using cudaError_t=int;
constexpr int cudaSuccess=0;int launchFailure=0;
int cudaGetLastError(){return launchFailure;}
void setLastError(const char*,int){}void setLastErrorText(const char*){}
struct State{int nCells=1,nFaces=0,hostTurbulenceModel=0;void*deviceState=nullptr;double *gasFluxPositivityScale=nullptr,*gasDiffusionNumber=nullptr,*sstSourceNumber=nullptr;};
'''
    shim+='\n'.join(f'template<class... T> void {k}(T...){{}}'for k in kernels)
    code=shim+'\n'+source+r'''
int main(){
 State state;
 const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
 for(double a:{0.,.2,2.,nan,inf,-inf})for(double b:{0.,.4,4.,nan,inf,-inf}){
  int calls=0;GasCourantMaxima result;result.maximum=-777;
  auto reduce=[&](const double*,int,void*,double& value){value=calls++==0?a:b;return 0;};
  assert(computeGasCurrentCourant(&state,.1,.4,0.,32,32,nullptr,result,reduce)==0&&calls==2);
  assert((std::isnan(a)&&std::isnan(result.convective))||a==result.convective);
  assert((std::isnan(b)&&std::isnan(result.diffusion))||b==result.diffusion);
  const double expected=std::fmax(a,b);
  assert((std::isnan(expected)&&std::isnan(result.maximum))||expected==result.maximum);
 }
 GasCourantMaxima result;result.maximum=-777;
 auto fail=[](const double*,int,void*,double&){return 1;};
 assert(computeGasCurrentCourant(&state,.1,.4,0.,32,32,nullptr,result,fail)!=0&&result.maximum==-777);
 launchFailure=2;int calls=0;
 auto reduce=[&](const double*,int,void*,double&){++calls;return 0;};
 assert(computeGasCurrentCourant(&state,.1,.4,0.,32,32,nullptr,result,reduce)!=0&&calls==0&&result.maximum==-777);
}
'''
    src=tmp_path/'publication.cpp';src.write_text('#include <initializer_list>\n'+code)
    binary=tmp_path/'publication'
    subprocess.run(['g++','-std=c++17','-O2',str(src),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
