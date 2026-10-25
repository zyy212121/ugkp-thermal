"""Execute actual extracted Courant kernels/reductions serially; not CUDA runtime."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import re
import subprocess
import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture

ROOT=SOURCE_ROOT

def launches(source):
    pattern=r'((?:\w+::)*\w+(?:<[^;{}]*?>)?)\s*<<<([\s\S]*?)>>>\s*\(([^;]*?)\);'
    def convert(m):
        grid,block,*_=m[2].split(',')
        return 'emulateLaunch([&](){'+m[1]+'('+m[3]+');},'+grid+','+block+');'
    return re.sub(pattern,convert,source)

CUDA=r'''
#include <algorithm>
#include <string>
#include <vector>
using cudaError_t=int;using cudaStream_t=void*;const int cudaSuccess=0;
int launches=0,failLaunch=0,reductions=0;std::string last;
int cudaGetLastError(){return launches==failLaunch?71:0;}
void setLastError(const char*p,int){last=p;}void setLastErrorText(const char*p){last=p;}
template<class F>void emulateLaunch(F f,int grid,int block){++launches;blockDim.x=block;for(blockIdx.x=0;blockIdx.x<grid;++blockIdx.x)for(threadIdx.x=0;threadIdx.x<block;++threadIdx.x)f();}
struct Host:State{
 State*deviceState=this;int fixedCellBlockThreads=4,fixedFaceBlockThreads=4,hostTurbulenceModel=0;
};
struct Maximum{
 template<class T>int operator()(const T*field,int n,cudaStream_t,double&out)const{
 ++reductions;out=-INFINITY;for(int i=0;i<n;++i){if(std::isnan(field[i])){out=NAN;break;}out=std::max(out,double(field[i]));}return 0;}
};
'''

@pytest.mark.parametrize('mode',['convective','diffusion','sst','zero','large'])
def test_complete_current_state_maxima_and_launch_errors(tmp_path,mode):
    helper=launches((ROOT/'common/GpuGasCourant.cuh').read_text())
    main=r'''
int main(){Host s;initialise(s);s.sstConfigured=MODE_SST;s.hostTurbulenceModel=MODE_SST?3:0;
s.gasMu=MODE_DIFFUSION?Real(1e5):Real(0);s.Ux[0]=s.Ux[1]=100;s.rhoUx[0]=s.rhoUx[1]=200;
for(int c=0;c<2;++c)s.rhoE[c]+=Real(10000);
const double dt=MODE_ZERO?0.:MODE_LARGE?1e16:1e-3;
GasCourantMaxima out;out.convective=out.maximum=-777;
ck(computeGasCurrentCourant(&s,dt,2.,4,4,nullptr,out,Maximum{})==0,"current-state estimator failed");
ck(out.convective==std::max(double(s.gasFluxPositivityScale[0]),double(s.gasFluxPositivityScale[1])),"convective maximum incomplete");
ck(out.maximum==out.convective,"acoustic result differs");
ck(reductions==1,"unexpected reduction pass");
if(MODE_ZERO)ck(out.maximum==0,"zero Co changed");if(MODE_LARGE)ck(out.maximum>.5*OfGreat,"large finite Co not exercised");
const double acoustic=out.maximum;s.gasMu=Real(1e200);s.omega[0]=s.omega[1]=Real(1e200);s.rhoOmega[0]=s.rhoOmega[1]=Real(1e200);
ck(computeGasCurrentCourant(&s,dt,2.,4,4,nullptr,out,Maximum{})==0&&out.maximum==acoustic,"diffusion/source or target CFL contaminated acoustic estimate");
GasCourantMaxima sentinel;sentinel.maximum=-888;failLaunch=launches+1;
ck(computeGasCurrentCourant(&s,dt,2.,4,4,nullptr,sentinel,Maximum{})!=0&&sentinel.maximum==-888,"real launch failure hidden");
}
'''
    for name in ['SST','DIFFUSION','ZERO','LARGE']:
        main=main.replace('MODE_'+name,'true' if mode==name.lower() else 'false')
    compile_probe(tmp_path,fixture()+CUDA+helper+main)
