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
    source=source.replace('#include \"GpuGasTransportPreparation.cuh\"',(ROOT/'common/GpuGasTransportPreparation.cuh').read_text())
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
 State*deviceState=this;int fixedCellBlockThreads=4,fixedFaceBlockThreads=4,hostTurbulenceModel=0,hostGasFluxScheme=1;
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


def test_diffusive_courant_is_opt_in_for_all_wall_treatments(tmp_path):
    helper=launches((ROOT/'common/GpuGasCourant.cuh').read_text())
    compile_probe(tmp_path,fixture()+CUDA+helper+r'''
int main(){
    Host state;initialise(state);
    state.gasMu=Real(4000);
    const auto nut=state.nut;
    for(int treatment=0;treatment<3;++treatment){
        state.sstConfigured=treatment!=2;
        state.sstWallTreatment=treatment;
        state.nut=state.sstConfigured?nut:nullptr;
        if(state.nut)state.nut[0]=state.nut[1]=Real(10);
        GasCourantMaxima acoustic,diffusive;
        ck(computeGasCurrentCourant(&state,.001,0.,4,4,nullptr,acoustic,Maximum{})==0,"default Courant failed");
        ck(computeGasCurrentCourant(&state,.001,0.,4,4,nullptr,diffusive,Maximum{},true)==0,"diffusive Courant failed");
        ck(diffusive.maximum>acoustic.maximum,"diffusion switch ignored");
        bool onDevice=true;
        ck(computeGasCurrentCourantField(&state,.001,0.,4,4,nullptr,onDevice)==0,"configured switch failed");
        near(std::max(state.gasFluxPositivityScale[0],state.gasFluxPositivityScale[1]),Real(diffusive.maximum),"configured switch differs from host switch");
        onDevice=false;
        ck(computeGasCurrentCourantField(&state,.001,0.,4,4,nullptr,onDevice)==0,"configured off switch failed");
        near(std::max(state.gasFluxPositivityScale[0],state.gasFluxPositivityScale[1]),Real(acoustic.maximum),"configured off switch added diffusion");
    }
}
''')
