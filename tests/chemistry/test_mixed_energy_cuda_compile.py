"""Native CUDA type/instantiation checks; no runtime or memory-check claim."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
SOURCE=r'''
#include "common/operators/advanceGasChemistryKernel.cuh"
template<class Real,class Energy,int Ns>
struct State:ugkwp::GasStateView<Real,double,int,Energy>{ugkwp::GasSpeciesState<Real,Ns> gasSpecies;};
void instantiate(State<float,float,2>*f2,State<float,double,2>*m2,State<double,double,2>*d2,
                 State<float,float,10>*f10,State<float,double,10>*m10,State<double,double,10>*d10,
                 const float* vf,const double* vd){
 advanceGasChemistryKernel<<<1,32>>>(f2,1e-6,vf);advanceGasChemistryKernel<<<1,32>>>(m2,1e-6,vf);advanceGasChemistryKernel<<<1,32>>>(d2,1e-6,vd);
 advanceGasChemistryKernel<<<1,32>>>(f10,1e-6,vf);advanceGasChemistryKernel<<<1,32>>>(m10,1e-6,vf);advanceGasChemistryKernel<<<1,32>>>(d10,1e-6,vd);
}
'''


def test_native_cuda_mixed_energy_chemistry_instantiations(tmp_path):
    nvcc=shutil.which('nvcc')
    if not nvcc:pytest.skip('CUDA compiler unavailable; no native execution claimed')
    source=tmp_path/'mixed.cu';source.write_text(SOURCE)
    result=subprocess.run([nvcc,'-std=c++17','-O3','--fmad=false','-arch=sm_89','-Xptxas=-v','-I'+str(ROOT),'-c',str(source),'-o',str(tmp_path/'mixed.o')],capture_output=True,text=True)
    (tmp_path/'ptxas.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
