"""Native CUDA type/instantiation checks; no runtime or memory-check claim."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=SOURCE_ROOT
SOURCE=r'''
#include "common/operators/advanceGasChemistryKernel.cuh"
template<class Real,class Energy,int Ns>
struct State:ugkwp::GasStateView<Real,double,int,Energy>{ugkwp::GasSpeciesState<Real,Ns> gasSpecies;};
void instantiate(State<double,double,2>*d2,State<double,double,10>*d10,const double* vd){
 advanceGasChemistryKernel<<<1,32>>>(d2,1e-6,vd);
 advanceGasChemistryKernel<<<1,32>>>(d10,1e-6,vd);
}
'''


def test_native_cuda_double_chemistry_instantiations(tmp_path):
    nvcc=shutil.which('nvcc')
    if not nvcc:pytest.skip('CUDA compiler unavailable; no native execution claimed')
    source=tmp_path/'mixed.cu';source.write_text(SOURCE)
    result=subprocess.run([nvcc,'-std=c++17','-O3','--fmad=false','-arch=sm_89','-Xptxas=-v','-I'+str(ROOT),'-c',str(source),'-o',str(tmp_path/'mixed.o')],capture_output=True,text=True)
    (tmp_path/'ptxas.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
