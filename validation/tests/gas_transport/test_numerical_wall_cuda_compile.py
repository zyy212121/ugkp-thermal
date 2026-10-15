"""Native CUDA compile gates for selected-wall source-number evaluation."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import os
import shutil
import subprocess
import pytest

ROOT=SOURCE_ROOT

def test_source_number_compiles_as_native_device_function(tmp_path):
    nvcc=os.environ.get('NVCC') or shutil.which('nvcc')
    if not nvcc:
        pytest.skip('CUDA compiler not installed; this test never claims CUDA runtime')
    source=(ROOT/'common/operators/computeSstFaceFluxKernel.cuh').read_text()
    start=source.index('template<class GasState>\n__device__ GPU_OPERATOR_REAL sstKSourceNumber')
    end=source.index('template<class GasState>\n__global__ void applySstFluxAndSourceKernel',start)
    real='double'
    pre=r'''
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
using R=REAL;
#define GPU_OPERATOR_REAL R
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) static_cast<R>(x)
constexpr R OfGreat=R(1e30);
struct State{R *rhoK,*rho,sstKMin;int selected;};
namespace ugkwp {__device__ int gasBoundaryLayerOwnerSlot(const State&s,int){return s.selected;}}
__device__ R clampMin(R a,R b){return fmax(a,b);}
'''.replace('using R=REAL;',f'using R={real};')
    path=tmp_path/'source_number.cu';path.write_text(pre+source[start:end]+'\n__global__ void probe(State*s,R*out){*out=sstKSourceNumber(*s,0,*out,1.);}\n')
    run=subprocess.run([nvcc,'-std=c++17','-arch=sm_89','-c',str(path),'-o',str(tmp_path/'source_number.o')],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr


def test_sst_source_callers_retain_double_time_products(tmp_path):
    from test_mixture_state import compile_probe
    from test_legacy_ale_transport import fixture
    compile_probe(tmp_path,fixture()+r'''
int main(){
 for(int channel=0;channel<2;++channel){
  State s;initialise(s);s.sstConfigured=1;s.cellPlaneCount[0]=0;
  s.rhoMin=Real(1e-30);s.rho[0]=Real(1e-20);s.k[0]=Real(.1);
  s.omega[0]=Real(1e15);s.rhoK[0]=s.rho[0]*s.k[0];
  s.rhoOmega[0]=s.rho[0]*s.omega[0];s.sstF1[0]=Real(.5);
  if(channel==0)s.sstCoefficients.beta1=s.sstCoefficients.beta2=Real(0);
  else s.sstCoefficients.betaStar=Real(0);
  const double dt=1e-50; // Not representable as float; the resulting numbers are.
  Real sk,so;sstSourcesForCell(s,0,Real(0),sk,so);
  const Real expected=Real(std::max(std::abs(dt*double(sk))/double(s.rhoK[0]),
      std::abs(dt*double(so))/double(s.rhoOmega[0])));
  ck(std::isfinite(expected)&&expected>Real(0),"source-time precision fixture lost its finite product");
  computeSstStabilityNumberKernel(&s,dt,Real(.5));
  ck(std::abs(s.sstSourceNumber[0]/(Real(2)*expected)-Real(1))
      <Real(8)*std::numeric_limits<Real>::epsilon(),"stability source-time product narrowed");
  applySstFluxAndSourceKernel(&s,dt);
  ck(std::abs(s.sstSourceNumber[0]/expected-Real(1))
      <Real(8)*std::numeric_limits<Real>::epsilon(),"update source-time product narrowed");
 }
}
''')


def test_full_sst_source_kernels_compile_with_double_time(tmp_path):
    """Instantiate both production callers, not only the scalar helper.

    In FP32, the stored k number and the double-time omega expression used to
    select a host-only mixed fmax overload. No relaxed-constexpr flag is used.
    """
    nvcc=os.environ.get('NVCC') or shutil.which('nvcc')
    if not nvcc:
        pytest.skip('CUDA compiler not installed; this test never claims CUDA runtime')
    probe=((VALIDATION_ROOT / 'tests/gas_transport/gas_state_probe.cpp')).read_text()
    prefix=probe.split('void hashBytes',1)[0]
    # Use the shared fixture's actual operator includes with native CUDA
    # annotations and launch indices, rather than its serial host adapters.
    host_adapters=('#define __host__','#define __global__','#define __device__',
                   '#define __forceinline__','#define asm(', 'struct ThreadIndex ')
    prefix='\n'.join(line for line in prefix.splitlines()
                     if not line.startswith(host_adapters))
    source='#include <cuda_runtime.h>\n'+prefix+r'''
using NativeSstState=ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients,double>;
void launchSstSourceChecks(NativeSstState* state,Time dt)
{
    applySstFluxAndSourceKernel<<<1,32>>>(state,dt);
    computeSstStabilityNumberKernel<<<1,32>>>(state,dt,Real(.5));
}
'''
    path=tmp_path/'sst_sources.cu';path.write_text(source)
    run=subprocess.run([nvcc,'-std=c++17','-arch=sm_89',
        '-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gpu'),
        '-I'+str(ROOT/'common/gasNumerics'),'-I'+str((VALIDATION_ROOT / 'tests/gas_transport')),
        '-c',str(path),'-o',str(tmp_path/'sst_sources.o')],
        capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
