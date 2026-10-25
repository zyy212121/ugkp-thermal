"""Native CUDA compile gates for selected-wall source-number evaluation."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import os
import shutil
import subprocess
import pytest

ROOT=SOURCE_ROOT





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
}
'''
    path=tmp_path/'sst_sources.cu';path.write_text(source)
    run=subprocess.run([nvcc,'-std=c++17','-arch=sm_89',
        '-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gpu'),
        '-I'+str(ROOT/'common/gasNumerics'),'-I'+str((VALIDATION_ROOT / 'tests/gas_transport')),
        '-c',str(path),'-o',str(tmp_path/'sst_sources.o')],
        capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
