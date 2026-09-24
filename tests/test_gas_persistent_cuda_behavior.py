"""Run the gas directory adapter and shared queue on CUDA with exact sums.

The accumulation stand-ins supply independently known task results. Production
task mapping, shared scheduling, finalizers, and launch/reset helpers are real.
"""
from pathlib import Path
import re, subprocess, shutil, os
import pytest
ROOT=Path(__file__).resolve().parents[1]

def item(s,name):
    m=re.search(r'^(?:__\w+__\s+)*(?:void|int|double|struct|cudaError_t) '+name+r'\b',s,re.M)
    assert m,name
    a=m.start();b=s.index('{',a)+1;d=1
    while d:d+=(s[b]=='{')-(s[b]=='}');b+=1
    return s[a:b]+(';' if s[a:].startswith('struct ') else '')

def test_gas_nonempty_queue_reuse_and_exact_coverage(tmp_path):
    nvcc=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    if not Path(nvcc).exists():pytest.skip('CUDA compiler required')
    source=(ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    fixture=(ROOT/'tests/fixtures/persistent_queue/gas_prefix.cu.in').read_text()
    pieces=[]
    names=[('csrReductionTileParticles',''),('countCsrReductionTasksKernel',''),('writeCsrReductionTask',''),('materializeCsrReductionTasksKernel',''),
           ('preparePoissonPoolSamplingCell',''),('poissonCollisionProbabilityForCell',''),
           ('CsrPoolOperation','template<bool PoissonMode, HeavyDirectoryKind DirectoryKind>'),
           ('accumulateCsrSegmentedPoolTasksPersistentKernel','template<bool PoissonMode, HeavyDirectoryKind DirectoryKind=HeavyDirectoryKind::full>'),
           ('CsrPoolFinalizeOperation',''),('finalizeCsrSegmentedPoolCellsKernel',''),
           ('launchCsrSegmentedPoolReduction',''),('CsrMomentOperation','template<bool GatherSurvivors=false>'),
           ('accumulateCsrSegmentedMomentTasksPersistentKernel','template<bool GatherSurvivors=false>'),('CsrMomentFinalizeOperation',''),
           ('finalizeCsrSegmentedMomentCellsKernel',''),('launchCsrSegmentedMomentReduction','')]
    for name,prefix in names:pieces.append(prefix+'\n'+item(source,name))
    text=fixture+'\n#include "CsrPersistentQueue.cuh"\n'+'\n'.join(pieces)
    text+=(ROOT/'tests/fixtures/persistent_queue/gas_main.cu.in').read_text()
    cpp=tmp_path/'gas_queue.cu';exe=tmp_path/'gas_queue';cpp.write_text(text)
    run=subprocess.run([nvcc,'-std=c++17','-O3','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'-I'+str(ROOT/'common'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=90)
    assert run.returncode==0,run.stdout+run.stderr
