"""Common operator ownership and Poisson access/commit behavior."""
from pathlib import Path
import re, subprocess

ROOT=Path(__file__).resolve().parents[1]
ENTRIES=['applications/gasUGKP/private_backend/GpuResidentStrict.cu',
         'applications/FSH/private_backend/GpuResidentStrict.cu',
         'applications/CHT/gpu/GpuResidentStrict.cu']
OWNERS={
    'accumulateOnePoolParticle':'GpuCollisionPoolParticle.cuh',
    'accumulateCsrSplitLogicalPoolParticle':'GpuCollisionPoolSplitParticle.cuh',
    'accumulateParticleMomentsSegmentedKernel':'GpuParticleMoments.cuh',
    'publishParticleMomentsCell':'GpuParticleMoments.cuh',
    'accumulateCsrSegmentedMomentTasksPersistentKernel':'GpuSegmentedMomentWorkers.cuh',
    'runCsrPersistentQueue':'CsrPersistentQueue.cuh',
    'launchPostTransportMomentPipeline':'GpuMomentPipeline.cuh',
    'launchCommonPressureKick':'GpuPressurePipeline.cuh',
}

def closure(path,seen=None):
    seen=set() if seen is None else seen
    path=path.resolve()
    if path in seen or not path.is_file():return seen
    seen.add(path)
    for name in re.findall(r'^\s*#include\s+"([^"\n]+)"',path.read_text(),re.M):
        for directory in [path.parent,ROOT/'common',ROOT/'gpu/thermal']:
            candidate=directory/name
            if candidate.is_file():closure(candidate,seen);break
    return seen

def test_one_owner_reached_by_every_application():
    for entry in ENTRIES:
        if not (ROOT/entry).exists():continue
        files=closure(ROOT/entry)
        for name,owner in OWNERS.items():
            pattern=r'\b(?:void|int)\s+'+name+r'\s*\([^;{]*\)\s*\{'
            definitions=[]
            for path in files:
                code=re.sub(r'//[^\n]*|/\*[\s\S]*?\*/','',path.read_text())
                if re.search(pattern,code):definitions.append(path)
            assert definitions==[ROOT/'common'/owner],(entry,name,definitions)
        for path in files:
            assert 'GPU_POOL_S1_LATE_THETA_AND_RNG' not in path.read_text(),path

def test_default_poisson_access_order_and_final_state(tmp_path):
    source=ROOT/'tests/fixtures/shared_operators/uniform_pool_contract.cpp'
    binary=tmp_path/'uniform-pool'
    built=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert built.returncode==0,built.stdout+built.stderr
    ran=subprocess.run([str(binary)],capture_output=True,text=True)
    assert ran.returncode==0,ran.stdout+ran.stderr
