"""Exact original successful particle calculations; actual revision-correct bodies.

Serial host CUDA shims do not certify GPU runtime, compaction, or timing.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import io
import os
import subprocess
import tarfile

import pytest

ROOT = SOURCE_ROOT
ORIGINAL = 'd6110c02745a8ad726208a426c4bb3a28233c1fd'
CURRENT_ROOT = Path(os.environ.get('PARTICLE_CURRENT_ROOT', ROOT))
FIXTURES = (VALIDATION_ROOT / 'tests/fixtures/particle_compatibility')


@pytest.fixture(scope='module')
def original_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('original-particle-tree')
    archive = subprocess.check_output(['git', 'archive', ORIGINAL, 'common', 'applications/CHT/gpu/GpuResidentStrict.cu', 'applications/CHT/gpu/GpuParticleRadiationMath.H'], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(root, filter='data')
    return root


def build_run(folder, label, root, body):
    src = folder / (label + '.cpp')
    exe = folder / label
    src.write_text(body)
    cmd = ['g++', '-std=c++17', '-O2', '-ffp-contract=off',
           '-I' + str(root / 'common'),
           '-I' + str(root / 'common/wall'), '-I' + str(root), str(src), '-o', str(exe)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    (folder / (label + '_build.log')).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    (folder / (label + '.log')).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize('fixture', ['thermal_readers', 'cold1d', 'facewalk', 'facewalk_periodic', 'affine'])
def test_original_particle_success_exact(tmp_path, original_root, fixture):
    body = (FIXTURES / (fixture + '.cpp')).read_text()
    before = build_run(tmp_path, 'original', original_root, body)
    after = build_run(tmp_path, 'corrected', CURRENT_ROOT, body)
    assert after == before


def pressure_source(root):
    text = (root / 'common/GpuPressureKickAccumulation.cuh').read_text()
    start = text.index('__device__ inline PressureReal pressureLocalConvexScale')
    end = text.index('template<class Scratch>', start)
    return ((VALIDATION_ROOT / 'tests/fixtures/particle_compatibility/pressure_prefix.cpp')).read_text() + text[start:end] + ((VALIDATION_ROOT / 'tests/fixtures/particle_compatibility/pressure_main.cpp')).read_text()


def test_original_pressure_fallback_exact(tmp_path, original_root):
    before = build_run(tmp_path, 'original', original_root, pressure_source(original_root))
    after = build_run(tmp_path, 'corrected', CURRENT_ROOT, pressure_source(CURRENT_ROOT))
    assert after == before


def test_thermal_terminal_extensions_still_admitted(tmp_path, original_root):
    body = ((VALIDATION_ROOT / 'tests/fixtures/particle_compatibility/extensions.cpp')).read_text()
    before = build_run(tmp_path, 'original', original_root, body)
    after = build_run(tmp_path, 'corrected', CURRENT_ROOT, body)
    assert before == '0 0 0\n'
    assert after == '1 1 1\n'


def radiation_source(root):
    text = (root / 'applications/CHT/gpu/GpuResidentStrict.cu').read_text()
    def block(name):
        start = text.index(name)
        brace = text.index('{', start)
        depth, end = 1, brace + 1
        while depth:
            depth += (text[end] == '{') - (text[end] == '}')
            end += 1
        return text[start:end]
    prefix = ((VALIDATION_ROOT / 'tests/fixtures/particle_compatibility/thermal_readers.cpp')).read_text().split('int main()', 1)[0]
    prefix = prefix.replace('struct DeviceState {', 'struct DeviceState { int nCells=1,particleCapacity=1,count=1; int* particleCountDevice=&count; int pStatus[1]={1},pCellId[1]={0}; unsigned long long pOrigId[1]={71}; GpuReal radiationMobileMass[1]={},radiationMobileTemperatureMass[1]={},radiationMobileDiameterMass[1]={};')
    prefix += r'''
#include "applications/CHT/gpu/GpuParticleRadiationMath.H"
#include "GpuParticleThermalConsumers.H"
#define __global__
struct Dim {int x=1;}; Dim gridDim,blockDim; struct Index {int x=0;}; Index blockIdx,threadIdx;
using std::isfinite;
int atomicCAS(int* p,int old,int value){int saved=*p;if(saved==old)*p=value;return saved;}
template<class T>void atomicAdd(T* p,T value){*p+=value;}
'''
    kernels = ['applyParticleRadiationAffineTemperatureKernel', 'clearMobileParticleRadiationSumsKernel', 'accumulateMobileParticleRadiationSumsAtomicKernel']
    if 'struct ParticleRadiationValidationError' in text:
        prefix += '#define ORIGINAL_RADIATION_PREFLIGHT\n'
        prefix += block('struct ParticleRadiationValidationError') + ';\n' + block('enum ParticleRadiationValidationCode') + ';\n'
        kernels.insert(0, 'validateParticleRadiationAffineTemperatureKernel')
    source = prefix + '\n'.join(block('__global__ void ' + name) for name in kernels) + ((VALIDATION_ROOT / 'tests/fixtures/particle_compatibility/radiation_main.cpp')).read_text()
    return source.replace('asm("trap;");', 'std::abort();')


def test_native_radiation_preflight_apply_and_atomic_snapshot_exact(tmp_path, original_root):
    before = build_run(tmp_path, 'original', original_root, radiation_source(original_root))
    after = build_run(tmp_path, 'corrected', CURRENT_ROOT, radiation_source(CURRENT_ROOT))
    assert after == before


def test_sentinel_recovery_preserves_original_normal_fallbacks(tmp_path, original_root):
    body = ((VALIDATION_ROOT / 'tests/fixtures/particle_compatibility/sentinel_recovery.cpp')).read_text()
    def qualified(root):
        text = (root / 'common/operators/updateLegacyGasBoundaryMirrorKernel.cuh').read_text()
        start = text.index('template<class GasState>\n__device__ GPU_OPERATOR_REAL molecularGasConductivity')
        end = text.index('\n}', start) + 2
        return body.replace('CONDUCTIVITY', text[start:end])
    before = build_run(tmp_path, 'original', original_root, qualified(original_root))
    after = build_run(tmp_path, 'corrected', CURRENT_ROOT, qualified(CURRENT_ROOT))
    assert after == before
