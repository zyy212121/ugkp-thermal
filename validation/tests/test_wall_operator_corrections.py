"""Run production wall algebra, gas-face caller and boundary tensor (FP64).

Mutations caught: retaining owner tensor normal derivatives, converting uTau to
nut with U/y instead of patch snGrad, and choosing zero/lower thermal roots.
CUDA-only trap is replaced by abort in this CPU harness; closure math is intact.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest

ROOT = SOURCE_ROOT


@pytest.fixture(scope='module')
def wall_probe( tmp_path_factory):
    work = tmp_path_factory.mktemp(f'wall_ops_64')
    source = ((VALIDATION_ROOT / 'tests/fixtures/wall_operator_corrections.cpp')).read_text()
    flux = (ROOT / 'common/operators/computeRiemannGasFaceFluxDevice.cuh').read_text()
    begin = flux.index('            const bool velocityFixed = boundaryKind == 2')
    end = flux.index('            normalTemperatureGradient = targetNormalGradT;', begin)
    end += len('            normalTemperatureGradient = targetNormalGradT;')
    source = source.replace('// PRODUCTION_BOUNDARY_FRAGMENT', flux[begin:end])
    caller = (ROOT / 'common/operators/gasFaceSubgridTransportProperties.cuh').read_text()
    source = source.replace('// PRODUCTION_GAS_FACE_CALLER', caller.replace('#pragma once', '').replace('asm("trap;");', 'std::abort();'))
    path = work / 'probe.cpp'
    path.write_text(source)
    exe = work / 'probe'
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                    '-I'+str(ROOT/'common'),
                    '-I'+str(ROOT/'common/gasNumerics'), str(path), '-o', str(exe)], check=True)
    return exe


@pytest.mark.parametrize('case', ('traction', 'spalding', 'thermal'))
def test_production_wall_operator(wall_probe, case):
    result = subprocess.run([str(wall_probe), case], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
