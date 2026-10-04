"""Shared maintenance boundaries; CUDA behavior is verified separately."""
from pathlib import Path
import re
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize('app',['gasUGKP','FSH','CHT'])
@pytest.mark.parametrize('name,header',[
    ('blockReduceComponentSums','common/GpuBlockComponentReduction.cuh'),
    ('clearPoissonThermalPoolKernel','common/operators/clearPoissonThermalPoolKernel.cuh'),
    ('preparePoissonPoolSamplingKernel','common/operators/preparePoissonPoolSamplingKernel.cuh'),
])
def test_computation_has_common_owner(app,name,header):
    source=(ROOT/'applications'/app/('gpu' if app=='CHT' else 'private_backend')/'GpuResidentStrict.cu').read_text()
    assert (ROOT/header).is_file(),f'{name} needs a common owner'
    assert Path(header).name in source,f'{app} must consume the common {name}'
    assert not re.search(r'\b'+name+r'\s*\([^;{}]*\)\s*\{',source),f'{app} still maintains a local {name}'
def test_sampling_header_does_not_require_contact_fields():
    source=(ROOT/'common/operators/preparePoissonPoolSamplingKernel.cuh').read_text()
    assert 'pStuck' not in source and 'appendSelectedStuckParticleIndex' not in source
