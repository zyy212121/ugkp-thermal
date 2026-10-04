"""Keep the complete decision-to-ready protocol owned by common."""
from pathlib import Path
import re
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize('app',['gasUGKP','FSH','CHT'])
def test_auto_protocol_and_state_have_one_owner(app):
    common=ROOT/'common/GpuAutomaticCsrSchedule.cuh'
    fields=ROOT/'common/GpuAutomaticCsrScheduleFields.inl'
    assert common.is_file() and fields.is_file(), 'automatic schedule needs one common owner'
    source=(ROOT/'applications'/app/('gpu' if app=='CHT' else 'private_backend')/'GpuResidentStrict.cu').read_text()
    assert 'GpuAutomaticCsrSchedule.cuh' in source and 'GpuAutomaticCsrScheduleFields.inl' in source
    a=source.index('int runToolB3');b=source.index('\n}',a)
    wrapper=source[a:b]
    assert 'runAutomaticCsrSchedule' in wrapper
    assert 'cudaMemcpy' not in wrapper and 'schedulingAdvanceCount' not in wrapper
    assert not re.search(r'__global__ void maximumDirectoryOccupancyKernel',source)
