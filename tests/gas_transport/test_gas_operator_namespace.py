"""Instantiate actual gas operators inside the legacy consumers' namespace.

The numerical bodies and legacy field fixture are production/shared test sources;
only CUDA annotations/indexing are emulated. This is host compilation evidence.
"""
from pathlib import Path
import re
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

@pytest.mark.parametrize('consumer',[
    'applications/gasUGKP/private_backend/GpuResidentStrict.cu',
    'applications/FSH/private_backend/GpuResidentStrict.cu',
    'applications/CHT/gpu/GpuResidentStrict.cu',
])
def test_shared_headers_are_global_before_legacy_operator_namespace(tmp_path,consumer):
    app=(ROOT/consumer).read_text().split('\nnamespace',1)[0]
    # Read the exact global shared-gas dependency includes from each entry TU.
    dependencies='\n'.join(line for line in app.splitlines() if re.match(
        r'#include "(?:gasTransport/|GpuGasOperatorDependencies\.cuh)',line))
    probe=(HERE/'gas_state_probe.cpp').read_text()
    begin=probe.index('#include "operators/clampMin.cuh"')
    end=probe.index('\n#ifdef LEGACY_OPTIONAL_SPECIES')
    source=probe[:begin]+dependencies+'\nnamespace legacy_consumer {\n'+probe[begin:end]+'\n}\nusing namespace legacy_consumer;\n'+probe[end:]
    cpp=tmp_path/'namespace.cpp';cpp.write_text(source)
    result=subprocess.run(['g++','-DLEGACY_REFERENCE','-std=c++17','-fsyntax-only',
        '-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(cpp)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
