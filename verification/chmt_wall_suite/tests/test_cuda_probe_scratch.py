"""CUDA source/storage contracts; these tests do not claim GPU execution."""
import importlib.util
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SUITE = ROOT / 'verification/chmt_wall_suite'
sys.path.insert(0, str(SUITE))


def load():
    spec = importlib.util.spec_from_file_location('scratch_host_cases', SUITE / 'host_cases.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_existing_cuda_probe_uses_bounded_external_scratch(tmp_path, monkeypatch):
    host = load()

    def compile_only(args, log, cwd=None):
        Path(log).write_text('unit fixture compiler; no real compilation or GPU execution\n'
                             'ptxas info    : Function properties for launchProbe\n'
                             '    123 bytes stack frame, 4 bytes spill stores, 8 bytes spill loads\n')
        Path(args[args.index('-o') + 1]).write_bytes(b'unit fixture artifact')
        return ''

    monkeypatch.setattr(host, '_command', compile_only)
    with pytest.raises(host.ArtifactBuilt) as built:
        host.wall_constant(ROOT, tmp_path / 'probe', backend='cuda-build')
    assert built.value.metadata['cuda_stack_limit_override'] is False
    assert built.value.metadata['ptxas_resource_report'][-1] == '123 bytes stack frame, 4 bytes spill stores, 8 bytes spill loads'
    source = (tmp_path / 'probe/probe.cu').read_text()
    assert 'cudaDeviceSetLimit' not in source
    assert 'cudaMalloc(&scratch,sizeof(ProbeScratch))' in source
    assert 'cudaFree(scratch)' in source
    assert 'launchProbe<<<1,1>>>(scratch,status)' in source
    device = source.split('__device__ int productionProbe', 1)[1].split('__global__', 1)[0]
    assert not re.search(r'WallWorkspace\s*<[^>]+>\s*\w+\s*;', device)


def test_multiple_workspace_slots_keep_alignment_and_reset_in_place(tmp_path):
    host = load()
    base = '''
#include <cstddef>
#include <cstdint>
struct alignas(64) Wide { double x[32]{}; };
struct Small { int value=7; };
'''
    body = '''
PROBE_WORKSPACE(0, a); PROBE_WORKSPACE(1, b);
if(a.x[0]!=0 || b.value!=7)return 2;
if(reinterpret_cast<std::uintptr_t>(&a)%alignof(Wide))return 3;
a.x[0]=9;
for(int i=0;i<2;++i){PROBE_WORKSPACE(2, c);if(c.value!=7)return 4;c.value=99;}
if(a.x[0]!=9 || b.value!=7)return 5;
return 0;
'''
    source = host.cuda_probe_source(base, body, ('Wide', 'Small', 'Small'))
    assert 'alignas(Wide)' in source and 'sizeof(Wide)' in source
    assert 'alignas(Small)' in source and 'sizeof(Small)' in source
    assert all('workspace_' + str(i) in source for i in range(3))
    # Compile the exact emitted storage and device body as C++ to check actual
    # object lifetime, alignment, disjoint slots and loop reinitialization.
    portable = source[:source.index('__global__')]
    portable = portable.replace('#include <cuda_runtime.h>', '')
    portable = portable.replace('__device__', '').replace('__host__', '').replace('__forceinline__', 'inline')
    portable += '\nint main(){ProbeScratch scratch;return productionProbe(&scratch);}\n'
    cpp = tmp_path / 'storage.cpp'
    cpp.write_text(portable)
    executable = tmp_path / 'storage'
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', str(cpp), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


def test_cpu_workspace_expansion_preserves_local_declarations():
    host = load()
    body = 'PROBE_WORKSPACE(0, w); use(w); PROBE_WORKSPACE(1, other);'
    actual = host.workspace_body(body, ('WallWorkspace<double,2,32>', 'WallWorkspace<float,10,64>'))
    assert actual == 'WallWorkspace<double,2,32> w; use(w); WallWorkspace<float,10,64> other;'


def test_bvp_profile_source_reads_physical_temperature(tmp_path, monkeypatch):
    host = load()
    bodies = []

    def capture(root, work, base, body, **kwargs):
        bodies.append(body)
        raise host.ArtifactBuilt({})

    monkeypatch.setattr(host, '_build_run', capture)
    with pytest.raises(host.ArtifactBuilt):
        host.wall_bvp(ROOT, tmp_path)
    assert 'reactingWallProfileTemperature(m.in,w,i)' in bodies[0]
    assert 'w.state[i][2]' not in bodies[0]


@pytest.mark.parametrize('body,types', [
    ('PROBE_WORKSPACE(1, w);', ('One',)),
    ('PROBE_WORKSPACE(0, a); PROBE_WORKSPACE(0, b);', ('One',)),
    ('PROBE_WORKSPACE(0, a);', ('One', 'Unused')),
])
def test_workspace_slots_reject_missing_duplicate_or_out_of_range_declarations(body, types):
    with pytest.raises(ValueError):
        load().workspace_body(body, types)
