"""Exercise production S1 survivor bins, moments, payload gather and pressure on CUDA.

Detects missing S1 fusion, directory holes, overlapping destinations, lost payloads,
wrong pressure indices, premature buffer swaps and restart side effects.
"""
from pathlib import Path
import os, re, shutil, subprocess
import pytest
ROOT = Path(__file__).resolve().parents[1]

def test_s1_survivor_fusion_addresses_payload_and_pressure(tmp_path):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists(): pytest.skip("CUDA compiler required")
    backend = ROOT / "applications/gasUGKP/private_backend"
    source = (backend / "GpuResidentStrict.cu").read_text()
    advance = source[source.index('    UGKP_DEV_PROBE_ENTER(ProbeBinPost);'):]
    survivor_arg = re.search(r'binParticlesByCell\(s, block, (.*?)\) != 0', advance).group(1)
    pressure_arg = re.search(r'applyCollisionalPressureKick\(s, 0.5\*dt, block, 0,\s*(.*?)\) != 0', advance, re.S).group(1)
    struct = source[source.index('struct DeviceState'):source.index('\n};',source.index('struct DeviceState'))]
    ptr = re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int)\*\s+(\w+)\s*=', struct, re.M)
    ptr = [(t,n) for t,n in ptr if 'TempStorage' not in n and n not in ['diagnosticPreTransportParticleCount', 'sourceInjectedCount']]
    names = {n for t,n in ptr}
    pairs = [(t,'p'+n[len('compactP'):],n) for t,n in ptr if n.startswith('compactP') and 'p'+n[len('compactP'):] in names]
    code = (ROOT / 'tests/fixtures/s1_survivor_fusion.cu.in').read_text()
    allocations = '\n'.join('mem(d->'+n+',N);' for t,n in ptr)
    initialize = '\n'.join(f'for(int i=0;i<N;++i){{d->{a}[i]=static_cast<{t}>(i%17+{j+1});d->{b}[i]=static_cast<{t}>(91);}}' for j,(t,a,b) in enumerate(pairs))
    checks = '\n'.join(f'CHECK(d->{b}[pos]==d->{a}[i]);' for t,a,b in pairs)
    code = code.replace('@ALLOCATE@',allocations).replace('@INITIALIZE@',initialize).replace('@CHECK_PAYLOAD@',checks)
    code = code.replace('@SURVIVOR_ARG@',survivor_arg).replace('@PRESSURE_ARG@',pressure_arg)
    cpp=tmp_path/'s1_survivor.cu';exe=tmp_path/'s1_survivor';cpp.write_text(code)
    q=subprocess.run([nvcc,'-std=c++17','-O3','--fmad=true','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'-I'+str(backend),'-I'+str(ROOT/'common'),'-I'+str(backend.parent/'gpu'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
    q=subprocess.run([str(exe)],capture_output=True,text=True,timeout=120)
    assert q.returncode==0,q.stdout+q.stderr
