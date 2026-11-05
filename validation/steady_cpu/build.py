#!/usr/bin/env python3
"""Build serial CPU execution of the production gas device headers (no CUDA)."""
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'validation/tests/gas_transport'

def build(output, backend="cpu", compiler=None, particles=False):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pre = (HERE / 'gas_state_probe.cpp').read_text().split('void hashBytes')[0]
    fields = re.findall(r'    ([\w:]+\*) (\w+)\{\};', (HERE/'legacy_gas_fixture.hpp').read_text())
    allocations = '\n'.join(f's.{name}=new {ty[:-1]}[storage]{{}};' for ty,name in fields)
    body = (Path(__file__).parent/'gas_reference.cpp').read_text().replace('// GENERATED_ALLOCATIONS',allocations)
    if particles:
        spec=importlib.util.spec_from_file_location('steady_particles',HERE/'steady_cpu_particles.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        body='#define STEADY_CPU_PARTICLES\n'+body.replace('// PARTICLE_FRAGMENT',module.fragment(ROOT))
    if backend == 'cuda':
        pre = '#include <cuda_runtime.h>\n' + pre
        pre = re.sub(r'^#define __(?:host|global|device|forceinline)__.*$', '', pre, flags=re.M)
        pre = re.sub(r'^#define asm.*$', '', pre, flags=re.M)
        pre = re.sub(r'^struct ThreadIndex.*$', '', pre, flags=re.M)
        body = body.replace('blockDim.x=1;blockIdx.x=0;', '')
        body = re.sub(r'template<class F>void (cells|faces)\(DeviceState&s,F fn\)\{[^\n]+',
                      r'template<class F>void \1(DeviceState&s,F fn){fn();cudaDeviceSynchronize();}', body)
        body = re.sub(r'([A-Za-z_][A-Za-z_0-9]*Kernel(?:<[^>]+>)?)\(&s',
                      r'\1<<<(s.nFaces+127)/128,128>>>(&s', body)
        body = re.sub(r'new (\w+(?:::\w+)*(?:<[^>]+>)?)\[([^]]+)\]\{([^}]*)\}',
                      r'managedArray<\1>(\2,{\3})', body)
        body = body.replace('DeviceState s;initialise(s,a);', 'DeviceState& s=*managedArray<DeviceState>(1,{});initialise(s,a);')
        pre += '\n#include <initializer_list>\n' + \
            'template<class T>T* managedArray(size_t n,std::initializer_list<T> init){T*p=nullptr;cudaMallocManaged(&p,n*sizeof(T));for(size_t i=0;i<n;++i)new(p+i)T{};size_t i=0;for(const auto&v:init)p[i++]=v;return p;}\n'
    source = output.with_suffix('.cu' if backend == 'cuda' else '.cpp')
    source.write_text(pre + body)
    subprocess.run([compiler or ('nvcc' if backend=='cuda' else 'g++'),'-std=c++17','-O2','-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),'-I'+str(ROOT),'-I'+str(ROOT/'applications/gasUGKP/private_backend'),str(source),'-o',str(output)],check=True)
    manifest={'backend':backend,'particles':particles,'generated_source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'production_headers':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [ROOT/'common',ROOT/'applications/gasUGKP/private_backend',ROOT/'applications/gasUGKPSteady/private_backend'] for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in ['.H','.cuh','.inl','.cu']}}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return output

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='/tmp/gasUGKPSteadyCpuReference')
    parser.add_argument('--backend',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--compiler')
    parser.add_argument('--particles',action='store_true',help='CPU gas+particle path with production atomic collision pool')
    args=parser.parse_args()
    print(build(args.output,args.backend,args.compiler,args.particles))
