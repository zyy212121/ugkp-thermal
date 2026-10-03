#!/usr/bin/env python3
"""Generate field registries and verify typed lifecycle closure without CUDA.

The CPU probe executes the actual allocation/release and transfer expressions,
actual copy/swap bodies and actual schema7 codec. It does not emulate launch
scheduling, allocator failures, restart validation, or cold-wall rollback paths.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import sys


def registry(meta):
    lines = ['#pragma once', '// Generated from ParticleFieldManifest.json by tools/particle_field_contract.py.',
             '// Keep field order: copy/swap instruction order is part of the operator contract.', '']
    groups = {}
    for field in meta['fields']:
        groups.setdefault(field['group'], []).append(field)
    for group, fields in groups.items():
        lines.append('#define GPU_PARTICLE_FIELDS_' + group + '(X) \\')
        for i, field in enumerate(fields):
            value = field['width_symbol'] if field['width'] != 1 else field['copy_value']
            lines.append('    X(' + ', '.join((field['name'], field['compact'], value)) + ')' + (' \\' if i + 1 < len(fields) else ''))
        lines.append('')
    return '\n'.join(lines)


def restart_fields(meta):
    fields = sorted((f for f in meta['fields'] if f['schema7'] is not None), key=lambda f: f['schema7'])
    return '// Generated schema7 order; changing this requires a new disk schema.\n' + ''.join(
        f'    fn(v.{f["name"]}, begin*{f["width"]}u, count*{f["width"]}u);\n' for f in fields)


def function_body(text, name):
    match = re.search(r'\b' + re.escape(name) + r'\s*\([^;{}]*\)\s*\{', text)
    if not match:
        raise ValueError('missing function: ' + name)
    start = match.end()
    level = 1
    for index in range(start, len(text)):
        if text[index] == '{':
            level += 1
        elif text[index] == '}':
            level -= 1
            if level == 0:
                return text[start:index]
    raise ValueError('unclosed function: ' + name)


def calls(text, name):
    # Balanced argument extraction preserves production expressions and labels.
    for match in re.finditer(r'\b' + re.escape(name) + r'\s*\(', text):
        level, string, escape = 1, False, False
        for index in range(match.end(), len(text)):
            c = text[index]
            if string:
                if escape:
                    escape = False
                elif c == '\\':
                    escape = True
                elif c == '"':
                    string = False
            elif c == '"':
                string = True
            elif c == '(':
                level += 1
            elif c == ')':
                level -= 1
                if level == 0:
                    yield text[match.start():index + 1]
                    break


def build_probe(root, meta, app, bits):
    source = root / 'applications' / app / ('gpu' if app == 'CHT' else 'private_backend') / 'GpuResidentStrict.cu'
    text = source.read_text()
    state = text[text.index('struct DeviceState'):]
    state = state[:state.index('\n};')]
    fields = [f for f in meta['fields'] if app in f['storage']]
    scratch = [f for f in meta.get('scratch_fields', []) if app in f['storage']]
    storage_fields = fields + scratch
    names = {n for f in storage_fields for n in (f['name'], f['compact']) if n}
    pointers = dict((n, t.strip()) for t, n in re.findall(r'^\s*([\w ]+)\*\s*(\w+)\s*=\s*nullptr;', state, re.M))
    declared_compact = {n for n in pointers if n.startswith('compactP') and n not in ('compactPressureParameters',)}
    expected_compact = {f['compact'] for f in fields}
    if declared_compact != expected_compact:
        raise ValueError(f'{app}: unregistered/missing compact fields: {sorted(declared_compact ^ expected_compact)}')
    declared_primary = {n for n in pointers if re.match(r'^p(?:[A-Z]|[xyz]$|u[xyz](?:Old)?$|[dm]$)', n)}
    expected_primary = {f['name'] for f in storage_fields}
    if declared_primary != expected_primary:
        raise ValueError(f'{app}: unregistered/missing primary fields: {sorted(declared_primary ^ expected_primary)}')
    for f in storage_fields:
        for name in (f['name'], f['compact']):
            if not name:
                continue
            if pointers.get(name) != f['storage'][app]:
                raise ValueError(f'{app}: missing/wrong declaration {name}; expected {f["storage"][app]}, got {pointers.get(name)}')
    release = function_body(text, 'releaseState')
    upload = function_body(text, 'ugkwpGpuResidentStrictUploadParticleRestartMirror')
    download = function_body(text, 'ugkwpGpuResidentStrictDownloadParticleRestartMirror')
    if app != 'gasUGKP':
        saved = (root / 'common/GpuParticleSavedVelocityRestart.inl').read_text()
        upload += function_body(saved, 'ugkwpGpuResidentStrictUploadParticleSavedVelocity')
        download += function_body(saved, 'ugkwpGpuResidentStrictDownloadParticleSavedVelocity')
    alloc_calls, release_calls, upload_calls, download_calls, aliases = [], [], [], [], {}
    for f in storage_fields:
        for name in (f['name'], f['compact']):
            if not name:
                continue
            selected = [call for call in calls(text, 'allocate') if re.match(r'allocate\s*\(\s*s->'+name+r'\s*,', call)]
            if not selected and f['group'].startswith('COLD'):
                matches = re.findall(r's->' + name + r'\s*=\s*(\w+)\s*;', text)
                matches = [n for n in matches if n != 'nullptr']
                if len(matches) != 1:
                    raise ValueError(f'{app}: cold installation is not unique for {name}')
                alias = matches[0]
                aliases[name] = alias
                selected = [call for call in calls(text, 'allocate') if re.match(r'allocate\s*\(\s*'+alias+r'\s*,', call)]
            if len(selected) != 1:
                raise ValueError(f'{app}: allocation closure for {name}: expected one, got {len(selected)}')
            alloc_calls += selected
            selected_release = [call for call in calls(release, 'release') if re.match(r'release\s*\(\s*s->'+name+r'\s*\)', call)]
            if len(selected_release) != 1:
                raise ValueError(f'{app}: release closure for {name}: expected one, got {len(selected_release)}')
            release_calls += selected_release
        if f in fields:
            up = [call for call in calls(upload, 'copyToDevice') if re.match(r'copyToDevice\s*\(\s*s->'+f['name']+r'\s*,', call)]
            down = [call for call in calls(download, 'copyToHost') if re.search(r',\s*s->'+f['name']+r'\s*,', call)]
            if len(up) != 1 or len(down) != 1:
                raise ValueError(f'{app}: transfer closure for {f["name"]}: {len(up)} uploads, {len(down)} downloads')
            upload_calls += up
            download_calls += down
    if len(upload_calls) != len(fields) or len(download_calls) != len(fields):
        raise ValueError(f'{app}: transfer closure mismatch: {len(fields)} fields, {len(upload_calls)} uploads, {len(download_calls)} downloads')
    code = r'''
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#define __host__
#define __device__
#define __forceinline__ inline
struct alignas(16) float4 {float x,y,z,w;};
namespace Foam { namespace gpuThermal {
constexpr int coldWallAxialNodeCount=8, coldWallRadialRingCount=8,
 coldWall2DNodeCount=64, coldWall2DRadialNodeCount=8;
constexpr unsigned char particleWallMobile=0;
} namespace gpuWall { template<class S> void publishWallBoundParticleIndex(S&,int){} } }
std::map<void*,size_t> live;
template<class T> int allocate(T*& p,size_t n,const char*) {
 if(p) throw std::runtime_error("duplicate allocation"); p=new T[n]{};live[p]=n;return 0;
}
template<class T> void release(T*& p) {
 if(!p || !live.erase(p)) throw std::runtime_error("invalid release"); delete[] p;p=nullptr;
}
template<class T,class U> int copyToDevice(T* d,const U* s,size_t n,const char*) {
 if(!d || live.at(d)!=n) throw std::runtime_error("upload size mismatch");
 std::copy(s,s+n,d);return 0;
}
template<class T,class U> int copyToHost(T* d,const U* s,size_t n,const char*) {
 if(!s || live.at(const_cast<U*>(s))!=n) throw std::runtime_error("download size mismatch");
 std::copy(s,s+n,d);return 0;
}
'''
    code += f'using GpuReal={"float" if bits==32 else "double"};using GpuTime=double;\n'
    code += 'struct DeviceState { int coldWallSolidificationEnabled=1,coldWall2DEnabled=1;\n'
    code += ''.join(f'{pointers[n]}* {n}=nullptr;\n' for n in sorted(names)) + '};\n'
    if app != 'gasUGKP':
        code += '#include "GpuCellLocalThermalFields.cuh"\n'
        code += f'using ProbeExtra=CellLocalThermalExtraFields<{"true" if app=="CHT" else "false"}, {"true" if app=="FSH" else "false"}>;\n#define GPU_PARTICLE_EXTRA_FIELDS ProbeExtra\n'
    code += '#include "GpuCellLocalPrimary.cuh"\n#include "operators/swapParticlePointerDevice.cuh"\n'
    commit = (root / 'common/GpuParticleBufferCommit.cuh').read_text()
    code += 'void swapParticleBuffersDevice(DeviceState& s) {' + function_body(commit, 'swapParticleBuffersDevice') + '}\n'
    code += '#include "GpuParticleRestartTheta.H"\n' if app == 'CHT' else ''
    if app != 'gasUGKP':
        code += '#include "GpuThermalParticleRestartCodec.H"\nstruct View {size_t count=3;\n'
        code += ''.join(f'{f["host"][app]}* {f["name"]};\n' for f in fields if f['schema7'] is not None) + '};\n'
    code += 'int main(int argc,char**argv){try{DeviceState state;auto*s=&state;const size_t np=3,n=3,particleCapacity=3,count=3;\n'
    for constant in ('coldNodeStorage', 'coldRingStorage', 'cold2DNodeStorage', 'cold2DRingStorage'):
        match = re.search(r'const size_t '+constant+r'\s*=[^;]+;', text)
        if match:
            code += match.group(0) + '\n'
    for name, alias in aliases.items():
        code += f'{pointers[name]}* {alias}=nullptr;\n'
    code += ''.join(call + ';\n' for call in alloc_calls)
    code += ''.join(f's->{name}={alias};\n' for name, alias in aliases.items())
    for f in storage_fields:
        for name in (f['name'], f['compact']):
            if not name:
                continue
            code += f'if(!s->{name} || live.at(s->{name})!=n*{f["width"]}) throw std::runtime_error("allocation closure: {name}");\n'
    for index, f in enumerate(fields):
        name, typ, width = f['name'], f['host'][app], f['width']
        code += f'std::vector<{typ}> h_{name}(n*{width}); auto* {name}=h_{name}.data();\n'
        code += f'for(size_t j=0;j<n*{width};++j) {name}[j]=static_cast<{typ}>({index+1}*7+j);\n'
    code += 'pStatus[0]=pStatus[1]=pStatus[2]=1;pCellId[0]=pCellId[1]=pCellId[2]=9;\n'
    if app != 'gasUGKP':
        code += 'pStuck[0]=0;pStuck[1]=2;pStuck[2]=3;auto*x=puxOld;auto*y=puyOld;auto*z=puzOld;\n'
    if app == 'CHT':
        code += 'std::vector<GpuReal> physicalTheta(n);std::vector<GpuTime> contactAge(n);\n'
        code += 'unpackParticleRestartTheta(n,pStuck,pTheta,physicalTheta.data(),contactAge.data());\n'
    code += ''.join(call + ';\n' for call in upload_calls)
    # Exercise the production copy body, including cold arrays and conditional independent age.
    code += 'copyCellLocalParticle(*s,1,9,0);\n'
    for f in fields:
        name, compact, width = f['name'], f['compact'], f['width']
        if name in ('pCellId', 'pStatus'):
            expected = '9' if name == 'pCellId' else '1'
            code += f'if(s->{compact}[0]!={expected}) throw std::runtime_error("copy semantic: {name}");\n'
        else:
            code += f'for(size_t j=0;j<{width};++j) if(s->{compact}[j]!=s->{name}[{width}+j]) throw std::runtime_error("copy closure: {name}");\n'
    # Read back the original uploaded arrays before swap; same ABI conversions and width expressions.
    code += ''.join(call + ';\n' for call in download_calls)
    if app == 'CHT':
        code += 'packParticleRestartTheta(n,pStuck,contactAge.data(),pTheta);\n'
        code += 'if(s->pTheta[1]!=0 || s->pContactAge[0]!=0 || pTheta[1]!=contactAge[1]) throw std::runtime_error("theta wire mapping");\n'
    for f in fields:
        if f['name'] == 'pContactAge':
            continue
        name, index = f['name'], fields.index(f)
        if name in ('pStatus', 'pCellId', 'pStuck'):
            continue
        code += f'for(size_t j=0;j<n*{f["width"]};++j) if({name}[j]!=static_cast<{f["host"][app]}>({index+1}*7+j)) throw std::runtime_error("roundtrip: {name}");\n'
    if app != 'gasUGKP':
        code += 'View v;\n' + ''.join(f'v.{f["name"]}={f["name"]};\n' for f in fields if f['schema7'] is not None)
        disk_time = 'float' if app == 'FSH' else 'double'
        code += f'std::ostringstream wire(std::ios::binary);Foam::gpuThermalRestart::write<{disk_time}>(wire,v);\n'
        code += 'std::ofstream out(argv[1],std::ios::binary);out<<wire.str();out.close();\n'
        code += 'std::istringstream input(wire.str(),std::ios::binary);std::string marker;size_t total;std::uint32_t bound;input>>marker>>total>>bound;\n'
        code += f'Foam::gpuThermalRestart::readPayload<{disk_time}>(input,bound,v);\n'
    code += ''.join(f'auto* old_{f["name"]}=s->{f["name"]};auto* next_{f["name"]}=s->{f["compact"]};\n' for f in fields)
    code += 'swapParticleBuffersDevice(*s);\n'
    code += ''.join(f'if(s->{f["name"]}!=next_{f["name"]} || s->{f["compact"]}!=old_{f["name"]}) throw std::runtime_error("swap closure: {f["name"]}");\n' for f in fields)
    code += ''.join(call + ';\n' for call in release_calls)
    code += 'if(!live.empty()) throw std::runtime_error("unreleased particle storage");std::cout<<"PASS typed lifecycle/copy/swap/transfer\\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\\n";return 1;}}\n'
    return code


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--generate', action='store_true', help='Write only the two generated field registries')
    ap.add_argument('--cpu', action='store_true', help='Compile/run typed CPU probes; no CUDA toolchain or GPU use')
    ap.add_argument('--output', type=Path, help='Probe output outside checkout')
    args = ap.parse_args()
    root = args.root.resolve()
    meta = json.loads((root / 'common/ParticleFieldManifest.json').read_text())
    if meta['schema'] != 1 or meta['disk_schema'] != 7:
        raise ValueError('unsupported field metadata or disk schema')
    ordinals = [f['schema7'] for f in meta['fields'] if f['schema7'] is not None]
    if sorted(ordinals) != list(range(30)):
        raise ValueError('schema7 must contain each of its 30 wire ordinals exactly once')
    for field in meta['fields']:
        if field['schema7'] is None and 'wire_alias' not in field:
            raise ValueError('persistent field lacks an explicit restart mapping: ' + field['name'])
    generated = {'GpuParticleFields.cuh': registry(meta), 'GpuParticleRestartFields.inl': restart_fields(meta)}
    for name, data in generated.items():
        path = root / 'common' / name
        if args.generate:
            path.write_text(data)
        elif path.read_text() != data:
            raise ValueError('generated field registry drift: ' + name)
    if args.cpu and (not args.output or root == args.output.resolve() or root in args.output.resolve().parents):
        ap.error('--cpu needs --output outside the checkout')
    golden_path = root / 'tests/fixtures/shared_operators/schema7_wire_golden.json'
    golden = json.loads(golden_path.read_text()) if golden_path.is_file() else {}
    for app in ('gasUGKP', 'FSH', 'CHT'):
        if not (root / 'applications' / app).is_dir():
            continue
        for bits in ((32, 64) if app == 'CHT' else (64,)):
            code = build_probe(root, meta, app, bits)
            if args.cpu:
                args.output.mkdir(parents=True, exist_ok=True)
                label = app + str(bits)
                cpp, exe, wire = (args.output / (label + ext) for ext in ('.cpp', '.probe', '.wire'))
                cpp.write_text(code)
                subprocess.run(['g++', '-std=c++17', '-O0', '-I'+str(root/'common'), str(cpp), '-o', str(exe)], check=True)
                subprocess.run([str(exe), str(wire)], check=True)
                if app != 'gasUGKP':
                    digest = hashlib.sha256(wire.read_bytes()).hexdigest()
                    if golden.get(label) != digest:
                        raise ValueError(f'{label}: schema7 binary golden mismatch ({digest})')
            print(f'PASS {app}{bits} field declaration/allocation/release/transfer closure')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print('particle field contract error:', exc, file=sys.stderr)
        raise SystemExit(1)
