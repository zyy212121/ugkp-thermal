"""Execute real build scripts in minimal archives with observable tool stubs.

The stubs model successful outputs and real compiler/archiver failures. They
prove entry control flow, not CUDA/OpenFOAM compilation or GPU correctness.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = SOURCE_ROOT
PUBLIC_SCRIPTS = {app: 'tool/'+app+'/build_public_frontend.sh' for app in ('gasUGKP','FSH')}
SCRIPTS = {
    'gasUGKP': 'tool/gasUGKP/build_private_backend.sh',
    'FSH': 'tool/FSH/build_private_backend.sh',
    'CHT': 'tool/CHT/build_cuda_solver.sh',
    'CHMT': 'applications/CHMT/Allwmake',
}
STUB = r'''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
name=Path(sys.argv[0]).name; args=sys.argv[1:]
with open(os.environ['TOOL_LOG'],'a') as f:f.write(json.dumps([name,args,str(Path.cwd())])+'\n')
if name=='ldd':
    print('libcuda libcudart libOpenFOAM libfiniteVolume libmeshTools')
    sys.exit(0)
if name in ('git','sha256sum'):sys.exit(93)
if '--version' in args:print(name+' stub version');sys.exit(0)
if os.environ.get('FAIL_TOOL')==name:sys.exit(17)
if name=='patch':sys.exit(0)
if name=='ar':
    if os.environ.get('EXPECT_FRESH_ARCHIVE')=='1' and Path(args[1]).exists():sys.exit(20)
    if not all(Path(p).is_file() for p in args[2:]):sys.exit(18)
    output=Path(args[1])
elif name=='wmake':
    if not all(Path(p).is_file() for p in os.environ.get('UGKWP_CUDA_EXE_LIBS','').split() if p.endswith('.a')):sys.exit(18)
    app=Path.cwd().name
    files=Path('Make/files')
    if files.exists():app=files.read_text().split('/')[-1].strip()
    output=Path(os.environ['FOAM_USER_APPBIN'])/app
else:
    if '-o' not in args:sys.exit(19)
    if name in ('g++','c++') and not all(Path(p).is_file() for i,p in enumerate(args) if p.endswith(('.a','.o')) and (i==0 or args[i-1]!='-o')):sys.exit(18)
    output=Path(args[args.index('-o')+1])
if os.environ.get('MISSING_TOOL')!=name:
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('#!/bin/sh\nexit 0\n');output.chmod(0o755)
'''


def put(root, relative, data, executable=False):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data)
    if executable:
        path.chmod(0o755)
    return path


def copy_file(root, relative, source=ROOT):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / relative, path)
    return path


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / 'archive'
    cases = tmp_path / 'case-archive'
    for relative in ('example/tools/case_paths.sh', 'example/tools/case_paths.py'):
        copy_file(cases, relative, source=CASES_ROOT)
    for relative in ['Allwmake', *SCRIPTS.values(), *PUBLIC_SCRIPTS.values(),
                     'tool/CHMT/build_info.py', 'tool/cuda_build_flags.sh',
                     'validation/tools/CHMT/devtools/source_fingerprint.py']:
        copy_file(root, relative)
    for tool in ('managed_mirrors.py', 'particle_field_contract.py', 'compute_source_hash.py'):
        put(root, 'tool/' + tool, "raise SystemExit('offline tool must not run in build')\n")
    put(root, 'validation/tools/CHMT/devtools/source_fingerprint.py', "raise RuntimeError('no source hash traversal in build')\n")
    put(root, 'validation/tools/CHMT/devtools/check_host.sh', 'echo host-checks > "$HOST_MARKER"\n', True)
    for app in ('gasUGKP', 'FSH', 'CHT', 'CHMT'):
        (root / 'applications' / app / 'gpu').mkdir(parents=True, exist_ok=True)
    put(root, 'tool/CHT/bartzSolidReplay/Make/files', 'forbidden-helper')
    put(root, 'applications/CHMT/gpu/SharedGasAdvance.cuh', 'not hashed')
    bindir = tmp_path / 'bin'
    for tool in ('g++', 'c++', 'ar', 'wmake', 'git', 'ldd', 'sha256sum', 'patch'):
        put(bindir, tool, STUB, True)
    put(tmp_path, 'cuda/bin/nvcc', STUB, True)
    put(tmp_path, 'foam/src/finiteVolume/lnInclude/fvCFD.H', '// available header')
    appbin = tmp_path / 'appbin'
    appbin.mkdir()
    env = {**os.environ, 'UGKP_SOURCE_ROOT': str(root), 'UGKP_CASES_ROOT': str(cases),
           'PATH': str(bindir) + ':' + os.environ['PATH'],
           'TOOL_LOG': str(tmp_path / 'calls.jsonl'), 'CUDA_HOME': str(tmp_path / 'cuda'),
           'NVCC': str(tmp_path / 'cuda/bin/nvcc'), 'CXX': str(bindir / 'g++'),
           'WM_PROJECT_DIR': str(tmp_path / 'foam'), 'WM_PROJECT_VERSION': 'archive-custom',
           'WM_PROJECT': 'OpenFOAM', 'WM_OPTIONS': 'test-options',
           'FOAM_LIBBIN': str(tmp_path / 'foamlib'), 'FOAM_MPI': 'dummy',
           'FOAM_USER_APPBIN': str(appbin), 'CHMT_CUDA_ARCH': 'compute_89',
           'CHMT_BUILD': str(root / 'custom-output'),
           'CHMT_FRONTEND_BUILD': str(root / 'custom-frontend-output'),
           'UGKWP_CUDA_EXE_INC': '-DUGKWP_USE_CUDA', 'HOST_MARKER': str(tmp_path/'host-marker')}
    for key in ('FAIL_TOOL', 'MISSING_TOOL', 'GAS_UGKP_FMAD', 'FSH_FMAD', 'UGKP_DEVELOPMENT_PROBES'):
        env.pop(key, None)
    return root, env


def run(tree, relative, *args, **changes):
    root, env = tree
    return subprocess.run(['bash', str(root / relative), *args], env={**env, **changes}, capture_output=True, text=True)


def calls(tree):
    path = Path(tree[1]['TOOL_LOG'])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize('app', SCRIPTS)
def test_normal_build_runs_compilers_without_offline_admission(tree, app):
    result = run(tree, SCRIPTS[app])
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = calls(tree)
    assert any(name == 'nvcc' and '-c' in args for name, args, _ in recorded)
    assert not any(name in ('ldd', 'git', 'sha256sum') for name, _, _ in recorded)
    assert not any('bartzSolidReplay' in cwd for _, _, cwd in recorded)
    assert (Path(tree[1]['FOAM_USER_APPBIN']) / app).is_file()


@pytest.mark.parametrize('app,tool', [
    ('gasUGKP','nvcc'),('gasUGKP','ar'),('gasUGKP','c++'),('gasUGKP','wmake'),
    ('FSH','nvcc'),('FSH','ar'),('FSH','c++'),('FSH','wmake'),
    ('CHT','nvcc'),('CHT','ar'),('CHT','wmake'),
    ('CHMT','nvcc'),('CHMT','g++'),
])
@pytest.mark.parametrize('fault', ['FAIL_TOOL', 'MISSING_TOOL'])
def test_actual_compiler_failure_or_absent_output_still_fails(tree, app, tool, fault):
    result = run(tree, SCRIPTS[app], **{fault: tool})
    assert result.returncode != 0, result.stdout
    assert any(name == tool and '--version' not in args for name, args, _ in calls(tree))


@pytest.mark.parametrize('frontend', [False, True])
def test_chmt_host_optimization_and_explicit_frontend_mode(tree, frontend):
    result = run(tree, SCRIPTS['CHMT'], *(['--frontend-only'] if frontend else []))
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = calls(tree)
    hosts = [args for name, args, _ in recorded if name == 'g++' and '-o' in args]
    assert len(hosts) == 1 and '-O2' in hosts[0]
    cuda = [args for name, args, _ in recorded if name == 'nvcc' and '-c' in args]
    assert len(cuda) == (0 if frontend else 1)
    if cuda:
        assert '-O2' in cuda[0] and '-arch=compute_89' in cuda[0]
    expected = 'CHMT-input-check' if frontend else 'CHMT'
    out = tree[1]['CHMT_FRONTEND_BUILD' if frontend else 'CHMT_BUILD']
    assert (Path(out) / expected).is_file()


@pytest.mark.parametrize('fault', ['FAIL_TOOL', 'MISSING_TOOL'])
@pytest.mark.parametrize('mode', [[], ['--frontend-only']])
def test_chmt_host_compiler_errors_and_missing_outputs(tree, fault, mode):
    result = run(tree, SCRIPTS['CHMT'], *mode, **{fault: 'g++'})
    assert result.returncode != 0, result.stdout


def test_selected_host_checks_do_not_start_native_build(tree):
    result = run(tree, SCRIPTS['CHMT'], '--host-checks')
    assert result.returncode == 0, result.stderr
    assert Path(tree[1]['HOST_MARKER']).read_text().strip() == 'host-checks'
    assert calls(tree) == []
    assert run(tree, SCRIPTS['CHMT'], '--unknown').returncode == 2


@pytest.mark.parametrize('app,fmad', [('gasUGKP','GAS_UGKP_FMAD'), ('FSH','FSH_FMAD')])
@pytest.mark.parametrize('value,expected', [('off','--fmad=false'),('yes','--fmad=true')])
def test_private_backend_fmad_and_probe_arguments_unchanged(tree, app, fmad, value, expected):
    result = run(tree, SCRIPTS[app], **{fmad: value, 'UGKP_DEVELOPMENT_PROBES':'on'})
    assert result.returncode == 0, result.stdout + result.stderr
    nvcc = next(args for name, args, _ in calls(tree) if name == 'nvcc' and '-c' in args)
    assert expected in nvcc and '-DUGKP_DEVELOPMENT_PROBES=1' in nvcc
    assert run(tree, SCRIPTS[app], **{fmad:'invalid'}).returncode == 2


def test_cht_builds_one_resident_and_wall_energy_without_stale_objects(tree):
    root, env = tree
    stale = root / 'applications/CHT/Make' / env['WM_OPTIONS'] / 'gpu'
    stale.mkdir(parents=True, exist_ok=True)
    for name in ('GpuResidentStrict32.o', 'GpuResidentStrict64.o', 'GpuPrecisionDispatch.o'):
        (stale / name).write_text('obsolete')
    archive = root / 'applications/CHT/gpu/libugkpcht_cuda_internal.a'
    archive.write_text('obsolete archive members')
    result = run(tree, SCRIPTS['CHT'], EXPECT_FRESH_ARCHIVE='1')
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = calls(tree)
    cuda = [args for name, args, _ in recorded if name == 'nvcc']
    assert len(cuda) == 2
    assert any(arg.endswith('GpuResidentStrict.cu') for arg in cuda[0])
    assert any(arg.endswith('GpuWallEnergy64.cu') for arg in cuda[1])
    assert all('--fmad=false' in args for args in cuda)
    assert not any('UGKWP_GPU_REAL_BITS' in arg or 'GpuBackendNames' in arg or 'Fp32' in arg
                   for args in cuda for arg in args)
    ar = next(args for name, args, _ in recorded if name == 'ar')
    assert [Path(arg).name for arg in ar[2:]] == ['GpuResidentStrict.o', 'GpuWallEnergy64.o']
    assert not any((stale / name).exists() for name in
                   ('GpuResidentStrict32.o', 'GpuResidentStrict64.o', 'GpuPrecisionDispatch.o'))
    assert not any('GpuPrecisionDispatch.C' in arg for _, args, _ in recorded for arg in args)


def test_top_level_build_reaches_all_children_without_contracts_or_source_scan(tree):
    result = run(tree, 'Allwmake')
    assert result.returncode == 0, result.stdout + result.stderr
    assert len([1 for name, args, _ in calls(tree) if name == 'nvcc' and '-c' in args]) == 5


def test_archive_build_header_compiles_without_git_manifest_or_identity_lock(tree, tmp_path):
    root, env = tree
    script = root / 'tool/CHMT/build_info.py'
    output = root / 'inside-checkout-output'
    command = [sys.executable, str(script), '--compile-only', '--cxx', shutil.which('g++'), '--output', str(output)]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    source = tmp_path / 'identity.cpp'
    source.write_text('#include "CHMTBuildIdentity.H"\n#include <string>\nint main(){return std::string(CHMT_SOURCE_FINGERPRINT)=="UNKNOWN"?0:1;}\n')
    exe = tmp_path / 'identity'
    compiler = subprocess.run([shutil.which('g++'), '-std=c++17', '-I'+str(output), str(source), '-o', str(exe)], capture_output=True, text=True)
    assert compiler.returncode == 0, compiler.stderr
    assert subprocess.run([str(exe)]).returncode == 0
    assert not any(name == 'git' for name, _, _ in calls(tree))
    data = json.loads((output/'build.json').read_text())
    assert data['openfoam_version'] == 'archive-custom'
    assert data['solver_commit'] == 'UNKNOWN'
    assert 'shared_adapter_sha256' not in data
    missing = subprocess.run(command + ['--cxx', str(tmp_path/'no-compiler')], env=env, capture_output=True, text=True)
    assert missing.returncode != 0
    locked = tmp_path / 'locked'
    locked.mkdir(); locked.chmod(0o555)
    denied = subprocess.run(command[:-1] + [str(locked/'output')], env=env, capture_output=True, text=True)
    assert denied.returncode != 0
    locked.chmod(0o755)


@pytest.mark.parametrize('case,solver', [('dustyWave','dustyWaveUGKP'), ('windSandShockTube','windSandUGKP')])
def test_consistency_build_runs_despite_offline_tests_and_link_policy(tree, case, solver):
    root, env = tree
    cases = Path(env['UGKP_CASES_ROOT'])
    base = 'example/tools/consistency/' + case
    script = copy_file(cases, base + '/assets/build/build_validation_solver.sh', source=CASES_ROOT)
    result_dir = cases / 'example/results/consistency' / case
    put(cases, base + '/assets/build/clean_validation_solver.sh', '#!/bin/sh\nexit 0\n', True)
    for filename in ('test_constant_response_time_contract.py', 'test_staged_coupling_contract.py'):
        put(cases, base+'/assets/solver_source/tests/'+filename, 'raise SystemExit(94)\n')
    for relative in ('a.H','a.C','gpu/GpuDragModel.H','private_backend/GpuResidentStrict.cu','private_backend/GpuBackendServer.C','Make/options'):
        put(root, 'applications/gasUGKP/'+relative, 'minimal\n')
    put(root, 'applications/gasUGKP/Make/files', 'EXE = $(FOAM_USER_APPBIN)/gasUGKP\n')
    for relative in ('gpu/GpuDragModel.H','private_backend/GpuDragModels.cuh','GpuResidentStrict.constant-response-time.patch','GpuResidentStrict.axial-fluctuation.patch'):
        put(cases, base+'/assets/solver_source/'+relative, 'adapter\n')
    put(root, 'common/GpuSchedulingConfiguration.H', '// scheduler')
    result = run(tree, script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (result_dir/'.validation-bin'/solver).is_file()
    assert not any(name in ('ldd','sha256sum') for name, _, _ in calls(tree))
    assert run(tree, script, FAIL_TOOL='nvcc').returncode != 0
    (result_dir/'.validation-bin'/solver).unlink()
    assert run(tree, script, MISSING_TOOL='wmake').returncode != 0


@pytest.mark.parametrize('header', ['UGKP_PARTICLES_SCHEMA4 1','UGKP_PARTICLES_SCHEMA2 97'])
def test_windsand_launch_preserves_input_without_numeric_precheck(tree, header):
    root, env = tree
    cases = Path(env['UGKP_CASES_ROOT'])
    base = 'example/tools/consistency/windSandShockTube'
    script = copy_file(cases, base+'/Allrun', source=CASES_ROOT)
    audit_tool = copy_file(cases, base+'/assets/initialization/prepare_axial_particles.py', source=CASES_ROOT)
    input_dir = cases / 'example/consistency/windSandShockTube'
    result_dir = cases / 'example/results/consistency/windSandShockTube'
    put(cases, base+'/assets/build/clean_validation_solver.sh', '#!/bin/sh\nexit 0\n', True)
    put(cases, base+'/assets/build/build_validation_solver.sh', '#!/bin/sh\nexit 0\n', True)
    marker = result_dir/'ran'
    put(result_dir, '.validation-bin/windSandUGKP', '#!/bin/sh\necho run > "'+str(marker)+'"\n', True)
    put(input_dir, 'constant/polyMesh/points', 'mesh')
    particle = put(input_dir, '0/gpuResidentStrictParticles.dat', header+'\n0 0 0 7 9 8 1 0 0 0 0 0 0\n')
    original = particle.read_bytes()
    result = run(tree, script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.is_file()
    assert particle.read_bytes() == (result_dir/'0/gpuResidentStrictParticles.dat').read_bytes() == original
    audit = subprocess.run([str(audit_tool), '--check'], env=env, capture_output=True, text=True)
    assert audit.returncode != 0


@pytest.mark.parametrize('script,case', [('Allrun01','01_single_phase_nozzle/flux/tadmor'), ('Allrun02','02_two_phase_validation/sparse')])
def test_performance_launch_does_not_scan_or_reject_other_compute_pids(tree, script, case):
    root, env = tree
    cases = Path(env['UGKP_CASES_ROOT'])
    script = copy_file(cases, 'example/tools/performance/'+script, source=CASES_ROOT)
    case_dir = cases/'example/performance'/case
    case_dir.mkdir(parents=True)
    if script.name == 'Allrun02':
        (case_dir/'0').mkdir()
        put(case_dir, 'system/controlDict', 'startTime 0;\n')
    bin_dir=Path(env['CXX']).parent
    marker=root/'solver-ran'
    put(bin_dir,'gasUGKP','#!/bin/sh\necho ran > "'+str(marker)+'"\n',True)
    put(bin_dir,'blockMesh','#!/bin/sh\nexit 0\n',True)
    scan=root/'gpu-scan'
    put(bin_dir,'nvidia-smi','#!/bin/sh\necho scan > "'+str(scan)+'"\necho 12345\n',True)
    result=run(tree,script,case.split('/')[-1])
    assert result.returncode==0,result.stdout+result.stderr
    assert marker.is_file()
    assert not scan.exists()
    put(bin_dir,'gasUGKP','#!/bin/sh\nexit 27\n',True)
    result=run(tree,script,case.split('/')[-1])
    assert result.returncode==1,result.stdout+result.stderr


@pytest.mark.parametrize('app', PUBLIC_SCRIPTS)
def test_direct_public_build_does_not_scan_link_policy_and_propagates_wmake(tree, app):
    result=run(tree,PUBLIC_SCRIPTS[app])
    assert result.returncode==0,result.stdout+result.stderr
    assert (Path(tree[1]['FOAM_USER_APPBIN'])/app).is_file()
    assert not any(name=='ldd' for name,_,_ in calls(tree))
    assert run(tree,PUBLIC_SCRIPTS[app],FAIL_TOOL='wmake').returncode!=0
    (Path(tree[1]['FOAM_USER_APPBIN'])/app).unlink()
    assert run(tree,PUBLIC_SCRIPTS[app],MISSING_TOOL='wmake').returncode!=0


@pytest.mark.parametrize('species', ['0','+2'])
@pytest.mark.parametrize('path', [SCRIPTS['gasUGKP'], PUBLIC_SCRIPTS['gasUGKP'], SCRIPTS['CHMT']])
def test_defined_species_count_is_passed_to_actual_compiler_without_positivity_admission(tree, species, path):
    result=run(tree,path,UGKWP_GAS_SPECIES=species)
    assert result.returncode==0,result.stdout+result.stderr
    recorded=calls(tree)
    if path==SCRIPTS['gasUGKP']:
        assert any('-DUGKWP_GAS_SPECIES='+species in args for name,args,_ in recorded if name=='nvcc')
    elif path==SCRIPTS['CHMT']:
        assert any('-DUGKWP_GAS_SPECIES='+species in args for name,args,_ in recorded if name=='g++')
        assert json.loads((Path(tree[1]['CHMT_BUILD'])/'generated/build.json').read_text())['Ns']==int(species)
    else:
        assert any(name=='wmake' for name,_,_ in recorded)


@pytest.mark.parametrize('header,extra', [('UGKP_PARTICLES_SCHEMA2 97',['extra','columns']),('UGKP_PARTICLES_SCHEMA4 97',[]),('UGKP_PARTICLES_SCHEMA4 1',['extra'])])
def test_explicit_particle_preparation_uses_actual_fields_preserving_metadata(tree, header, extra):
    root,env=tree
    cases=Path(env['UGKP_CASES_ROOT'])
    base='example/consistency/windSandShockTube'
    tool=copy_file(cases,'example/tools/consistency/windSandShockTube/assets/initialization/prepare_axial_particles.py',source=CASES_ROOT)
    row=['0','0','0','7','9','8','1','0','0','0','0','0','0',*extra]
    particle=put(cases,base+'/0/gpuResidentStrictParticles.dat',header+'\n'+' '.join(row)+'\n')
    result=subprocess.run([str(tool)],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    lines=particle.read_text().splitlines()
    expected=row[:];expected[4:6]=['0','0']
    assert lines[0]==header and lines[1].split()==expected


@pytest.mark.parametrize('fault',['truncated','missing'])
def test_actual_missing_particle_fields_or_file_still_fails_preparation(tree,fault):
    root,env=tree
    cases=Path(env['UGKP_CASES_ROOT'])
    base='example/consistency/windSandShockTube'
    tool=copy_file(cases,'example/tools/consistency/windSandShockTube/assets/initialization/prepare_axial_particles.py',source=CASES_ROOT)
    if fault=='truncated':
        particle=put(cases,base+'/0/gpuResidentStrictParticles.dat','LEGACY_METADATA 9\n0 0 0 7 9\n')
        original=particle.read_bytes()
    result=subprocess.run([str(tool)],env=env,capture_output=True,text=True)
    assert result.returncode!=0
    if fault=='truncated':
        assert 'IndexError' in result.stderr and particle.read_bytes()==original
    else:assert 'FileNotFoundError' in result.stderr
