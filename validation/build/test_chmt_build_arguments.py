"""Exercise the real CHMT entrypoint with compiler-output stubs, no GPU needed."""
from pathlib import Path
import json, os, shutil, subprocess
import pytest
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('shared,override,expected', [
    (None, None, 'sm_89'), ('sm_80', None, 'sm_80'), ('sm_80', 'sm_75', 'sm_75')])
def test_chmt_architecture_defaults_reach_every_cuda_unit(tmp_path, shared, override, expected):
    source = tmp_path / 'source'
    for relative in ['applications/CHMT/Allwmake', 'tool/CHMT/build_info.py', 'tool/cuda_build_flags.sh']:
        p = source / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, p)
    tools = tmp_path / 'tools'
    tools.mkdir()
    stub = r"""#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['ARG_LOG'],'a') as f:f.write(json.dumps([Path(sys.argv[0]).name,args])+'\n')
if '--version' in args:print('test compiler');sys.exit(0)
p=Path(args[args.index('-o')+1]);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('test output');p.chmod(0o755)
"""
    for name in ['nvcc', 'cxx']:
        p = tools / name
        p.write_text(stub)
        p.chmod(0o755)
    env = dict(os.environ, NVCC=str(tools / 'nvcc'), CXX=str(tools / 'cxx'),
               CUDA_HOME=str(tmp_path / 'cuda'), WM_PROJECT_DIR=str(tmp_path / 'foam'),
               FOAM_LIBBIN=str(tmp_path / 'lib'), FOAM_MPI='dummy',
               FOAM_USER_APPBIN=str(tmp_path / 'bin'), ARG_LOG=str(tmp_path / 'args.jsonl'))
    for key in ['CHMT_BUILD', 'CHMT_CUDA_ARCH', 'UGKWP_CUDA_ARCH', 'UGKWP_CUDA_FAST_COMPILE']:
        env.pop(key, None)
    if shared: env['UGKWP_CUDA_ARCH'] = shared
    if override: env['CHMT_CUDA_ARCH'] = override
    result = subprocess.run(['bash', str(source / 'applications/CHMT/Allwmake')],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / 'args.jsonl').read_text().splitlines()]
    cuda = [args for name, args in calls if name == 'nvcc' and '-c' in args]
    assert cuda
    assert all('-arch=' + expected in args and '--Ofast-compile=min' in args for args in cuda)
    manifest = json.loads((source / 'applications/CHMT/.build/native/generated/build.json').read_text())
    assert manifest['cuda_arch'] == expected
    assert (tmp_path / 'bin/CHMT').is_file()
