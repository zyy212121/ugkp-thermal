"""Link native thermal-owner probes against the production CUDA operators."""
from pathlib import Path
import ctypes
import hashlib
import os
import shutil
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT


def link_thermal_cuda(build, source_root=SOURCE_ROOT):
    build=Path(build);source_root=Path(source_root)
    if not (source_root/"common/gpu/SparseConjugateGradient.cu").exists():
        return
    nvcc=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    if not Path(nvcc).is_file():
        pytest.skip('native thermal owners require the CUDA toolkit')
    cuda=Path(nvcc).resolve().parent.parent
    try:
        runtime=ctypes.CDLL(str(cuda/'lib64/libcudart.so'))
        count=ctypes.c_int()
        if runtime.cudaGetDeviceCount(ctypes.byref(count)) or count.value==0:
            pytest.skip('native thermal owners require a CUDA device')
    except OSError:
        pytest.skip('native CUDA runtime unavailable')
    objects=[]
    for name in ('SparseConjugateGradient','SolidThermalProperties','ImplicitDiffusion','ThermalFieldOperators'):
        obj=build/(name+'.o');objects.append(str(obj))
        command=[nvcc,'-std=c++17','-O2','--fmad=false','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),
            '-c',str(source_root/'common/gpu'/(name+'.cu')),'-o',str(obj)]
        signature=hashlib.sha256((' '.join(command)).encode())
        for dependency in sorted((source_root/'common/gpu').glob('*')):
            if dependency.is_file():signature.update(dependency.read_bytes())
        receipt=build/(name+'.sha256');digest=signature.hexdigest()
        if obj.exists() and receipt.exists() and receipt.read_text()==digest:
            continue
        result=subprocess.run(command,capture_output=True,text=True)
        (build/('cuda-'+name+'.log')).write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr
        receipt.write_text(digest)
    options=build/'Make/options'
    text='\n'.join(line for line in options.read_text().splitlines()
        if not (line.startswith('EXE_LIBS +=') and 'SparseConjugateGradient.o' in line))
    options.write_text(text+'\nEXE_LIBS += '+' '.join(objects)+' -L'+str(cuda/'lib64')+' -Wl,-rpath,'+str(cuda/'lib64')+' -lcudart\n')
