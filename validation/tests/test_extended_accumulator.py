"""Real x86 extended arithmetic oracle for GPU ordered inventory accumulation."""
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).with_name("extended_accumulator.cu")

@pytest.mark.parametrize("cuda",[False,True],ids=["host","cuda"])
def test_ordered_extended_accumulator(tmp_path,cuda):
    if cuda:
        compiler=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        if not Path(compiler).exists():pytest.skip("CUDA toolkit required")
        command=[compiler,"-std=c++17","-O2","-arch=sm_89"]
    else:command=["g++","-x","c++","-std=c++17","-O2"]
    executable=tmp_path/"extended"
    built=subprocess.run(command+["-I"+str(ROOT/"common"),str(SOURCE),"-o",str(executable)],capture_output=True,text=True)
    assert built.returncode==0,built.stdout+built.stderr
    result=subprocess.run([str(executable)],capture_output=True,text=True)
    if result.returncode==77:pytest.skip("native CUDA or x86 binary80 oracle unavailable")
    assert result.returncode==0,result.stdout+result.stderr
