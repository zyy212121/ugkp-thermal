"""Packet rows against the preserved original x86 long-double batch function."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[4]
APP=ROOT/"applications/CHMT"
SOURCE=Path(__file__).with_name("material_packet_cuda.cu")


def run(command):
    result=subprocess.run([str(x) for x in command],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout


@pytest.mark.parametrize("species",[2,3])
@pytest.mark.parametrize("cuda",[False,True],ids=["host","cuda"])
def test_original_packet_inventory(tmp_path,species,cuda):
    baseline=Path(os.environ.get("UGKWP_MATERIAL_BASELINE_ROOT",ROOT.parent/"gpu-coupled-baseline-88bc5434"))
    if baseline.exists():source=(baseline/"applications/CHMT/materials/MaterialTransport.C").read_text()
    else:
        source=run(["git","-C",ROOT,"show","88bc5434:applications/CHMT/materials/MaterialTransport.C"]);baseline=ROOT
    start=source.index("bool applyMaterialPacketBatchImpl(");end=source.index("bool materialPointsClose(",start)
    original=('#include "materials/MaterialTransport.H"\n#include "coupling/Coordinator.H"\n#include <algorithm>\nnamespace chmt {\n'+source[start:end]+'\n}\n').replace("applyMaterialPacketBatch","baselineApplyMaterialPacketBatch")
    path=tmp_path/"baseline.C";path.write_text(original);obj=tmp_path/"baseline.o"
    run(["g++","-std=c++17","-O2","-ffp-contract=off",f"-DUGKWP_GAS_SPECIES={species}","-I"+str(baseline/"applications/CHMT"),"-I"+str(baseline/"common"),"-c",path,"-o",obj])
    common=[f"-DUGKWP_GAS_SPECIES={species}","-I"+str(APP),"-I"+str(ROOT/"common")]
    exe=tmp_path/"packets"
    if cuda:
        nvcc=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        if not Path(nvcc).exists():pytest.skip("native CUDA toolkit required")
        run([nvcc,"-std=c++17","--expt-relaxed-constexpr","--fmad=false","-O2","-arch=sm_89",*common,SOURCE,obj,"-o",exe])
    else:run(["g++","-std=c++17","-O2","-ffp-contract=off",*common,"-x","c++",SOURCE,"-x","none",obj,"-o",exe])
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    if result.returncode==77:pytest.skip("native CUDA or binary80 host required")
    assert result.returncode==0,result.stdout+result.stderr
    print(result.stdout)
