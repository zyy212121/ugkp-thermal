"""Original 88bc5434 CPU face algorithm vs shared CPU and real CUDA transport."""
from pathlib import Path
import os
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[4]
APP=ROOT/"applications/CHMT"
SOURCE=Path(__file__).with_name("material_transport_cuda.cu")


def run(command):
    result=subprocess.run([str(x) for x in command],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout


@pytest.mark.parametrize("species",[2,3])
@pytest.mark.parametrize("cuda",[False,True],ids=["host","cuda"])
def test_preserved_material_transport(tmp_path,species,cuda):
    reference=Path(os.environ.get("UGKWP_MATERIAL_BASELINE_ROOT",ROOT.parent/"gpu-coupled-baseline-88bc5434"))
    if reference.exists():source=(reference/"applications/CHMT/materials/MaterialTransport.C").read_text()
    else:
        source=run(["git","-C",ROOT,"show","88bc5434:applications/CHMT/materials/MaterialTransport.C"])
        reference=ROOT
    prefix=source[:source.index("bool applyMaterialPacketBatchImpl(")]
    start=source.index("bool evaluateMaterialTransportImpl(")
    finish=source.index("bool advanceMaterialReactionsImpl(",start)
    original=(prefix+source[start:finish]+"\n} // namespace chmt\n").replace("evaluateMaterialTransport","baselineEvaluateMaterialTransport")
    baseline=tmp_path/"baseline.C";baseline.write_text(original)
    objects=[]
    for name,path,base in (("baseline",baseline,reference),("candidate",APP/"materials/MaterialTransport.C",ROOT)):
        obj=tmp_path/(name+".o")
        run(["g++","-std=c++17","-O2","-ffp-contract=off",f"-DUGKWP_GAS_SPECIES={species}","-I"+str(base/"applications/CHMT"),"-I"+str(base/"common"),"-c",path,"-o",obj]);objects.append(obj)
    exe=tmp_path/"transport"
    common=[f"-DUGKWP_GAS_SPECIES={species}","-I"+str(APP),"-I"+str(ROOT/"common")]
    if cuda:
        nvcc=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        if not Path(nvcc).exists():pytest.skip("native CUDA toolkit required")
        run([nvcc,"-std=c++17","--expt-relaxed-constexpr","--fmad=false","-O2","-arch=sm_89",*common,SOURCE,*objects,"-o",exe])
    else:run(["g++","-std=c++17","-O2","-ffp-contract=off",*common,"-x","c++",SOURCE,"-x","none",*objects,"-o",exe])
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    if result.returncode==77:pytest.skip("CUDA device required")
    assert result.returncode==0,result.stdout+result.stderr
    print(result.stdout)
