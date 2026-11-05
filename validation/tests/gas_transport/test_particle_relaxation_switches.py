"""Shared production relaxation: absent inactive snapshot and active-law oracle."""
import math
import os
from pathlib import Path
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path(__file__).with_name("particle_relaxation_switches.cu")


def build(tmp_path,native,cuda,root=ROOT):
    exe=tmp_path/(f"relaxation-{native}-"+("cuda" if cuda else "cpu"))
    includes=["-I"+str(root/"common"),"-I"+str(root/"common/gasNumerics")]
    defines=[f"-DGPU_THERMAL_RELAX_NATIVE_ORDER={native}"]
    if cuda:
        nvcc=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        if not Path(nvcc).exists():pytest.skip("native CUDA toolkit required")
        command=[nvcc,"-std=c++17","--expt-relaxed-constexpr","-O2","--fmad=false","-arch=sm_89"]
    else:command=["g++","-x","c++","-std=c++17","-O0","-ffp-contract=off"]
    result=subprocess.run(command+includes+defines+[str(SOURCE),"-o",str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return exe


def observe(exe,active_only=False):
    result=subprocess.run([str(exe)]+(["--active-only"] if active_only else []),capture_output=True,text=True)
    if result.returncode==77:pytest.skip("CUDA device required")
    assert result.returncode==0,result.stdout+result.stderr
    return [[float(v) for v in line.split()] for line in result.stdout.splitlines()]


def oracle(rows):
    for mode,stuck,tg,cp,px,py,pz,tp,theta in rows:
        mode=int(mode)
        initial=(0.,0.,0.) if stuck else (1.,2.,-1.)
        gas=(3.,-2.,1.)
        speed=math.sqrt(sum((g-p)**2 for g,p in zip(gas,initial)))
        mu=1.8e-5*(tg/300)*math.sqrt(tg/300)*(300+110.4)/(tg+110.4)
        re=1.2*.0002*speed/mu
        velocity=initial
        if mode&1 and not stuck:
            cd=24/re*(1+.15*re**.687) if re<1000 else .44
            rate=.75*cd*1.2*speed/(2000*.0002)
            alpha=math.exp(-rate*1e-4)
            velocity=tuple(g+(p-g)*alpha for g,p in zip(gas,initial))
        temperature=250.
        if mode&2:
            nu=2+.6*math.sqrt(re)*(.72**(1/3))
            rate=6*nu*(mu*1000/.72)/(2000*cp*.0002**2)
            temperature=tg+(250.-tg)*math.exp(-rate*1e-4)
        assert (px,py,pz)==pytest.approx(velocity,rel=3e-14,abs=1e-14)
        assert tp==pytest.approx(temperature,rel=3e-14)
        assert theta==(.05*.75 if mode&1 else .05)


@pytest.mark.parametrize("native",[0,1])
@pytest.mark.parametrize("cuda",[False,True],ids=["host","cuda"])
def test_inactive_models_need_no_temperature_and_active_laws_match(tmp_path,native,cuda):
    rows=observe(build(tmp_path,native,cuda))
    assert len(rows)==24
    oracle(rows)


@pytest.mark.parametrize("native",[0,1])
@pytest.mark.parametrize("cuda",[False,True],ids=["host","cuda"])
def test_optional_active_exact_baseline(tmp_path,native,cuda):
    baseline=os.environ.get("UGKWP_RELAXATION_BASELINE_ROOT")
    if not baseline:pytest.skip("set UGKWP_RELAXATION_BASELINE_ROOT to the preserved baseline checkout")
    before=tmp_path/"before";before.mkdir()
    a=observe(build(before,native,cuda,Path(baseline)),True)
    b=observe(build(tmp_path,native,cuda),True)
    assert a==b
    oracle(b)
