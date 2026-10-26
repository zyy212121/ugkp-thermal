"""Production CUDA molecular viscosity + BVP constitutive operators, no shims."""
from pathlib import Path
import math
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[3]
PARAMETERS=((1.8e-5,300,110.4,.028),(2.1e-5,300,127,.032),(8.9e-6,300,72,.002))
def reference(T,Y):
    mu=[a*(T/b)**1.5*(b+c)/(T+c) for a,b,c,m in PARAMETERS]
    mole=[y/p[3] for y,p in zip(Y,PARAMETERS)]
    return sum(x*mu[i]/sum(z*(1+math.sqrt(mu[i]/mu[j])*(PARAMETERS[j][3]/PARAMETERS[i][3])**.25)**2/math.sqrt(8*(1+PARAMETERS[i][3]/PARAMETERS[j][3])) for j,z in enumerate(mole)) for i,x in enumerate(mole) if x)

def test_production_cuda_viscosity_and_wall(tmp_path):
    nvcc=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    if not Path(nvcc).exists(): pytest.skip('native CUDA toolkit required')
    executable=tmp_path/'molecular_viscosity'
    subprocess.run([nvcc,'-std=c++17','--expt-relaxed-constexpr','-O2','--fmad=false','-arch=sm_89','-I'+str(ROOT/'common'),str(Path(__file__).with_name('molecular_viscosity.cu')),'-o',str(executable)],check=True)
    result=subprocess.run([str(executable)],text=True,capture_output=True,check=True)
    rows=result.stdout.splitlines()
    assert len(rows)==25
    for row in rows[:24]:
        _,T,y0,y1,y2,mu,const=row.split()
        assert float(mu)==pytest.approx(reference(float(T),list(map(float,(y0,y1,y2)))),rel=3e-14)
        assert float(const)==7.25e-6
    _,a,b,traction,heat,ok=rows[-1].split()
    ma,mb=reference(300,[.2,.3,.5]),reference(900,[.6,.1,.3])
    assert float(a)==pytest.approx(ma,rel=3e-14)
    assert float(b)==pytest.approx(mb,rel=3e-14)
    # Existing positive linear-coefficient integral: logarithmic mean.
    conductance=(mb-ma)/math.log(mb/ma)
    assert float(traction)==pytest.approx(conductance*10/.01,rel=3e-13)
    cp=sum(y*c for y,c in zip([.4,.2,.4],[1000,1400,1800]))
    assert float(heat)==pytest.approx(-conductance*cp/.72*600/.01,rel=3e-13)
    assert ok=='1'


def test_actual_cuda_ambient_gas_drag_and_heat(tmp_path):
    nvcc=shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    if not Path(nvcc).exists():pytest.skip('native CUDA toolkit required')
    exe=tmp_path/'particle'
    subprocess.run([nvcc,'-std=c++17','--expt-relaxed-constexpr','-O2','--fmad=false','-arch=sm_89','-I'+str(ROOT/'common'),str(Path(__file__).with_name('molecular_particle.cu')),'-o',str(exe)],check=True)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    for line in result.stdout.splitlines():
        T,mu,beta,temperature,kappa,direct=map(float,line.split())
        expected=PARAMETERS[0][0]*(T/300)**1.5*(300+110.4)/(T+110.4)
        Re=1.2*1e-4*3/expected
        cd=24/Re*(1+.15*Re**.687)
        expected_beta=.75*cd*1.2*3/(2000*1e-4+1e-15)
        nu=2+.6*math.sqrt(Re)*.72**(1/3)
        rate=6*nu*(expected*1000/.72)/(2e6*1e-8+1e-300)
        assert mu==pytest.approx(expected,rel=2e-14)
        assert beta==pytest.approx(expected_beta,rel=2e-14)
        assert temperature==pytest.approx(T+(250-T)*math.exp(-rate*1e-4),rel=2e-14)
        assert kappa==pytest.approx(expected*1000/.72,rel=2e-14)
        assert direct==.123
