"""Exact original public radiation results, including nontrapping IEEE outputs.

This exercises real OpenFOAM production TUs; no GPU execution is claimed.
"""
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[4]
ORIGINAL='d6110c02745a8ad726208a426c4bb3a28233c1fd'
CURRENT=Path(os.environ.get('C04_CURRENT_ROOT',ROOT))
spec=importlib.util.spec_from_file_location('c04_radiation_cases',Path(__file__).with_name('test_radiation_continuation.py'))
cases=importlib.util.module_from_spec(spec);spec.loader.exec_module(cases)
case=cases.case

@pytest.fixture(scope='module')
def original(tmp_path_factory):
 p=tmp_path_factory.mktemp('original-radiation')
 raw=subprocess.check_output(['git','archive',ORIGINAL,'common','applications/CHT/thermal','applications/CHT/gpu'],cwd=ROOT)
 with tarfile.open(fileobj=io.BytesIO(raw)) as archive:archive.extractall(p,filter='data')
 return p

@pytest.fixture(scope='module')
def probes(tmp_path_factory,original):
 if not os.environ.get('WM_PROJECT_DIR'):pytest.skip('source real OpenFOAM')
 result=[]
 reader=(ROOT/'tests/fixtures/particle_compatibility/thermal_readers.cpp').read_text().split('int main()',1)[0]
 body=(Path(__file__).parent/'radiationOriginalSuccess/TestRadiationOriginalSuccess.C').read_text().replace('// C04_READER_ADAPTER',reader)
 for label,root in [('original',original),('current',CURRENT)]:
  p=tmp_path_factory.mktemp(label+'-radiation-build');(p/'Make').mkdir();(p/'probe.C').write_text(body)
  thermal=root/'applications/CHT/thermal';omp='-fopenmp' if os.environ.get('T13_OPENMP') else ''
  (p/'Make/files').write_text(f'probe.C\n{thermal}/MieTable.C\n{thermal}/MieDoRadiationSolver.C\n{thermal}/GpuParticleRadiationCoupler.C\nEXE = {p}/probe\n')
  (p/'Make/options').write_text(f'EXE_INC = {omp} -std=c++17 -I{thermal} -I{root}/common -I{root}/common/wall -I{root}/applications/CHT/gpu -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = {omp} -lfiniteVolume -lmeshTools -lOpenFOAM\n')
  r=subprocess.run(['wmake'],cwd=p,capture_output=True,text=True);(p/'build.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
  result.append(p/'probe')
 return result

def table(path,change='ordinary'):
 cases.table(path);s=path.read_text()
 if change=='reader':s=s.replace('Tvals 300 4000','Tvals 300 3500')
 if change=='zero_diameter':s=s.replace('Dvals 0.0001 0.0002','Dvals 0 0.0002')
 if change=='negative_diameter':s=s.replace('Dvals 0.0001 0.0002','Dvals -0.0002 -0.0001')
 if change=='equal_logarithms':s=s.replace('Dvals 0.0001 0.0002','Dvals 1e100 1.0000000000000002e100')
 if change=='wide_temperature':s=s.replace('Tvals 300 4000','Tvals -1e308 1e308')
 # Keep actual original payload identity valid, so admission cannot mask arithmetic.
 begin=s.index('Tvals ');end=s.index('END_MIE_TABLE\n')+len('END_MIE_TABLE\n');s=s[:end]+'payloadSha1 '+hashlib.sha1(s[begin:end].encode()).hexdigest()+'\n';path.write_text(s);return path

def run(probe,mode,tmp_path,path=None,mesh=None,t='4100',mu='0',d='0.00015',extra=None):
 env=dict(os.environ,C04_MODE=mode,C04_T=t,C04_D=d,C04_MU=mu,FOAM_SIGFPE='false')
 if extra:env.update(extra)
 if path:env['C04_TABLE']=str(path)
 args=[str(probe)]
 if mesh:args+=['-case',str(mesh)]
 r=subprocess.run(args,env=env,capture_output=True,text=True)
 name=probe.parent.name+'-'+mode
 (tmp_path/(name+'.log')).write_text(r.stdout+r.stderr)
 return r

def equal_success(probes,mode,tmp_path,**kwargs):
 results=[run(p,mode,tmp_path,**kwargs) for p in probes]
 assert results[0].returncode==0,results[0].stdout+results[0].stderr
 assert results[1].returncode==0,results[1].stdout+results[1].stderr
 data=[bytes('\n'.join(x for x in r.stdout.splitlines() if x.startswith('DATA ')),'ascii') for r in results]
 assert data[0] and data[0]==data[1]

@pytest.mark.parametrize('mode',['scale_ordinary','scale_tiny','scale_zero'])
def test_original_boundary_helper_success(probes,tmp_path,mode):equal_success(probes,mode,tmp_path)

@pytest.mark.parametrize('mode',['ordinary','above','below','reader','cancellation','inactive'])
def test_original_complete_transaction_success(probes,tmp_path,case,mode):
 equal_success(probes,mode,tmp_path,path=table(tmp_path/'table',mode),mesh=case)

def test_original_parallel_transaction_success(probes,tmp_path,case):
 if not os.environ.get('T13_OPENMP'):pytest.skip('separate real OpenMP build')
 equal_success(probes,'parallel',tmp_path,path=table(tmp_path/'table'),mesh=case)

@pytest.mark.parametrize('change,t,mu',[('ordinary','4100','0'),('ordinary','inf','0'),('ordinary','-inf','0'),('ordinary','4100','inf'),('ordinary','4100','-inf'),('zero_diameter','4100','0'),('negative_diameter','4100','0'),('wide_temperature','4100','0'),('equal_logarithms','4100','0')])
def test_original_public_table_query_success(probes,tmp_path,change,t,mu):
 equal_success(probes,'query',tmp_path,path=table(tmp_path/'table',change),t=t,mu=mu)

@pytest.mark.parametrize('change',['zero_diameter','negative_diameter','equal_logarithms'])
def test_original_downstream_do_still_refuses_nonfinite_table(probes,tmp_path,case,change):
 for probe in probes:
  r=run(probe,'do',tmp_path,path=table(tmp_path/'table',change),mesh=case)
  assert r.returncode!=0 and 'FAIL ' in r.stderr

def generator(root,name):
 s=importlib.util.spec_from_file_location(name,root/'applications/CHT/thermal/mieTables/make_alumina_mie_table.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

@pytest.fixture(scope='module')
def generators(original):return generator(original,'c04_original_generator'),generator(CURRENT,'c04_current_generator')

@pytest.mark.parametrize('values',[[1.,2.,1.],[1e-310,2e-310,1e-310],[0.,0.,0.],[-1.,-2.,-1.],[float('inf')]*3,[float('nan')]*3])
def test_original_phase_normalization_fallback(generators,values):
 mu=np.array([-1.,0.,1.]);a,b=[m.normalize_phi(np.array(values),mu) for m in generators];assert a.tobytes()==b.tobytes()

@pytest.mark.parametrize('n',[-1,0,1,4])
def test_original_logspace_helper(generators,n):
 a,b=[m.logspace(1.,2.,n) for m in generators];assert a.tobytes()==b.tobytes()

def test_original_compute_node_tiny_scattering_success(generators):
 outputs=[]
 for m in generators:
  lam=np.array([1.,1.0001]);T=m.C2/670.;mu=np.linspace(-1,1,17)
  weights=[.5*x*np.log(lam[-1]/lam[0])*m.planck_weight_lambda(x,T) for x in lam]
  qs=[m.mie_coefficients(np.pi*1e-6/x,m.alumina_refractive_index(x,T))[1] for x in lam]
  assert sum(weights)>1e-300 and 0<sum(w*q for w,q in zip(weights,qs))<=1e-300
  values=m.compute_node(1e-6,T,mu,lam)
  outputs.append(np.concatenate([np.asarray(v).reshape(-1) for v in values]).tobytes())
 assert outputs[0]==outputs[1]


@pytest.mark.parametrize('above',[False,True])
def test_original_boundary_identity_threshold(probes,tmp_path,above):
 tau=1e-10+4096*np.finfo(float).eps
 raw=np.nextafter(tau,np.inf if above else 0.)
 equal_success(probes,'scale_custom',tmp_path,extra={'C04_A':repr(float(-raw/2)),'C04_B':repr(float(raw)),'C04_G':'1','C04_TOL':'1e-10'})

def test_original_boundary_identity_nontrapping_overflow(probes,tmp_path):
 equal_success(probes,'scale_custom',tmp_path,extra={'C04_A':'1e308','C04_B':'1e308','C04_G':'1e308','C04_TOL':'10'})

@pytest.mark.parametrize('diameter',['inf','-inf'])
def test_original_public_diameter_endpoint(probes,tmp_path,diameter):
 equal_success(probes,'query',tmp_path,path=table(tmp_path/'table'),d=diameter)

@pytest.mark.parametrize('qsca',[0.,1e-310,-1.])
def test_original_phase_scattering_fallback(generators,qsca):
 outputs=[]
 for m in generators:
  _,_,_,_,an,bn=m.mie_coefficients(.4,complex(1.5,-.01))
  outputs.append(m.mie_phase_raw(.4,qsca,an,bn,np.linspace(-1,1,17)).tobytes())
 assert outputs[0]==outputs[1]
