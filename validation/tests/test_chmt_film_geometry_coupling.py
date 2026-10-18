"""One physical material update, with a consistent same-dt film/geometry RHS."""
from pathlib import Path
import subprocess,os,shlex,shutil,tempfile,unittest,json
ROOT=Path(__file__).resolve().parents[2];FIX=Path(__file__).parent/'fixtures/film_geometry_coupling'
WRAPPER=r"""
#include "film/CpuFilmDriver.H"
#include <iostream>
#include <iomanip>
namespace chmt {
bool observedFilmTransport(const HostState&s,const PhysicsConfig&p,Real dt,const std::vector<CpuFilmDrive>& d,std::vector<FilmQ>& r,std::string&e,std::vector<FilmAux>*a=nullptr,std::vector<Vec3>*g=nullptr,std::vector<Real>*v=nullptr,bool*n=nullptr){
 bool ok=evaluateCpuFilmTransport(s,p,dt,d,r,e,a,g,v,n);
 if(ok)for(size_t f=0;f<r.size();++f){std::cout<<std::setprecision(17)<<"RATE "<<f<<' '<<r[f].mass<<' '<<r[f].enthalpy;for(int j=0;j<Ns;++j)std::cout<<' '<<r[f].species[j];std::cout<<' '<<s.surface.area[f]<<'\n';}
 return ok;
}
bool observedFilmAdvance(const HostState&s,const PhysicsConfig&p,Real dt,const std::vector<CpuFilmForcing>&f,const std::vector<CpuFilmDrive>&d,CpuFilmCandidate&o,std::string&e,bool*n=nullptr){std::cout<<"PHYSICAL_FILM_UPDATE\n";return advanceCpuFilmCandidate(s,p,dt,f,d,o,e,n);}
}
"""
class FilmGeometryCoupling(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  foam=Path(os.environ.get('WM_PROJECT_DIR','/opt/openfoam10'))
  if not (foam/'etc/bashrc').is_file():raise unittest.SkipTest('real OpenFOAM10 unavailable')
  cls.tmp=tempfile.TemporaryDirectory(prefix='ugkp-film-consistency-');cls.addClassCleanup(cls.tmp.cleanup)
  out=Path(cls.tmp.name);shutil.copytree(FIX/'case',out/'case');cls.out=out
  ref=os.environ.get('UGKP_MATH_TEST_DRIVER_REF')
  production=subprocess.check_output(['git','show',ref+':applications/CHMT/materials/CpuMaterialDriver.C'],cwd=ROOT,text=True) if ref else (ROOT/'applications/CHMT/materials/CpuMaterialDriver.C').read_text()
  source=production.replace('evaluateCpuFilmTransport(','observedFilmTransport(').replace('advanceCpuFilmCandidate(','observedFilmAdvance(')
  (out/'driver.C').write_text(WRAPPER+source)
  q=shlex.quote;app=ROOT/'applications/CHMT'
  env='source '+q(str(foam/'etc/bashrc'))+'\nset -eo pipefail\n'
  env+='export FOAM_USER_APPBIN='+q(str(out/'bin'))+' FOAM_USER_LIBBIN='+q(str(out/'lib'))+' CUDA_CACHE_PATH='+q(str(out/'cache/cuda'))+' TMPDIR='+q(str(out/'tmp'))+' XDG_CACHE_HOME='+q(str(out/'cache'))+'\nmkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN" "$CUDA_CACHE_PATH" "$TMPDIR"\n'
  env+='export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$FOAM_LIBBIN/$FOAM_MPI:$FOAM_LIBBIN:$FOAM_LIBBIN/dummy"\n'
  cmd='g++ -std=c++17 -O1 -ffp-contract=off -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -DUGKWP_GAS_SPECIES=2 -I'+q(str(app))+' -I'+q(str(ROOT/'common'))+' '
  cmd+=' '.join('-I$WM_PROJECT_DIR/src/'+i+'/lnInclude' for i in ['OpenFOAM','finiteVolume','meshTools','OSspecific/POSIX'])+' '
  cmd+=' '.join(q(str(p)) for p in [FIX/'probe.C',out/'driver.C',*[app/i for i in ['materials/MaterialTransport.C','film/CpuFilmDriver.C','ablation/CpuSurfaceInterface.C','mesh/Geometry.C']]])
  cmd+=' -L$FOAM_LIBBIN -lfiniteVolume -lmeshTools -lOpenFOAM -L$FOAM_LIBBIN/$FOAM_MPI -lPstream -o '+q(str(out/'probe'))+'\n'
  script=out/'build.sh';script.write_text(env+cmd);run=subprocess.run(['bash',str(script)],text=True,capture_output=True)
  cls.records=[{'command':['bash',str(script)],'body':script.read_text(),'exit':run.returncode}];cls.logs={'build.log':run.stdout+run.stderr}
  if run.returncode:raise RuntimeError(cls.logs['build.log'][-8000:])
  cls.results={}
  for mode,dt in [('unselected','.001'),('unselected','.0005'),('unselected','.00025'),('coupled','.001')]:
   sh=out/('run-'+mode+'-'+dt+'.sh');sh.write_text(env+q(str(out/'probe'))+' -case '+q(str(out/'case'))+' -step '+dt+' -phaseCase '+mode+'\n')
   p=subprocess.run(['bash',str(sh)],text=True,capture_output=True);cls.records.append({'command':['bash',str(sh)],'body':sh.read_text(),'exit':p.returncode});text=p.stdout+p.stderr;cls.logs[mode+'-'+dt+'.log']=text
   rates={};success=False
   for line in text.splitlines():
    if line.startswith('RATE '):values=line.split();rates.setdefault(int(values[1]),[]).append([float(x) for x in values[2:]])
    elif line.startswith('RESULT '):success='success=1 ' in line and 'steps=1 ' in line
   cls.results[mode,dt]={'exit':p.returncode,'success':success,'rates':rates,'updates':text.count('PHYSICAL_FILM_UPDATE')}
  dest=os.environ.get('UGKP_MATH_TEST_OUTPUT')
  if dest:
   d=Path(dest);d.mkdir(parents=True,exist_ok=True)
   for name,text in cls.logs.items():(d/name).write_text(text)
   (d/'commands.json').write_text(json.dumps(cls.records,indent=2));(d/'results.json').write_text(json.dumps({str(k):v for k,v in cls.results.items()},indent=2));shutil.copy2(out/'driver.C',d/'driver.C')

 def test_given_geometry_uses_endpoint_transport_from_first_evaluation(self):
  for dt in ['.001','.0005','.00025']:
   r=self.results['unselected',dt];self.assertEqual(r['exit'],0);self.assertTrue(r['success'],self.logs['unselected-'+dt+'.log']);self.assertEqual(r['updates'],1)
   for face,seq in r['rates'].items():
    self.assertGreaterEqual(len(seq),2)
    self.assertLessEqual(float(dt)*abs(seq[-1][0]-seq[0][0]),1e-10,'given geometry must be used before interface/phase evaluation')

 def test_recession_converges_same_dt_without_replaying_material(self):
  r=self.results['coupled','.001'];self.assertEqual(r['exit'],0);self.assertTrue(r['success'],self.logs['coupled-.001.log']);self.assertEqual(r['updates'],1)
  for seq in r['rates'].values():
   self.assertGreaterEqual(len(seq),2)
   self.assertLessEqual(.001*abs(seq[-1][0]-seq[-2][0]),1e-10)
   self.assertLessEqual(.001*abs(seq[-1][1]-seq[-2][1]),1e-10+1e-10*1e6)

if __name__=='__main__':unittest.main()
