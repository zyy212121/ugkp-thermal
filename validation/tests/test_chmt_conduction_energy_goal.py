"""True OF10 nativeConduction regression: finite tiny updates must not hide energy error.

Build the actual anonymous production function with a test-only trace insertion.
No production branch, equation or solver setting is changed by the fixture.
"""
import json, os, pathlib, shlex, shutil, subprocess, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]
FIX=pathlib.Path(__file__).parent/'fixtures/conduction_energy'

class ConductionEnergyGoal(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  foam=pathlib.Path(os.environ.get('WM_PROJECT_DIR','/opt/openfoam10'))
  if not (foam/'etc/bashrc').is_file():raise unittest.SkipTest('real OpenFOAM 10 unavailable')
  cls.tmp=tempfile.TemporaryDirectory(prefix='ugkp-conduction-goal-');cls.addClassCleanup(cls.tmp.cleanup)
  out=pathlib.Path(cls.tmp.name);shutil.copytree(FIX/'case',out/'case')
  source=(ROOT/'applications/CHMT/materials/CpuMaterialDriver.C').read_text().split('void filmForcing(')[0]
  needle=next(line for line in source.splitlines() if line.strip().startswith('const Real energyTolerance='))
  if source.count(needle)!=1:raise RuntimeError('update trace hook for refactored nativeConduction')
  source=source.replace(needle,needle+'\n        std::cout<<"TRACE "<<iteration<<\' \'<<change<<\' \'<<residual<<\' \'<<energyTolerance<<\' \'<<linear.initialResidual()<<\' \'<<linear.finalResidual()<<\' \'<<linear.nIterations()<<\'\\n\';')
  (out/'probe.C').write_text('#include <iostream>\n#include <iomanip>\n#include <cassert>\n'+source+(FIX/'probe-main.C').read_text())
  q=shlex.quote
  script='source '+q(str(foam/'etc/bashrc'))+'\nset -eo pipefail\n'
  script+='export FOAM_USER_APPBIN='+q(str(out/'bin'))+' FOAM_USER_LIBBIN='+q(str(out/'lib'))+' CUDA_CACHE_PATH='+q(str(out/'cuda-cache'))+' XDG_CACHE_HOME='+q(str(out/'cache'))+' TMPDIR='+q(str(out/'tmp'))+'\n'
  script+='mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN" "$CUDA_CACHE_PATH" "$TMPDIR"\n'
  script+='export PATH="$FOAM_USER_APPBIN:$FOAM_APPBIN:/usr/bin:/bin"\nexport LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$FOAM_LIBBIN:$FOAM_LIBBIN/$FOAM_MPI"\n'
  script+='g++ -std=c++17 -O1 -ffp-contract=off -DNoRepository -DWM_DP -DWM_LABEL_SIZE=32 -ffunction-sections -fdata-sections '
  script+='-I'+q(str(ROOT/'applications/CHMT'))+' -I'+q(str(ROOT/'common'))+' '
  script+='-I$WM_PROJECT_DIR/src/OpenFOAM/lnInclude -I$WM_PROJECT_DIR/src/finiteVolume/lnInclude -I$WM_PROJECT_DIR/src/meshTools/lnInclude -I$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude '
  script+=q(str(out/'probe.C'))+' '+q(str(ROOT/'applications/CHMT/mesh/Geometry.C'))+' -Wl,--gc-sections -L$FOAM_LIBBIN -lfiniteVolume -lmeshTools -lOpenFOAM -L$FOAM_LIBBIN/$FOAM_MPI -lPstream -o '+q(str(out/'probe'))+'\n'
  script+=q(str(out/'probe'))+' -case '+q(str(out/'case'))+'\n'
  (out/'run.sh').write_text(script)
  run=subprocess.run(['bash',str(out/'run.sh')],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  evidence=os.environ.get('UGKP_MATH_TEST_OUTPUT')
  if evidence:
   dest=pathlib.Path(evidence);dest.mkdir(parents=True,exist_ok=True)
   (dest/'run.log').write_text(run.stdout);(dest/'run.sh').write_text(script);shutil.copy2(out/'probe.C',dest/'probe.C')
   (dest/'command.json').write_text(json.dumps({'command':['bash',str(out/'run.sh')],'exit':run.returncode,'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()},indent=2))
   if (out/'probe').exists():shutil.copy2(out/'probe',dest/'probe')
  if run.returncode:raise RuntimeError('OF10 probe build/run failed:\n'+run.stdout[-6000:])
  cls.cases=[]
  for line in run.stdout.splitlines():
   if line.startswith('CASE '):cls.cases.append({'id':int(line.split()[1]),'trace':[]})
   elif line.startswith('TRACE '):cls.cases[-1]['trace'].append([float(x) for x in line.split()[1:]])
   elif line.startswith('RESULT '):cls.cases[-1]['returned']=line.split()[1]=='1'

 def test_energy_goal_survives_small_temperature_updates(self):
  self.assertEqual(len(self.cases),27)
  failed=[(c['id'],c['trace'][-1][2],c['trace'][-1][3]) for c in self.cases if not c['returned'] or abs(c['trace'][-1][2])>c['trace'][-1][3]]
  self.assertEqual(failed,[], 'finite temperature stopping hid unmet energy balance')

 def test_linear_precision_does_not_spin_to_nonlinear_budget(self):
  capped=[c['id'] for c in self.cases if len(c['trace'])>=100]
  self.assertEqual(capped,[], 'PCG zero updates must not cause identical outer solves')

 def test_manufactured_nonconstant_capacity_counterexample(self):
  self.assertLess(abs(self.cases[12]['trace'][-1][2]),1.2e-4)
  self.assertLess(abs(self.cases[9]['trace'][-1][2]),2.1e-5)

if __name__=='__main__':unittest.main()
