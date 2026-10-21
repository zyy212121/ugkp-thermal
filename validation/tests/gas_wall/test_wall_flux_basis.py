from pathlib import Path
import subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[3]
class WallFluxBasis(unittest.TestCase):
 def test_repeated_auxiliary_fluxes_reuse_only_unchanged_wall_coefficients(self):
  with tempfile.TemporaryDirectory()as d:
   exe=Path(d)/'average';source=Path(__file__).with_name('wall_flux_basis_probe.cpp')
   b=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-pthread','-I'+str(ROOT),str(source),'-o',str(exe)],capture_output=True,text=True)
   self.assertEqual(b.returncode,0,b.stderr)
   p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=120);self.assertEqual(p.returncode,0,p.stdout+p.stderr);print(p.stdout,end='')
if __name__=='__main__':unittest.main()
