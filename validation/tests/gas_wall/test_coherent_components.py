"""Checkout-only component coverage; no external captures or pytest dependency."""
from pathlib import Path
import subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[3]
class CoherentComponents(unittest.TestCase):
 def test_energy_identity_profiles_and_capacity_status(self):
  for name in ['coherent_energy_template_probe.cpp','coherent_profile_team_probe.cpp','coherent_engine_capacity_probe.cpp']:
   with self.subTest(probe=name),tempfile.TemporaryDirectory() as d:
    exe=Path(d)/'probe';source=Path(__file__).with_name(name)
    b=subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(ROOT),str(source),'-o',str(exe)],capture_output=True,text=True);self.assertEqual(b.returncode,0,b.stderr)
    r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=120);self.assertEqual(r.returncode,0,r.stdout+r.stderr);print(r.stdout,end='')
if __name__=='__main__':unittest.main()
