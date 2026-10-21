"""Self-contained CPU production route regression; works directly or under pytest."""
from pathlib import Path
import tempfile,subprocess,unittest
ROOT=Path(__file__).resolve().parents[3]
class CoherentRoute(unittest.TestCase):
 def test_recorded_warm_root_and_pooled_history(self):
  with tempfile.TemporaryDirectory() as d:
   exe=Path(d)/'route';src=Path(__file__).with_name('coherent_production_route_probe.cpp')
   b=subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True);self.assertEqual(b.returncode,0,b.stderr)
   for lanes in [1,4,7,32]:
    r=subprocess.run([str(exe),str(lanes)],capture_output=True,text=True,timeout=120);self.assertEqual(r.returncode,0,r.stdout+r.stderr);print(r.stdout,end='')
if __name__=='__main__':unittest.main()
