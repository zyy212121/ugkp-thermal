from pathlib import Path
import subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[3]
class OrderedProfileAverage(unittest.TestCase):
 def test_parallel_point_evaluation_preserves_serial_quadrature_bits(self):
  with tempfile.TemporaryDirectory()as d:
   exe=Path(d)/'average';source=Path(__file__).with_name('ordered_profile_average_probe.cpp')
   b=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-pthread','-I'+str(ROOT),str(source),'-o',str(exe)],capture_output=True,text=True)
   self.assertEqual(b.returncode,0,b.stderr)
   p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=120);self.assertEqual(p.returncode,0,p.stdout+p.stderr);print(p.stdout,end='')
if __name__=='__main__':unittest.main()
