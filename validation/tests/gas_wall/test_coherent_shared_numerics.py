"""Standalone shared-numerics checks; run directly without pytest dependencies."""
from pathlib import Path
import subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[3]
CPP=r'''
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include "common/gasWall/WallCoherentNumerics.H"
using namespace ugkwp::gaswall;
int main(){
 using namespace coherent;
 const double y=.7,hl=.1,hr=.17;
 assert(std::abs(detail::logShearDerivative(std::log(y-hl),std::log(y),std::log(y+hr),hl,hr,y)-1/y)<2e-14);
 for(int nodes:{24,48,128}) {
  WallModelConfig<double> c;c.nodes=nodes;WallWorkspace<double,2,128> a{},b{};
  for(int i=0;i<nodes;++i){a.y[i]=i;for(int j:{4,5})a.state[i][j]=i+1;}
  b=a;b.state[3][4]+=.01;assert(pchipFdCompatible(c,a,b));
  a.state[3][4]=a.state[2][4];b=a;b.state[3][4]+=.001;assert(pchipFdCompatible(c,a,b));
  b=a;b.state[3][4]-=.001;assert(pchipFdCompatible(c,a,b));
  a.state[3][4]=std::nextafter(a.state[2][4],INFINITY);b=a;b.state[3][4]=a.state[2][4];assert(!pchipFdCompatible(c,a,b));
 }
 WallModelConfig<double> c;c.nodes=5;WallWorkspace<double,3,5>a{},b{};
 for(int i=0;i<5;++i){a.y[i]=i;a.state[i][5]=a.state[i][6]=i+1;}
 a.state[2][6]=std::nextafter(a.state[1][6],INFINITY);b=a;b.state[2][6]=a.state[1][6];assert(!pchipFdCompatible(c,a,b));
 BoundedVector<double,36> matrix;matrix.assign(36,0.);BoundedVector<double,6> rhs;rhs.resize(6);
 const double expected[6]={1,-2,3,-4,5,-6};
 for(int i=0;i<6;++i)for(int j=0;j<6;++j)matrix[i*6+j]=(i==j?10.:1./(1+i+j));
 matrix[0]=.01;matrix[6]=3.;
 for(int i=0;i<6;++i){rhs[i]=0;for(int j=0;j<6;++j)rhs[i]+=matrix[i*6+j]*expected[j];}
 const auto original=rhs;CachedLU<1> lu;assert(lu.factor(matrix,6));lu.solve(rhs);
 for(int i=0;i<6;++i)assert(std::abs(rhs[i]-expected[i])<1e-13);
 rhs=original;lu.solve(rhs);for(int i=0;i<6;++i)assert(std::abs(rhs[i]-expected[i])<1e-13);
 std::cout<<"shared numerics pass: capacities24/48/128, N=3 indices, tie/ULP, pivot/reuse\n";
}
'''
CPP_FLAG=r'''
#include <cassert>
#include <cmath>
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
int main(){using namespace ugkwp::gaswall;CapturedSstWall m;auto c=m.config;c.enableSst=true;
 const double y=.7,hl=.1,hr=.17;detail::LayerPoint<double,2> l,p,r;p.rho=1;p.k=1;p.omega=1000;p.z=1/std::sqrt(p.omega);l=p;r=p;
 for(int j=0;j<2;++j){l.u[j]=(j+1)*std::log(y-hl);p.u[j]=(j+1)*std::log(y);r.u[j]=(j+1)*std::log(y+hr);}
 auto old=p,exact=p,coherent=p;
 detail::applySstGradients(m.input,y,detail::centeredDerivative(l.u[0],p.u[0],r.u[0],hl,hr),detail::centeredDerivative(l.u[1],p.u[1],r.u[1],hl,hr),0.,0.,0.,old);
 detail::pointSst(m.input,c,y,l,r,hl+hr,p,hl);assert(p.sourceK==old.sourceK);assert(p.sourceOmega==old.sourceOmega);
 c.coherentDiscretization=true;assert(detail::wallComputationConfig(c).coherentDiscretization);detail::applySstGradients(m.input,y,1/y,2/y,0.,0.,0.,exact);detail::pointSst(m.input,c,y,l,r,hl+hr,coherent,hl);
 assert(std::abs(coherent.sourceK-exact.sourceK)<1e-12);assert(std::abs(coherent.sourceOmega-exact.sourceOmega)<1e-10);
}
'''
class SharedNumerics(unittest.TestCase):
 def test_coherent_seed_flag_is_local_and_default_legacy(self):
  self.check_cpp(CPP_FLAG)
 def check_cpp(self,source):
  with tempfile.TemporaryDirectory() as d:
   src=Path(d)/'probe.cpp';src.write_text(source);exe=Path(d)/'probe'
   b=subprocess.run(['g++','-std=c++17','-O2','-I',str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
   self.assertEqual(b.returncode,0,b.stderr)
   r=subprocess.run([str(exe)],capture_output=True,text=True)
   self.assertEqual(r.returncode,0,r.stdout+r.stderr);print(r.stdout,end='')
 def test_actual_shared_math(self):
  with tempfile.TemporaryDirectory() as d:
   src=Path(d)/'probe.cpp';src.write_text(CPP);exe=Path(d)/'probe'
   b=subprocess.run(['g++','-std=c++17','-O2','-I',str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
   self.assertEqual(b.returncode,0,b.stderr)
   r=subprocess.run([str(exe)],capture_output=True,text=True)
   self.assertEqual(r.returncode,0,r.stdout+r.stderr);print(r.stdout,end='')
if __name__=='__main__':unittest.main()
