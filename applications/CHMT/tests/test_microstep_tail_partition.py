"""Exercise the exact production first-step selection and actual time additions."""
from pathlib import Path
import re
import subprocess

APP=Path(__file__).resolve().parents[1]


def test_production_microsteps_keep_feasible_endpoint_tail(tmp_path):
    source=(APP/'gpu/Backend.cu').read_text()
    selection=re.search(r'b.microTime=b.trial.time;Real dt=(.*);\n    for\(int retry=',source).group(1)
    cpp=r'''
#include "gpu/GasWindowProgram.H"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace chmt;
Real choose(Real time,Real end,Real cap,Real maximum,Real minimum){
 struct B {HostState trial;ModelConfig model;WallProgram program;Real microTime;} b;
 b.trial.time=time;b.microTime=time;b.model.physics.maxDt=maximum;b.model.physics.minDt=minimum;
 b.program.interval.end=end;
 return SELECTION;
}
int main(){
 const Real minimum=1e-14;
 for(bool retry:{false,true})for(int j=0;j<1000;++j){
  Real begin=.0000137*j,time=begin,end=begin+.001,integrated=0;
  for(int i=0;i<2000 && time<end;++i){
   const Real remaining=end-time,cap=1e-6,dt=choose(time,end,cap,2e-6,minimum);
   assert(dt>=minimum && dt<=cap && dt<=2e-6 && dt<=remaining && time+dt>time);
   // A retry can only shrink the accepted step. Shrinking must not leave a
   // smaller endpoint tail. Retry once early to exercise another partition.
   const Real accepted=(retry&&i==3?dt*.5:dt);assert(accepted>=minimum);
   integrated+=accepted;time+=accepted;
  }
  assert(time==end);assert(std::abs(integrated-(end-begin))<1e-15);
 }
 // A feasible split never exceeds the caller's stable cap or the max step.
 assert(choose(0,2.1,1.5,1.25,1)==1.05);
 // Less than two minimum steps cannot be split under a cap below the endpoint.
 assert(choose(0,1.5,1.25,2,1)<1);
 assert(choose(0,.5,2,2,1)<1);
 // An exactly feasible last minimum step and a complete single step remain.
 assert(choose(0,2,1,1,1)==1);assert(choose(0,1.5,2,2,1)==1.5);
 std::cout<<"2000 endpoint partitions and min/cap/retry contracts passed\n";
}
'''.replace('SELECTION',selection)
    src=tmp_path/'tail.cpp';src.write_text(cpp);binary=tmp_path/'tail'
    build=subprocess.run(['g++','-std=c++14','-O2','-I',str(APP),'-I',str(APP.parents[1]/'common'),str(src),'-o',str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
