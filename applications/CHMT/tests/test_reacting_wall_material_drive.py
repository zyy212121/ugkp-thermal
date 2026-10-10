from pathlib import Path
import subprocess
APP=Path(__file__).resolve().parents[1]
def test_wall_matching_changes_limit_material_lag(tmp_path):
    source=r'''
#include "materials/MaterialDrive.H"
#include <cassert>
int main(){using namespace chmt;for(int component=0;component<6;++component){
 IntervalHistory history;CouplingInterval interval;interval.identity.sequence=1;interval.end=1;std::string error;assert(history.begin(interval,error));
 GasIntervalRecord a;a.microSequence=1;a.end=.5;a.gasTrace.resize(1);a.gasTrace[0].pressure=100000;a.gasTrace[0].temperature=500;a.gasTrace[0].rho=1;a.gasTrace[0].Y[0]=1;
 a.gasWallMatching.resize(1);auto& m=a.gasWallMatching[0];m.pressure=150000;m.mechanicalPressure=100000;m.state.temperature=500;m.state.massFraction[0]=1;m.state.k=1;m.state.omega=10;
 assert(history.appendAccepted(a,error));auto b=a;b.microSequence=2;b.begin=.5;b.end=1;auto& n=b.gasWallMatching[0];
 if(component==0)n.pressure=200000;if(component==1)n.state.temperature=700;if(component==2){n.state.massFraction[0]=.7;n.state.massFraction[1]=.3;}
 if(component==3)n.state.velocity[1]=1000;if(component==4)n.state.k=2;if(component==5)n.state.omega=20;
 assert(history.appendAccepted(b,error));Real end=0;std::uint64_t samples=0;
 assert(materialDriveSlabEnd(history,0,1,.05,1,end,samples,error));assert(end==.5&&samples==1);
 }}
'''
    src=tmp_path/'drive.cpp';src.write_text(source);exe=tmp_path/'drive'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(APP),str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
