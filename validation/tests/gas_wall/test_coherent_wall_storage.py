"""Standalone ownership/sizing probes; no pytest dependency or CUDA claim."""
from pathlib import Path
import subprocess,tempfile,unittest,ast
ROOT=Path(__file__).resolve().parents[3]
CPP=r'''
#include <cassert>
#include <cstring>
#include <iostream>
#include "common/gasTransport/GasCoherentBoundaryLayer.H"
using namespace ugkwp;using namespace ugkwp::gaswall;
template<int M>void check(){
 static coherent::PersistentWallState<M> a,b;
 static WallWorkspace<double,2,M> worker;
 WallHistoryFrame<2> frame;frame.valid=true;frame.nodes=M;frame.distance=.001;frame.stretch=2;frame.temperature=300;frame.normal[1]=1;frame.tangent[0][0]=frame.tangent[1][2]=1;frame.independent[0]=0;frame.dependent=1;
 detail::SerialWallTeam team;
 assert(!loadCoherentWallState(a,frame,worker,nullptr,team));
 for(int i=0;i<M;++i)for(int j=0;j<6;++j)worker.state[i][j]=i*6+j+.5;
 a.cache.haveState=true;saveCoherentWallState(a,frame,worker,team);
 const double original=a.state[4][2];
 for(int i=0;i<M;++i)for(int j=0;j<6;++j)worker.state[i][j]=-1;
 assert(!loadCoherentWallState(b,frame,worker,nullptr,team));
 assert(loadCoherentWallState(a,frame,worker,nullptr,team));
 assert(worker.state[4][2]==original&&a.state[4][2]==original);
 auto changed=frame;changed.temperature=302;a.cache.factors.valid=true;
 assert(loadCoherentWallState(a,changed,worker,nullptr,team));
 assert(worker.state[4][2]==original-2&&a.state[4][2]==original&&!a.cache.factors.valid);
 changed.distance*=2;assert(!loadCoherentWallState(a,changed,worker,nullptr,team));
 assert(!a.cache.haveState);
 double accepted[M*6];for(int i=0;i<M*6;++i)accepted[i]=i+.25;
 assert(loadCoherentWallState(a,changed,worker,accepted,team));
 assert(a.cache.haveState&&worker.state[2][4]==accepted[16]);
 worker.state[2][4]+=1;assert(accepted[16]==16.25);
 assert(gasCoherentWallScratchBytes(M)==sizeof(coherent::Scratch<M>));
 assert(gasCoherentWallStateBytes(M)==sizeof(coherent::PersistentWallState<M>));
 std::cout<<M<<' '<<sizeof(coherent::Scratch<M>)<<' '<<sizeof(coherent::PersistentWallState<M>)<<'\n';
}
int main(){check<24>();check<48>();check<128>();}
'''
class Storage(unittest.TestCase):
 def test_worker_scratch_and_wall_history_are_distinct(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);src=d/'probe.cpp';src.write_text(CPP);exe=d/'probe'
   b=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True);self.assertEqual(b.returncode,0,b.stderr)
   r=subprocess.run([str(exe)],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);print(r.stdout,end='')
 def test_uniform_team_seed_load_and_save(self):
  module=ast.parse((ROOT/'validation/tests/gas_wall/test_wall_team_equivalence.py').read_text())
  team=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='TEAM' for x in n.targets)).split('template<int Capacity>')[0]
  body=r'''
#include "common/gasTransport/GasCoherentBoundaryLayer.H"
#include <cassert>
using namespace ugkwp;using namespace ugkwp::gaswall;
template<int M>void check(int lanes){
 auto wall=std::make_unique<coherent::PersistentWallState<M>>();auto work=std::make_unique<WallWorkspace<double,2,M>>();
 WallHistoryFrame<2> f;f.valid=true;f.nodes=M;f.distance=.1;f.stretch=2;f.temperature=300;f.normal[1]=1;f.tangent[0][0]=f.tangent[1][2]=1;f.independent[0]=0;f.dependent=1;wall->frame=f;
 double accepted[M*6];for(int i=0;i<M*6;++i)accepted[i]=i+.125;
 TeamStorage storage(lanes);std::vector<std::thread> threads;bool result[32]{};
 for(int lane=0;lane<lanes;++lane)threads.emplace_back([&,lane]{CpuTeam t{storage,lane,lanes};result[lane]=loadCoherentWallState(*wall,f,*work,accepted,t);saveCoherentWallState(*wall,f,*work,t);});
 for(auto&t:threads)t.join();for(int lane=0;lane<lanes;++lane)assert(result[lane]);
 assert(wall->cache.haveState);for(int i=0;i<M;++i)for(int j=0;j<6;++j)assert(wall->state[i][j]==accepted[i*6+j]);
}
int main(){for(int lanes:{1,4,7,32}){check<24>(lanes);check<48>(lanes);check<128>(lanes);}}
'''
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);src=d/'team.cpp';src.write_text('#include <memory>\n'+team+body);exe=d/'team'
   b=subprocess.run(['g++','-std=c++17','-O2','-pthread','-I'+str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True);self.assertEqual(b.returncode,0,b.stderr)
   r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=60);self.assertEqual(r.returncode,0,r.stderr)
if __name__=='__main__':unittest.main()
