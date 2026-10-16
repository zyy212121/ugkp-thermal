"""Node-team scheduling must preserve the original serial wall solve, including failure publication."""
from pathlib import Path
import shutil
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT
from validation.tests.gas_wall.test_wall_model import BASE

TEAM = r"""
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <iomanip>
struct Barrier {
 std::mutex mutex;std::condition_variable cv;int size,arrived=0,generation=0;
 explicit Barrier(int n):size(n){}
 void wait(){std::unique_lock<std::mutex> lock(mutex);const int g=generation;
  if(++arrived==size){arrived=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}
};
struct TeamStorage {Barrier barrier;bool flags[32]{},flag=false;double scalar=0;explicit TeamStorage(int n):barrier(n){}};
struct CpuTeam {
 TeamStorage& storage;int lane,lanes;
 int rank()const{return lane;}int size()const{return lanes;}bool leader()const{return lane==0;}
 void sync()const{storage.barrier.wait();}
 bool broadcast(bool v)const{if(leader())storage.flag=v;sync();v=storage.flag;sync();return v;}
 double broadcast(double v)const{if(leader())storage.scalar=v;sync();v=storage.scalar;sync();return v;}
 bool all(bool v)const{storage.flags[lane]=v;sync();if(leader()){storage.flag=true;for(int i=0;i<lanes;++i)storage.flag=storage.flag&&storage.flags[i];}sync();v=storage.flag;sync();return v;}
};
template<int Capacity> void run(int mode,int lanes){
 Model m;WallModelConfig<double> c;c.nodes=mode==4?40:24;c.enableSst=mode!=0;c.maxIterations=mode==2?1:60;
 m.in.model.mode=mode==3?GasMode::MixtureChemistry:GasMode::MixtureFrozen;
 m.in.matching.k=.5;m.in.matching.omega=200;m.in.integrateOwnerKSource=false;
 if(mode==5)m.in.matching.massFraction[1]=0,m.in.matching.massFraction[0]=1;
 if(mode==6)m.in.matching.temperature=std::numeric_limits<double>::quiet_NaN();
 if(mode>=7&&mode<=9){unsigned seed=4321;const int selected=mode==7?11:(mode==8?29:150);
  for(int index=0;index<=selected;++index){auto random=[&](){seed=1664525*seed+1013904223;return double(seed)/4294967296.;};
   const double L=std::pow(10.,-4+3*random()),Tw=250+800*random(),Te=300+1500*random(),u=std::pow(10.,-1+4*random());
   const double k=std::pow(10.,-10+12*random()),omega=std::pow(10.,-2+7*random()),mass=std::pow(10.,-8+8*random());
   const bool sst=index%2==0;double formation=0,rate=2;if(!sst){formation=1e5+3e5*random();rate=std::pow(10.,3*random());}
   if(index==selected){m.in.matchingDistance=L;m.in.ownerDistance=.4*L;m.in.temperature=Tw;m.in.matching.temperature=Te;m.in.matching.velocity[0]=u;
    m.in.matching.k=k;m.in.matching.omega=omega;m.in.massFlux[0]=.7*mass;m.in.massFlux[1]=.3*mass;
    c.nodes=16;c.maxIterations=80;c.enableSst=sst;if(!sst){m.in.model.mode=GasMode::MixtureChemistry;m.coef[2]=formation;m.rx[0].highRate.preExponential=rate;m.in.model.diffusivity[0]=1e-4;m.in.model.diffusivity[1]=3e-4;}}
  }
 }
 if(mode==10){c.nodes=65;c.enableSst=false;}
 if(mode==11)c.model=BoundaryLayerModel::ConstantTransport;
 if(mode==12)c.model=static_cast<BoundaryLayerModel>(99);
 WallWorkspace<double,2,Capacity> w;WallOutput<double,2> out;WallStatus status;bool success=false;double profile[Capacity*9]{};
#if WALL_TEAM
 TeamStorage storage(lanes);std::vector<std::thread> threads;
 for(int lane=0;lane<lanes;++lane)threads.emplace_back([&,lane]{bool ok=evaluateGasBoundaryLayerWorkspace<true>(m.in,c,w,out,status,profile,CpuTeam{storage,lane,lanes});if(lane==0)success=ok;});
 for(auto& thread:threads)thread.join();
#else
 (void)lanes;success=evaluateGasBoundaryLayerWorkspace<true>(m.in,c,w,out,status,profile);
#endif
 std::cout<<success<<' '<<int(status.code)<<' '<<status.node<<' '<<status.iteration<<' '<<status.residual<<' ';
 std::cout<<out.traceTemperature<<' '<<out.traceDensity<<' '<<out.conductiveHeatFlux<<' '<<out.integratedKSource<<' '<<out.ownerOmega<<' '<<out.wallKFlux<<' '<<out.iterations<<' '<<out.residual<<' ';
 for(int i=0;i<3;++i)std::cout<<out.traceVelocity[i]<<' '<<out.traction[i]<<' ';
 for(int i=0;i<2;++i)std::cout<<out.traceMassFraction[i]<<' '<<out.wallSpeciesFlux[i]<<' '<<out.matchingSpeciesFlux[i]<<' '<<out.reactionIntegral[i]<<' '<<out.speciesBalanceResidual[i]<<' ';
 for(int i=0;i<c.nodes;++i)for(int j=0;j<6;++j)std::cout<<w.state[i][j]<<' ';
 for(int i=0;i<c.nodes*9;++i)std::cout<<profile[i]<<' ';
 std::cout<<'\n';
}
int main(int argc,char**argv){std::cout<<std::hexfloat;int mode=std::atoi(argv[1]),lanes=std::atoi(argv[2]);if(mode==4)run<48>(mode,lanes);else if(mode==10)run<128>(mode,lanes);else run<24>(mode,lanes);}
"""

@pytest.fixture(scope='module')
def programs(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('wall-team')
    oracle=tmp/'oracle'
    shutil.copytree(SOURCE_ROOT/'common',oracle/'common')
    # Independent unchanged committed implementation is the numerical oracle.
    original=subprocess.run(['git','show','HEAD:common/gasWall/ReactingWallLayer.H'],cwd=SOURCE_ROOT,check=True,capture_output=True,text=True).stdout
    (oracle/'common/gasWall/ReactingWallLayer.H').write_text(original)
    original_model=subprocess.run(['git','show','HEAD:common/gasTransport/GasBoundaryLayerModelState.H'],cwd=SOURCE_ROOT,check=True,capture_output=True,text=True).stdout
    (oracle/'common/gasTransport/GasBoundaryLayerModelState.H').write_text(original_model)
    cpp=tmp/'team.cpp';cpp.write_text(BASE+TEAM)
    executables=[]
    for root,enabled in ((oracle,0),(SOURCE_ROOT,1)):
        exe=tmp/('wall-'+str(enabled))
        result=subprocess.run(['g++','-std=c++17','-O2','-pthread','-DWALL_TEAM='+str(enabled),'-I',str(root),'-I',str(root/'common'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        executables.append(exe)
    return executables

@pytest.mark.parametrize('mode',range(13))
@pytest.mark.parametrize('lanes',[1,4,7,32])
def test_node_team_matches_original_wall_state_and_publication(programs,mode,lanes):
    baseline,team=programs
    expected=subprocess.run([str(baseline),str(mode),'1'],check=True,capture_output=True,text=True,timeout=60).stdout
    actual=subprocess.run([str(team),str(mode),str(lanes)],check=True,capture_output=True,text=True,timeout=60).stdout
    assert actual==expected
