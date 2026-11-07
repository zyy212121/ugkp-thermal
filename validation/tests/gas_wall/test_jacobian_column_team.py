"""Shared column teams preserve scalar Newton updates and simplex donor coupling."""
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT
from validation.tests.gas_wall.test_wall_model import BASE
from validation.tests.gas_wall.test_wall_team_equivalence import TEAM
from validation.tests.gas_wall.test_jacobian_point_cache import PROBE

CPU_TEAM = TEAM[:TEAM.index("template<int Capacity>")]
MODEL3 = PROBE[PROBE.index("struct Model3"):PROBE.index("void require")]
CHECK = r"""
#include <cstring>
#include <memory>
template<int N,class Fixture> int check(int nodes,int sst,int scenario,int lanes){
 using namespace ugkwp::gaswall::detail;
 constexpr int Capacity=65,B=N+4;
 Fixture model;auto& in=model.in;in.model.mode=GasMode::MixtureFrozen;
 in.matching.k=.5;in.matching.omega=200;in.integrateOwnerKSource=false;
 WallModelConfig<double> config;config.nodes=nodes;config.enableSst=sst;
 if(!sst)config.model=BoundaryLayerModel::EquilibriumOde;
 LayerContext<double,N> ctx;if(!makeContext(in,config,ctx))return 2;
 auto initial=std::make_unique<WallWorkspace<double,N,Capacity>>();
 for(int i=0;i<nodes;++i){const double f=std::pow(double(i)/(nodes-1),config.stretch);
  initial->y[i]=in.matchingDistance*f;
  for(int d=0;d<2;++d)initial->state[i][d]=f*ctx.edgeU[d];
  initial->state[i][2]=f*(in.matching.temperature-in.temperature);
  for(int j=0;j<N-1;++j)initial->state[i][3+j]=scenario==1?(j==N-2?1.:0.):in.matching.massFraction[ctx.independent[j]];
  initial->state[i][N+2]=sst?std::sqrt(in.matching.k)*f:0;
  initial->state[i][N+3]=sst?f/std::sqrt(in.matching.omega):1;
 }
 double norm=0;if(!allResidual(in,config,ctx,initial->state,initial->y,initial->residual,norm))return 3;
 if(scenario==2)initial->state[nodes/2][2]=-1-in.temperature;
 auto expected=std::make_unique<WallWorkspace<double,N,Capacity>>(*initial);
 auto actual=std::make_unique<WallWorkspace<double,N,Capacity>>(*initial);
 const bool reference=newtonStep(in,config,ctx,*expected);
 TeamStorage storage(lanes);std::vector<std::thread> threads;bool accepted=false;
 for(int lane=0;lane<lanes;++lane)threads.emplace_back([&,lane]{
  const bool ok=newtonStep(in,config,ctx,*actual,CpuTeam{storage,lane,lanes});if(!lane)accepted=ok;
 });
 for(auto& thread:threads)thread.join();
 if(reference!=accepted)return 4;
 if(std::memcmp(initial->state,actual->state,sizeof(initial->state)))return 5;
 if(std::memcmp(initial->residual,actual->residual,sizeof(initial->residual)))return 6;
 if(reference){
  if(std::memcmp(expected->delta,actual->delta,sizeof(actual->delta)))return 7;
  if(std::memcmp(expected->lower,actual->lower,sizeof(actual->lower)))return 8;
  if(std::memcmp(expected->diagonal,actual->diagonal,sizeof(actual->diagonal)))return 9;
  if(std::memcmp(expected->upper,actual->upper,sizeof(actual->upper)))return 10;
 }
 if(scenario==2&&reference)return 11;
 std::cout<<"PASS Ns="<<N<<" nodes="<<nodes<<" sst="<<sst<<" scenario="<<scenario<<" lanes="<<lanes<<" accepted="<<accepted<<'\n';
 return 0;
}
int main(int argc,char**argv){if(argc!=6)return 1;int ns=std::atoi(argv[1]),n=std::atoi(argv[2]),sst=std::atoi(argv[3]),scenario=std::atoi(argv[4]),lanes=std::atoi(argv[5]);
 return ns==2?check<2,Model>(n,sst,scenario,lanes):check<3,Model3>(n,sst,scenario,lanes);}
"""

@pytest.fixture(scope="module")
def column_team_probe(tmp_path_factory):
    root = tmp_path_factory.mktemp("jacobian-column-team")
    source = root / "probe.cpp"
    source.write_text(BASE + CPU_TEAM + MODEL3 + CHECK)
    binary = root / "probe"
    result = subprocess.run(["g++", "-std=c++17", "-O2", "-ffp-contract=off", "-pthread",
                             "-I", str(SOURCE_ROOT), "-I", str(SOURCE_ROOT / "common"),
                             str(source), "-o", str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return binary

@pytest.mark.parametrize("ns", [2, 3])
@pytest.mark.parametrize("nodes", [3, 8, 65])
@pytest.mark.parametrize("sst", [0, 1])
@pytest.mark.parametrize("scenario", [0, 1, 2], ids=["mixture", "local-donor", "invalid-temperature"])
@pytest.mark.parametrize("lanes", [4, 7, 32])
def test_column_team_preserves_scalar_newton(column_team_probe, ns, nodes, sst, scenario, lanes):
    result = subprocess.run([str(column_team_probe), *map(str, (ns, nodes, sst, scenario, lanes))],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr + f" exit={result.returncode}"
