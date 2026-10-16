"""Exact seed state/failure oracle and immutable construction work bound."""
import json
from pathlib import Path
import shutil
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT
from validation.tests.gas_wall.test_wall_model import BASE
from validation.tests.gas_wall.test_jacobian_point_cache import PROBE as JP
MODEL3=JP[JP.index('struct Model3'):JP.index('void require')]
PROBE=r'''
#include <cstring>
#include <cstdlib>
#include "seed_point_reuse_reference.H"
void require(bool ok,const char* why){if(!ok){std::cerr<<why<<'\n';std::exit(42);}}
template<class T> bool same(const T& a,const T& b){return std::memcmp(&a,&b,sizeof(T))==0;}
template<int N,class Fixture> int run(int nodes,bool sst,bool reacting,int scenario){
 using namespace ugkwp::gaswall::detail;
 Fixture m;auto& in=m.in;in.matching.k=scenario==5?0:.5;in.matching.omega=200;
 in.model.mode=reacting?GasMode::MixtureChemistry:GasMode::MixtureFrozen;
 if(N==2){m.coef[3]=1100;m.coef[5]=20000;}
 WallModelConfig<double> c;c.nodes=nodes;c.enableSst=sst;
 LayerContext<double,N> ctx;require(makeContext(in,c,ctx),"context");
 WallWorkspace<double,N,65> initial;
 for(int i=0;i<65;++i){const double f=std::pow(double(i)/double(nodes-1),c.stretch);
  initial.y[i]=in.matchingDistance*f;
  for(int j=0;j<N+4;++j)initial.state[i][j]=1234.+i+j;
  if(i<nodes){initial.state[i][0]=f*ctx.edgeU[0];initial.state[i][1]=f*ctx.edgeU[1];
   initial.state[i][2]=f*(in.matching.temperature-in.temperature);
   for(int j=0;j<N-1;++j)initial.state[i][3+j]=scenario==1?(j==N-2?1.:0.):in.matching.massFraction[ctx.independent[j]];
   initial.state[i][N+2]=sst?std::sqrt(in.matching.k)*f:0;
   initial.state[i][N+3]=sst?f/std::sqrt(in.matching.omega):1;}}
 if(scenario==2)initial.state[nodes/2][2]=199-in.temperature;
 if(scenario==3)initial.state[nodes/2][3]=-.1;
 if(scenario==4)initial.state[nodes/2][2]=std::numeric_limits<double>::quiet_NaN();
 auto expected=initial,actual=initial;seedPointCalls=0;seedIterations=0;
 bool want=seedSstLayerReference(in,c,ctx,expected);
 auto refCalls=seedPointCalls,refIterations=seedIterations;seedPointCalls=0;seedIterations=0;
 bool got=seedSstLayer(in,c,ctx,actual);
 require(got==want,"return status changed");
 require(same(expected.state,actual.state),"state/partial state changed");
 require(same(expected.seed,actual.seed),"coefficient/partial coefficients changed");
 require(same(expected.delta,actual.delta),"linear solution changed");
 require(same(expected.y,actual.y),"geometry changed");
 require(seedIterations==refIterations,"iteration trajectory changed");
 if(sst&&scenario>=2&&scenario<=4)require(!got,"invalid node accepted");
 std::cout<<"{\"ok\":"<<got<<",\"calls\":"<<seedPointCalls<<",\"iterations\":"<<seedIterations<<",\"reference_calls\":"<<refCalls<<"}\n";return 0;
}
int main(int argc,char**argv){require(argc==6,"arguments");int ns=atoi(argv[1]),n=atoi(argv[2]),sst=atoi(argv[3]),reacting=atoi(argv[4]),scenario=atoi(argv[5]);
 return ns==2?run<2,Model>(n,sst,reacting,scenario):run<3,Model3>(n,sst,reacting,scenario);}
'''
@pytest.fixture(scope='module')
def seed_probe(tmp_path_factory):
    root=tmp_path_factory.mktemp('seed-reuse');shutil.copytree(SOURCE_ROOT/'common',root/'common')
    header=root/'common/gasWall/ReactingWallLayer.H';text=header.read_text()
    body=text.index('){',text.index(' pointState('))+2
    text=text[:body]+'\n++::seedPointCalls;\n'+text[body:]
    marker='for(int iteration=0;iteration<200;++iteration){'
    assert text.count(marker)==1
    header.write_text(text.replace(marker,marker+'++::seedIterations;'))
    ref=Path(__file__).with_name('seed_point_reuse_reference.H').read_text()
    (root/'seed_point_reuse_reference.H').write_text(ref.replace(marker,marker+'++::seedIterations;'))
    source=root/'probe.cpp';source.write_text('long long seedPointCalls=0,seedIterations=0;\n'+BASE+MODEL3+PROBE)
    binary=root/'probe';result=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(root),'-I'+str(root/'common'),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return binary

def observe(binary,ns,nodes,sst,reacting,scenario):
    r=subprocess.run([str(binary),*map(str,(ns,nodes,sst,reacting,scenario))],capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stdout+r.stderr
    return json.loads(r.stdout)

@pytest.mark.parametrize('nodes',[2,3,24,33,65])
@pytest.mark.parametrize('ns',[2,3])
@pytest.mark.parametrize('sst',[0,1])
@pytest.mark.parametrize('reacting',[0,1])
@pytest.mark.parametrize('scenario',range(6),ids=['normal','pure','low-T','negative-Y','nan-T','zero-k'])
def test_seed_reuse_preserves_state_and_failure_writes(seed_probe,ns,nodes,sst,reacting,scenario):
    observe(seed_probe,ns,nodes,sst,reacting,scenario)

@pytest.mark.parametrize('nodes',[3,24,33,65])
@pytest.mark.parametrize('ns',[2,3])
def test_each_immutable_seed_node_is_constructed_once_per_iteration(seed_probe,ns,nodes):
    r=observe(seed_probe,ns,nodes,1,0,0)
    assert r['ok']==1,r
    # One initial wall point; n coefficient points and two interval endpoints per iteration.
    assert r['calls']==1+(nodes+2)*r['iterations'],r
    assert r['calls']<r['reference_calls'],r
