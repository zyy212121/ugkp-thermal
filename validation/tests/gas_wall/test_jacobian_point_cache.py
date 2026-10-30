"""Jacobian point reuse must preserve B's finite differences and partial writes.

These executable tests catch cached reaction/SST state, stale perturbed points,
changed donor/attempt ordering, and eager reads that alter failure publication.
The work bound counts real pointState calls in a temporary header copy only.
"""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from validation.tools.paths import SOURCE_ROOT
from validation.tests.gas_wall.test_wall_model import BASE


REFERENCE = Path(__file__).with_name('jacobian_point_cache_reference.H')
COUNTER = '#include <cstdint>\nlong long jacobianPointStateCalls=0;\n'
PROBE = r'''
#include <cstring>
#include <cstdlib>
#include "jacobian_point_cache_reference.H"

struct Model3 {
 SpeciesThermoData<double> sp[3];double coef[9]={1000,0,0,1100,0,20000,900,0,-5000};
 double elements[3]={1,1,1},basis[3]={-1,1,0};
 GasReactionData<double> rx[1];GasStoichTerm<double> re[1],pr[1];WallInput<double,3> in;
 Model3(){for(int s=0;s<3;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;
  sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;
  sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
  auto& t=in.model.thermo;t.species=sp;t.coefficients=coef;t.coefficientCount=9;
  t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
  rx[0].reactantCount=1;rx[0].productCount=1;rx[0].highRate.preExponential=2;
  re[0].species=0;re[0].coefficient=1;pr[0].species=1;pr[0].coefficient=1;
  auto& mech=in.model.mechanism;mech.reactions=rx;mech.reactionCount=1;mech.reactants=re;
  mech.reactantTermCount=1;mech.products=pr;mech.productTermCount=1;
  mech.referencePressure=101325;mech.speciesOrderHash=1;mech.stoichiometricBasis=basis;mech.independentRank=1;
  in.pressure=101325;in.temperature=300;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;
  in.matching.temperature=500;in.matching.velocity[0]=10;in.matching.massFraction[0]=.7;
  in.matching.massFraction[1]=.2;in.matching.massFraction[2]=.1;
  in.model.viscosity=2e-5;in.model.conductivity=.04;for(int s=0;s<3;++s)in.model.diffusivity[s]=1e-5;
 }
};

void require(bool ok,const char* message,int node=-1){
 if(!ok){std::cerr<<message<<" node="<<node<<'\n';std::exit(42);}
}
template<class T> bool sameBits(const T& a,const T& b){return std::memcmp(&a,&b,sizeof(T))==0;}

template<int N,class Fixture> int run(int nodes,int sst,int reacting,int scenario){
 using namespace ugkwp::gaswall::detail;
 constexpr int Capacity=65,B=N+4;Fixture model;auto& in=model.in;
 in.matching.k=scenario==5?0:.5;in.matching.omega=200;in.integrateOwnerKSource=false;
 in.model.mode=reacting?GasMode::MixtureChemistry:GasMode::MixtureFrozen;
 // Distinct Cp and formation enthalpies expose an accidentally accumulated base point.
 if(N==2){model.coef[3]=1100;model.coef[5]=20000;}
 WallModelConfig<double> config;config.nodes=nodes;config.enableSst=sst;
 LayerContext<double,N> context;require(makeContext(in,config,context),"context preparation failed");
 WallWorkspace<double,N,Capacity> initial;
 for(int i=0;i<Capacity;++i){
  const double eta=double(i)/double(nodes-1),f=std::pow(eta,config.stretch);
  initial.y[i]=in.matchingDistance*f;
  for(int k=0;k<B;++k)initial.state[i][k]=std::numeric_limits<double>::quiet_NaN();
  if(i<nodes){initial.state[i][0]=f*context.edgeU[0];initial.state[i][1]=f*context.edgeU[1];
   initial.state[i][2]=f*(in.matching.temperature-in.temperature);
   for(int k=0;k<N-1;++k)initial.state[i][3+k]=scenario==1?(k==N-2?1.:0.):in.matching.massFraction[context.independent[k]];
   initial.state[i][N+2]=sst?std::sqrt(in.matching.k)*f:0;
   initial.state[i][N+3]=sst?f/std::sqrt(in.matching.omega):1;}
  for(int r=0;r<B;++r)for(int col=0;col<B;++col){
   const double sentinel=1234.+double(i*B*B+r*B+col)/1024.;
   initial.lower[i][r][col]=sentinel;initial.diagonal[i][r][col]=-sentinel;
   initial.upper[i][r][col]=sentinel+.25;}
 }
 double norm=0;require(allResidual(in,config,context,initial.state,initial.y,initial.residual,norm),"valid initial residual failed");
 const int bad=nodes/2;
 if(scenario==2)initial.state[bad][2]=-1-in.temperature;
 if(scenario==3)initial.state[bad][3]=std::numeric_limits<double>::quiet_NaN();
 if(scenario==4)initial.state[bad][2]=std::numeric_limits<double>::quiet_NaN();
 if(scenario==6)initial.state[bad][2]=4000-in.temperature;
 auto expected=initial,actual=initial;
 long long referenceCalls=0,actualCalls=0;int failures=0,partialFailures=0;
 for(int j=0;j<nodes;++j){
  const auto before=expected;
  jacobianPointStateCalls=0;
  const bool reference=assembleJacobianNodeReference(in,config,context,expected,j);
  referenceCalls+=jacobianPointStateCalls;
  jacobianPointStateCalls=0;
  const bool result=assembleJacobianNode(in,config,context,actual,j);
  actualCalls+=jacobianPointStateCalls;
  require(reference==result,"Jacobian success/failure differs",j);
  require(sameBits(expected.lower,actual.lower),"lower blocks differ",j);
  require(sameBits(expected.diagonal,actual.diagonal),"diagonal blocks differ",j);
  require(sameBits(expected.upper,actual.upper),"upper blocks differ",j);
  require(sameBits(initial.state,actual.state)&&sameBits(initial.state,expected.state),"immutable iterate changed",j);
  require(sameBits(initial.y,actual.y)&&sameBits(initial.y,expected.y),"coordinates changed",j);
  require(sameBits(initial.residual,actual.residual)&&sameBits(initial.residual,expected.residual),"base residual changed",j);
  if(!result){++failures;if(!sameBits(before.lower,expected.lower)||!sameBits(before.diagonal,expected.diagonal)||!sameBits(before.upper,expected.upper))++partialFailures;}
 }
 if(scenario>=2&&scenario<=4)require(failures>0,"invalid point did not exercise failure");
 if(scenario==5&&sst)require(failures==0,"zero-k boundary probe must remain representable");
 if(scenario==0||scenario==1||scenario==6)require(failures==0,"valid Jacobian fixture failed");
 std::cout<<"{\"reference_calls\":"<<referenceCalls<<",\"actual_calls\":"<<actualCalls
          <<",\"failures\":"<<failures<<",\"partial_failures\":"<<partialFailures<<"}\n";
 return 0;
}
int main(int argc,char**argv){require(argc==6,"expected Ns nodes SST chemistry scenario");
 const int ns=std::atoi(argv[1]),nodes=std::atoi(argv[2]),sst=std::atoi(argv[3]);
 const int chemistry=std::atoi(argv[4]),scenario=std::atoi(argv[5]);
 return ns==2?run<2,Model>(nodes,sst,chemistry,scenario):run<3,Model3>(nodes,sst,chemistry,scenario);
}
'''


@pytest.fixture(scope='module')
def jacobian_probe(tmp_path_factory):
    root = tmp_path_factory.mktemp('jacobian-point-cache')
    shutil.copytree(SOURCE_ROOT / 'common', root / 'common')
    header = root / 'common/gasWall/ReactingWallLayer.H'
    text = header.read_text()
    signature = text.index(' pointState(')
    body = text.index('){', signature) + 2
    header.write_text(text[:body] + '\n++::jacobianPointStateCalls;\n' + text[body:])
    shutil.copyfile(REFERENCE, root / REFERENCE.name)
    source = root / 'jacobian.cpp'
    source.write_text(COUNTER + BASE + PROBE)
    binary = root / 'jacobian'
    command = ['g++', '-std=c++17', '-O2', '-ffp-contract=off', '-Wall', '-Wextra',
               '-I' + str(root), '-I' + str(root / 'common'), str(source), '-o', str(binary)]
    compiled = subprocess.run(command, capture_output=True, text=True)
    (root / 'build-command.json').write_text(json.dumps(command, indent=2) + '\n')
    (root / 'build.log').write_text(compiled.stdout + compiled.stderr)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return binary


def observe(binary, ns, nodes, sst, reacting, scenario):
    result = subprocess.run([str(binary), *map(str, (ns, nodes, sst, reacting, scenario))],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize('nodes', [2, 3, 4, 24, 32, 33, 48, 65])
@pytest.mark.parametrize('ns', [2, 3])
@pytest.mark.parametrize('sst', [0, 1])
@pytest.mark.parametrize('reacting', [0, 1])
@pytest.mark.parametrize('scenario', [0, 1, 2, 3, 4, 5, 6],
                         ids=['normal', 'pure-donor', 'negative-temperature', 'nan-species',
                              'nan-temperature', 'zero-k', 'thermo-range-endpoint'])
def test_cached_jacobian_matches_original_blocks_and_failure_writes(jacobian_probe, nodes, ns, sst, reacting, scenario):
    observe(jacobian_probe, ns, nodes, sst, reacting, scenario)


def test_point_state_work_is_reused_across_jacobian_columns(jacobian_probe):
    result = observe(jacobian_probe, 2, 24, 1, 0, 0)
    assert result['reference_calls'] > 10*result['actual_calls'], result
    assert result['actual_calls'] <= 258, result
    print(json.dumps(result, sort_keys=True))


def test_constant_cp_table_endpoint_does_not_add_artificial_retry(jacobian_probe):
    normal = observe(jacobian_probe, 2, 24, 1, 0, 0)
    retry = observe(jacobian_probe, 2, 24, 1, 0, 6)
    assert retry['reference_calls'] == normal['reference_calls'], (normal, retry)
    # Constant-Cp thermodynamics evaluates finite positive temperatures outside
    # tabulation bounds; the Newton probe is not a caloric inversion bracket.
