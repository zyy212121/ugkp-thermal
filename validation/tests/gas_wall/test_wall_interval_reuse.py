"""Count interval work and compare complete numerical output with uncached paths."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from validation.tools.paths import SOURCE_ROOT
from validation.tests.gas_wall.test_wall_model import BASE
from validation.tests.gas_wall.test_jacobian_point_cache import PROBE


def definition(text, marker):
    start=text.index(marker)
    end=text.index('{', start)+1
    depth=1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}')
        end+=1
    return text[start:end]


def build_probe(tmp_path, body, reference_jacobian=False):
    shutil.copytree(SOURCE_ROOT/'common',tmp_path/'common')
    header=tmp_path/'common/gasWall/ReactingWallLayer.H'
    text=header.read_text()
    if reference_jacobian:
        reference=definition(text,'template<class R,int N,int M,class TableReal> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool assembleJacobianNode(')
        reference=reference.replace('UGKWP_WALL_DEVICE_LEAF','inline')
        reference=reference.replace('assembleJacobianNode(', 'assembleJacobianNodeReference(',1)
        reference=reference.replace('cached(perturbed,j);', 'cached(perturbed,j);\n    UncachedIntervalRows<R,N,decltype(cached)> uncached{cached};')
        reference=reference.replace('ctx,cached,w.y', 'ctx,uncached,w.y')
        wrapper=r"""
namespace ugkwp { namespace gaswall { namespace detail {
template<class R,int N,class Rows> struct UncachedIntervalRows {
 const Rows& rows;
 UGKWP_GAS_HD CachedLayerRow<R,N> operator[](int i)const{return rows[i];}
};
"""
        (tmp_path/'jacobian_point_cache_reference.H').write_text(wrapper+reference+'\n}}}\n')
    for name,counter in [('pointState','jacobianPointStateCalls'),('intervalFlux','intervalCalls'),('layerRow','rowCalls')]:
        start=text.index(' '+name+'('); pos=text.index('{',start)+1
        text=text[:pos]+f'++::{counter};'+text[pos:]
    header.write_text(text)
    source=tmp_path/'probe.cpp'
    source.write_text('long long jacobianPointStateCalls=0,intervalCalls=0,rowCalls=0;\n'+BASE+body)
    binary=source.with_suffix('')
    command=['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I'+str(tmp_path),'-I'+str(tmp_path/'common'),str(source),'-o',str(binary)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return binary


@pytest.fixture(scope='module')
def interval_jacobian_probe(tmp_path_factory):
    root=tmp_path_factory.mktemp('jacobian-interval-cache')
    body=PROBE.replace('199-in.temperature','-1-in.temperature')
    body=body.replace('if(scenario==3)initial.state[bad][3]=-.1;','if(scenario==3)initial.state[bad][N+3]=-.1;')
    body=body.replace('if(scenario==5&&sst)require(partialFailures>0,"zero k did not exercise partial column publication");','')
    body=body.replace('long long referenceCalls=0,actualCalls=0;', 'long long referenceCalls=0,actualCalls=0,referenceIntervals=0,actualIntervals=0,referenceRows=0,actualRows=0;')
    body=body.replace('jacobianPointStateCalls=0;', 'jacobianPointStateCalls=intervalCalls=rowCalls=0;')
    body=body.replace('referenceCalls+=jacobianPointStateCalls;', 'referenceCalls+=jacobianPointStateCalls;referenceIntervals+=intervalCalls;referenceRows+=rowCalls;')
    body=body.replace('actualCalls+=jacobianPointStateCalls;', 'actualCalls+=jacobianPointStateCalls;actualIntervals+=intervalCalls;actualRows+=rowCalls;')
    body=body.replace('<<",\\"failures\\":"', '<<",\\"reference_intervals\\":"<<referenceIntervals<<",\\"actual_intervals\\":"<<actualIntervals<<",\\"reference_rows\\":"<<referenceRows<<",\\"actual_rows\\":"<<actualRows<<",\\"failures\\":"')
    return build_probe(root,body,reference_jacobian=True)


@pytest.mark.parametrize('nodes',[2,3,24,48])
@pytest.mark.parametrize('ns',[2,3])
@pytest.mark.parametrize('scenario',[0,1,2,3,4,5,6])
def test_interval_cache_keeps_jacobian_blocks_and_failure_writes(interval_jacobian_probe,nodes,ns,scenario):
    result=subprocess.run([str(interval_jacobian_probe),str(ns),str(nodes),'1','1',str(scenario)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    count=json.loads(result.stdout)
    assert count['actual_intervals']<=count['reference_intervals'],count
    assert count['actual_rows']==count['reference_rows'],count
    assert count['actual_calls']==count['reference_calls'],count
    if nodes==24 and ns==2 and scenario==0:
        print(count)
        assert count['reference_intervals']==804,count
        assert count['actual_intervals']<=320,count


PUBLICATION=r"""
#include <cstring>
#include "publication_interval_reference.H"
template<class T>void same(const T&a,const T&b){assert(std::memcmp(&a,&b,sizeof(T))==0);}
template<class T,int N>void sameOutput(const WallOutput<T,N>&a,const WallOutput<T,N>&b){
 same(a.traceTemperature,b.traceTemperature);same(a.traceDensity,b.traceDensity);
 same(a.traceVelocity,b.traceVelocity);same(a.traceMassFraction,b.traceMassFraction);
 same(a.traction,b.traction);same(a.conductiveHeatFlux,b.conductiveHeatFlux);
 same(a.heatConduction,b.heatConduction);same(a.heatViscous,b.heatViscous);
 same(a.heatTransport,b.heatTransport);same(a.heatBalanceResidual,b.heatBalanceResidual);
 same(a.wallSpeciesFlux,b.wallSpeciesFlux);same(a.matchingSpeciesFlux,b.matchingSpeciesFlux);
 same(a.reactionIntegral,b.reactionIntegral);same(a.chemistryMismatch,b.chemistryMismatch);
 same(a.speciesBalanceResidual,b.speciesBalanceResidual);same(a.integratedKSource,b.integratedKSource);
 same(a.ownerKDestruction,b.ownerKDestruction);same(a.ownerKContent,b.ownerKContent);
 same(a.ownerKBalanceResidual,b.ownerKBalanceResidual);same(a.ownerOmega,b.ownerOmega);
 same(a.wallKFlux,b.wallKFlux);same(a.kFluxCount,b.kFluxCount);same(a.kFluxHeight,b.kFluxHeight);
 same(a.kTransportFlux,b.kTransportFlux);same(a.omegaDiffusiveFlux,b.omegaDiffusiveFlux);
 same(a.residual,b.residual);same(a.chemistryMismatchAvailable,b.chemistryMismatchAvailable);
 same(a.iterations,b.iterations);
}
int main(){
 using namespace ugkwp::gaswall::detail;
 for(int nodes:{2,3,24,48})for(bool sst:{false,true})for(bool chemistry:{false,true})for(int blowing:{-1,0,1})for(bool volume:{false,true}){
  Model m;m.in.temperature=500;m.in.matching.temperature=700;m.in.matching.velocity[0]=30;
  m.in.matching.k=.5;m.in.matching.omega=200;
  m.in.model.mode=chemistry?GasMode::MixtureChemistry:GasMode::MixtureFrozen;
  m.in.massFlux[0]=blowing*7e-4;m.in.massFlux[1]=blowing*3e-4;
  double y[2]={.002,.006},weight[2]={2e-7,2e-7};m.in.quadrature={y,weight,2,4e-7,1.6e-9};
  m.in.volumeSstClosure=volume;m.in.integrateOwnerKSource=true;
  WallModelConfig<double> c;c.nodes=nodes;c.enableSst=sst;c.maxIterations=0;
  WallWorkspace<double,2,48>w;WallOutput<double,2> initial;WallStatus status;
  assert(evaluateWallModel(m.in,c,w,initial,status));LayerContext<double,2>ctx;assert(makeContext(m.in,c,ctx));
  auto actualWorkspace=w,referenceWorkspace=w;WallOutput<double,2>actual,expected;
  intervalCalls=0;assert(publishLayer(m.in,c,ctx,actualWorkspace,0,initial.residual,actual));const auto current=intervalCalls;
  intervalCalls=0;assert(referencePublishLayer(m.in,c,ctx,referenceWorkspace,0,initial.residual,expected));const auto reference=intervalCalls;
  sameOutput(actual,expected);same(actualWorkspace.seed,referenceWorkspace.seed);
  if(nodes==24&&sst&&volume)std::cout<<"publication intervals="<<reference<<"->"<<current<<" all_outputs=bitwise_equal\n";
  assert(current==nodes-1);assert(reference==nodes+1+(sst&&volume?nodes-1:0));
  actualWorkspace.state[nodes/2][2]=referenceWorkspace.state[nodes/2][2]=std::numeric_limits<double>::quiet_NaN();
  const auto keep=actual;assert(!publishLayer(m.in,c,ctx,actualWorkspace,0,0.,actual));sameOutput(actual,keep);
  assert(!referencePublishLayer(m.in,c,ctx,referenceWorkspace,0,0.,expected));sameOutput(expected,keep);
 }
}
"""


def test_one_publication_interval_sweep_preserves_every_output(tmp_path):
    shutil.copyfile(Path(__file__).with_name('publication_interval_reference.H'),tmp_path/'publication_interval_reference.H')
    binary=build_probe(tmp_path,PUBLICATION)
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    print(result.stdout,end='')
