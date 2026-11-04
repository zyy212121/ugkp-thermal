"""Count real publication leaves and compare every output with the old path.

Instrumentation lives only in a temporary copy of the production header. The
reference differs solely by unconditionally evaluating the old nodal SST source.
"""
from pathlib import Path
import subprocess
import shutil

import pytest
from test_wall_model import BASE, ROOT


def definition(text, marker):
    start = text.index(marker)
    end = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def instrumented_header():
    source = (ROOT/'common/gasWall/ReactingWallLayer.H').read_text()
    for name, counter in [('pointState', 'stateCalls'),
                          ('pointReaction', 'reactionCalls'), ('pointSst', 'sstCalls')]:
        start = source.index(' '+name+'(')
        body = source.index('{', start) + 1
        source = source[:body] + f'++::{counter};' + source[body:]
    marker = 'template<class R,int N,int M,class TableReal> UGKWP_GAS_HD UGKWP_WALL_DEVICE_LEAF bool publishLayer('
    reference = definition(source, marker).replace('bool publishLayer(', 'bool legacyPublishLayer(', 1)
    start = reference.index('for(int i=0;i<c.nodes;++i){LayerPoint<R,N> p;')
    end = reference.index('if(in.integrateOwnerKSource){w.seed', start)
    reference = reference[:start] + (
        'for(int i=0;i<c.nodes;++i){LayerPoint<R,N> p;'
        'if(!evaluatePoint(in,c,ctx,w.state,w.y,i,p))return false;\n        '
    ) + reference[end:]
    source = source.replace('} // detail', reference+'\n} // detail', 1)
    return 'inline int stateCalls=0,reactionCalls=0,sstCalls=0;\n'+source


BODY = r'''
#include <cstring>
template<class T> void same(T a,T b){assert(std::memcmp(&a,&b,sizeof(T))==0);}
template<class T,int N> void sameOutput(const WallOutput<T,N>& a,const WallOutput<T,N>& b){
 same(a.traceTemperature,b.traceTemperature);same(a.traceDensity,b.traceDensity);
 same(a.conductiveHeatFlux,b.conductiveHeatFlux);same(a.integratedKSource,b.integratedKSource);
 same(a.ownerOmega,b.ownerOmega);same(a.wallKFlux,b.wallKFlux);same(a.residual,b.residual);
 same(a.chemistryMismatchAvailable,b.chemistryMismatchAvailable);same(a.iterations,b.iterations);
 for(int d=0;d<3;++d){same(a.traceVelocity[d],b.traceVelocity[d]);same(a.traction[d],b.traction[d]);}
 for(int j=0;j<N;++j){same(a.traceMassFraction[j],b.traceMassFraction[j]);
  same(a.wallSpeciesFlux[j],b.wallSpeciesFlux[j]);same(a.matchingSpeciesFlux[j],b.matchingSpeciesFlux[j]);
  same(a.reactionIntegral[j],b.reactionIntegral[j]);same(a.chemistryMismatch[j],b.chemistryMismatch[j]);
  same(a.speciesBalanceResidual[j],b.speciesBalanceResidual[j]);}
}
int main(){
 for(bool sst:{false,true})for(bool chemistry:{false,true})for(bool blowing:{false,true}){
  Model m;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=30;
  m.in.matching.k=.5;m.in.matching.omega=200;
  m.in.model.mode=chemistry?GasMode::MixtureChemistry:GasMode::MixtureFrozen;
  if(blowing){m.in.massFlux[0]=GpuReal(7e-4);m.in.massFlux[1]=GpuReal(3e-4);}
  GpuReal y[2]={GpuReal(.002),GpuReal(.006)},weight[2]={GpuReal(2e-7),GpuReal(2e-7)};
  m.in.quadrature={y,weight,2,GpuReal(4e-7),GpuReal(1.6e-9)};
  WallModelConfig<GpuReal> c;c.enableSst=sst;c.nodes=TEST_NODES;c.maxIterations=100;
  WallWorkspace<GpuReal,2,48> w;WallOutput<GpuReal,2> solved;WallStatus status;
  if(!evaluateWallModel(m.in,c,w,solved,status)){
   std::cerr<<"solve failed "<<sst<<chemistry<<blowing<<" code "<<int(status.code)<<'\n';return 2;}
  auto in=detail::wallComputationInput(m.in);const auto controls=detail::wallComputationConfig(c);
  detail::LayerContext<WallReal,2> ctx;detail::makeContext(in,controls,ctx);
  for(bool diagnostic:{false,true}){
   in.integrateOwnerKSource=diagnostic;
   auto current=w,reference=w;WallOutput<WallReal,2> actual,expected;
   stateCalls=reactionCalls=sstCalls=0;
   assert(detail::publishLayer(in,controls,ctx,current,solved.iterations,WallReal(solved.residual),actual));
   const int states=stateCalls,reactions=reactionCalls,sources=sstCalls;
   stateCalls=reactionCalls=sstCalls=0;
   assert(detail::legacyPublishLayer(in,controls,ctx,reference,solved.iterations,WallReal(solved.residual),expected));
   sameOutput(actual,expected);
   assert(std::memcmp(current.seed,reference.seed,sizeof(current.seed))==0);
   assert(reactions==c.nodes&&reactionCalls==c.nodes);
   const int saved=sst&&!diagnostic?c.nodes-1:0;
   std::cout<<"nodes="<<c.nodes<<" sst="<<sst<<" chemistry="<<chemistry<<" blowing="<<blowing
    <<" diagnostic="<<diagnostic<<" pointState="<<stateCalls<<"->"<<states
    <<" pointSst="<<sstCalls<<"->"<<sources<<" all_outputs=bitwise_equal"<<std::endl;
   assert(states==stateCalls-2*saved);
   assert(sources==sstCalls-saved);
   if(sst&&!diagnostic){assert(states<stateCalls);assert(sources<sstCalls);}
   // Invalid profiles still reject without publishing any output, even when
   // the unused neighboring-state checks are skipped in the optimized path.
   current=w;reference=w;current.state[c.nodes/2][2]=reference.state[c.nodes/2][2]=
    std::numeric_limits<WallReal>::quiet_NaN();
   const auto keep=actual;const auto keepReference=expected;
   assert(!detail::publishLayer(in,controls,ctx,current,0,WallReal(0),actual));
   assert(!detail::legacyPublishLayer(in,controls,ctx,reference,0,WallReal(0),expected));
   sameOutput(actual,keep);sameOutput(expected,keepReference);
  }
 }
}
'''


@pytest.mark.parametrize('nodes', [24, 32])
def test_final_publication_skips_only_unused_sst_work(tmp_path, nodes):
    local = tmp_path/'instrumented'
    shutil.copytree(ROOT/'common/gasWall',local)
    (local/'WallModel.H').write_text((ROOT/'common/gasWall/WallModel.H').read_text())
    (local/'ReactingWallLayer.H').write_text(instrumented_header())
    source = tmp_path/'publication.cpp'
    source.write_text(BASE.replace('"common/gasWall/WallModel.H"', '"instrumented/WallModel.H"')
                      .replace('double', 'GpuReal') + BODY)
    exe = tmp_path/'publication'
    build = subprocess.run(['g++', '-std=c++17', '-O2', f'-DTEST_NODES={nodes}', '-I'+str(ROOT), '-I'+str(ROOT/'common'),
        '-I'+str(ROOT/'common/gasWall'), str(source), '-o', str(exe)], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout+build.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    print(result.stdout, end='')


def test_terminal_publication_retains_bounded_work_and_all_tails(tmp_path):
    local=tmp_path/'instrumented';shutil.copytree(ROOT/'common/gasWall',local)
    (local/'WallModel.H').write_text((ROOT/'common/gasWall/WallModel.H').read_text())
    (local/'ReactingWallLayer.H').write_text(instrumented_header())
    source=tmp_path/'terminal-publication.cpp'
    source.write_text(BASE.replace('"common/gasWall/WallModel.H"','"instrumented/WallModel.H"')+r'''
int main(){Model m;m.in.integrateOwnerKSource=false;WallModelConfig<double> c;c.nodes=24;c.maxIterations=100;c.enableSst=true;
m.in.matching.k=.5;m.in.matching.omega=200;WallWorkspace<double,2,24>w;WallOutput<double,2> initial,out;WallStatus status;
assert(evaluateWallModel(m.in,c,w,initial,status));detail::LayerContext<double,2> ctx;assert(detail::makeContext(m.in,c,ctx));
w.state[12][2]=std::numeric_limits<double>::infinity();out.ownerOmega=917;out.matchingSpeciesFlux[1]=918;
stateCalls=reactionCalls=sstCalls=0;
assert(!detail::publishLayer(m.in,c,ctx,w,3,std::numeric_limits<double>::quiet_NaN(),out,true));
assert(stateCalls>0&&stateCalls<20*c.nodes);
assert(out.ownerOmega==917&&out.matchingSpeciesFlux[1]==918);
std::cout<<"terminal states="<<stateCalls<<" reactions="<<reactionCalls<<" sst="<<sstCalls<<'\n';
}
''')
    binary=source.with_suffix('');cmd=['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasWall'),str(source),'-o',str(binary)]
    (tmp_path/'command.txt').write_text(' '.join(cmd));built=subprocess.run(cmd,capture_output=True,text=True)
    (tmp_path/'build.log').write_text(built.stdout+built.stderr);assert built.returncode==0,built.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True);(tmp_path/'run.log').write_text(run.stdout+run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
