"""Execute the real layerRow and a frozen numerical/work-count reference."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import json
import re
import shutil
import subprocess
import pytest
from test_wall_model import BASE, ROOT


@pytest.mark.parametrize('standard', [14, 17])
def test_layer_row_skips_already_checked_unused_endpoints(tmp_path, standard):
    common = tmp_path / 'common'
    shutil.copytree(ROOT / 'common', common)
    path = common / 'gasWall/ReactingWallLayer.H'
    source = path.read_text()
    source = source.replace('namespace ugkwp { namespace gaswall { namespace detail {',
        'namespace testCounts { static unsigned long long pointState=0,tail=0; }\n'
        'namespace ugkwp { namespace gaswall { namespace detail {', 1)
    match = re.search(r'\bbool pointState\(', source)
    first = source.index('{', match.end())
    source = source[:first+1] + ' ++testCounts::pointState; ' + source[first+1:]
    path.write_text(source)
    reference = (VALIDATION_ROOT / 'tests/gas_wall/wall_local_efficiency_reference.hpp').read_text().replace(
        '    LayerPoint<R,N> a,b;', '    ++testCounts::tail; LayerPoint<R,N> a,b;')
    body = r'''
#include <cstring>
#include <cstdint>
int main(){
 using namespace ugkwp::gaswall::detail;
 unsigned long long cases=0,successes=0,referenceCalls=0,actualCalls=0,expectedSaved=0;
 for(int nodes:{4,24,48})for(bool sst:{false,true})for(int scenario=0;scenario<7;++scenario){
  Model m;WallModelConfig<double> c;c.nodes=nodes;c.enableSst=sst;
  m.in.matching.k=scenario==1?1.e-4:.5;m.in.matching.omega=scenario==1?1.:200.;
  m.in.matching.velocity[0]=30.;m.in.temperature=500.;m.in.matching.temperature=700.;
  if(scenario==2){m.in.model.mode=GasMode::MixtureChemistry;m.in.massFlux[0]=.0007;m.in.massFlux[1]=.0003;}
  LayerContext<double,2> ctx;makeContext(m.in,c,ctx);
  double x[48][6]{},y[48]{};
  for(int j=0;j<nodes;++j){const double f=double(j)/(nodes-1);
   y[j]=m.in.matchingDistance*f*f;x[j][0]=f*ctx.edgeU[0];
   x[j][2]=f*(m.in.matching.temperature-m.in.temperature);
   x[j][3]=m.in.matching.massFraction[ctx.independent[0]];
   x[j][4]=sst?std::sqrt(m.in.matching.k)*f:0.;
   x[j][5]=sst?f/std::sqrt(m.in.matching.omega):1.;
  }
  if(scenario==3)x[1][2]=-600.; // current or neighboring endpoint failure
  if(scenario==4)x[nodes-2][3]=-1.;
  if(scenario==5)m.in.pressure=-1.;
  if(scenario==6)x[nodes-2][5]=std::numeric_limits<double>::quiet_NaN();
  double original[48][6];std::memcpy(original,x,sizeof(x));
  for(int i=0;i<nodes;++i){
   double a[6]{},b[6]{};
   testCounts::pointState=testCounts::tail=0;const bool referenceOK=referenceLayerRow(m.in,c,ctx,x,y,i,a);
   const auto referenceCount=testCounts::pointState;
   testCounts::pointState=0;const bool actualOK=layerRow(m.in,c,ctx,x,y,i,b);
   const auto actualCount=testCounts::pointState;
   assert(referenceOK==actualOK);assert(std::memcmp(a,b,sizeof(a))==0);
   assert(std::memcmp(x,original,sizeof(x))==0);
   const auto saved=testCounts::tail?(i==1?1:2):0;
   expectedSaved+=saved;referenceCalls+=referenceCount;actualCalls+=actualCount;
   successes+=referenceOK;++cases;
  }
 }
 std::cout<<"{\"cases\":"<<cases<<",\"successes\":"<<successes
  <<",\"mismatches\":0,\"reference_pointState\":"<<referenceCalls
  <<",\"actual_pointState\":"<<actualCalls<<",\"expected_saved\":"<<expectedSaved<<"}\n";
}
'''
    path = tmp_path / 'layer_row.cpp'
    path.write_text(BASE + reference + body)
    exe = tmp_path / 'layer_row'
    build = subprocess.run(['g++', f'-std=c++{standard}', '-O2', '-Wall', '-Wextra',
        '-Werror', '-I' + str(tmp_path), '-I' + str(common),
        str(path), '-o', str(exe)], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    counts = json.loads(run.stdout)
    print(json.dumps(counts, sort_keys=True))
    assert counts['mismatches'] == 0 and counts['successes'] > 0
    assert counts['reference_pointState'] - counts['actual_pointState'] == counts['expected_saved'] > 0
