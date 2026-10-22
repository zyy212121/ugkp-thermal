"""The production adapter retains the pre-postfit steady workspace route."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class SteadyProductionRoute(unittest.TestCase):
    def test_adapter_does_not_allocate_alternate_routes(self):
        adapter = (ROOT / 'applications/gasUGKP/private_backend/BoundaryLayerAdapter.cuh').read_text()
        for allocation in ('configureConservativeWallStorage(', 'model.referenceScratch=',
                           'model.coherentEnabled=true', 'gasBoundaryLayerModel.acceptedHistory,'):
            self.assertNotIn(allocation, adapter)

    def test_default_workspace_preserves_steady_nodes_and_publication(self):
        code = r'''
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include <cassert>
#include <cstring>
using namespace ugkwp;
using namespace ugkwp::gaswall;
void same(double a,double b){assert(std::memcmp(&a,&b,sizeof(double))==0);}
int main(){
 GasBoundaryLayerModelState<double,2> model;
 assert(!model.conservative.enabled && !model.coherentEnabled);
 assert(!model.referenceScratch && !model.acceptedHistory && !model.coherentScratch);
 CapturedSstWall f; f.config.nodes=24;
 WallWorkspace<double,2,24> a{},b{};WallOutput<double,2> oa{},ob{};WallStatus sa{},sb{};
 assert(evaluateWallModel(f.input,f.config,a,oa,sa));
 double profile[24*9]{};
 assert(evaluateGasBoundaryLayerWorkspace<true>(f.input,f.config,b,ob,sb,profile));
 for(int i=0;i<24;++i){same(a.y[i],b.y[i]);for(int j=0;j<6;++j)same(a.state[i][j],b.state[i][j]);
  WallProfileNode<double,2> p;assert(reactingWallProfileNode(f.input,f.config,a,i,p));
  same(profile[i*9],p.y);for(int j=0;j<3;++j)same(profile[i*9+1+j],p.velocity[j]);
  same(profile[i*9+4],p.temperature);same(profile[i*9+5],p.k);same(profile[i*9+6],p.omega);
  for(int j=0;j<2;++j)same(profile[i*9+7+j],p.massFraction[j]);}
 same(oa.ownerKContent,ob.ownerKContent);same(oa.ownerKDestruction,ob.ownerKDestruction);
 same(oa.integratedKSource,ob.integratedKSource);same(oa.ownerOmega,ob.ownerOmega);
 same(oa.conductiveHeatFlux,ob.conductiveHeatFlux);same(oa.wallKFlux,ob.wallKFlux);
 for(int j=0;j<3;++j)same(oa.traction[j],ob.traction[j]);
 for(int j=0;j<2;++j){same(oa.wallSpeciesFlux[j],ob.wallSpeciesFlux[j]);same(oa.matchingSpeciesFlux[j],ob.matchingSpeciesFlux[j]);}
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'route.cpp'
            binary = Path(directory) / 'route'
            source.write_text(code)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(ROOT),
                            str(source), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
