"""Matching-only common T/Y sampling; runnable with unittest or pytest."""
from pathlib import Path
import subprocess
import ast
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


class ConservativeMatchingPhysicalTrace(unittest.TestCase):
    def compile_run(self, body, production=False):
        prefix = ''
        if production:
            source = (HERE/'gas_state_probe.cpp').read_text()
            prefix = source[:source.index('void hashBytes')]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)/'matching.cpp'; exe=src.with_suffix('')
            src.write_text(prefix+body)
            result=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra',
                '-I'+str(ROOT),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),
                '-I'+str(HERE),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(exe)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_hd_sampler_changes_only_temperature_and_each_species(self):
        self.compile_run(r'''
#include "common/gasTransport/GasConservativeMatching.H"
#include <cassert>
#include <cmath>
#include <initializer_list>
struct State {
 int nCells=2;int faceOwner[1]={0},faceNeighbour[1]={1};
 double faceWeight[1]={.31},rho[2]={1.3,2.1};
 struct{double rho[6]={.78,.42,.39,1.05,.13,.63};}gasSpecies;
};
int main(){State s;ugkwp::gaswall::WallMatchingState<double,3> m;
 m.temperature=999;m.velocity[0]=7;m.velocity[1]=-2;m.velocity[2]=3;m.k=4;m.omega=5;
 for(double weight:{-.2,0.,.31,.5,1.,1.2}){
  s.faceWeight[0]=weight;const double w=weight<0?0:weight>1?1:weight;
  ugkwp::sampleConservativeWallThermocomposition(s,0,307.25,m);
  assert(m.temperature==307.25);
  for(int j=0;j<3;++j)assert(m.massFraction[j]==w*s.gasSpecies.rho[2*j]/s.rho[0]+(1-w)*s.gasSpecies.rho[2*j+1]/s.rho[1]);
  assert(m.velocity[0]==7&&m.velocity[1]==-2&&m.velocity[2]==3&&m.k==4&&m.omega==5);
 }
 // Every species comes from its own density; no last-species closure is added.
 s.faceWeight[0]=.31;s.gasSpecies.rho[5]=std::nextafter(s.gasSpecies.rho[5],1.);
 ugkwp::sampleConservativeWallThermocomposition(s,0,307.25,m);
 assert(m.massFraction[2]==.31*s.gasSpecies.rho[4]/s.rho[0]+(1-.31)*s.gasSpecies.rho[5]/s.rho[1]);
 // Sampling is not a new physical refusal or repair operation.
 ugkwp::sampleConservativeWallThermocomposition(s,0,-4.,m);assert(m.temperature==-4.);
}
''')

    def test_actual_bulk_temperature_and_common_sampling(self):
        self.compile_run(r'''
#include "gasTransport/GasConservativeMatching.H"
#include <cassert>
#include <initializer_list>
struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2> gasSpecies;};
int main(){State s;s.nCells=2;s.nFaces=s.nInternalFaces=1;s.Rgas=287;s.rhoMin=1e-12;s.TgasMin=200;
 Real rho[2]={1,1},u[2]={7,13},v[2]={-2,1},z[2]={0,0},p[2]={101325,102000},T[2]={100,300},weight[1]={.31};
 Real species[4]={.8,.84,.2,.16};int owners[1]={0},neighbours[1]={1},status[2]={0,0};
 s.rho=rho;s.Ux=u;s.Uy=v;s.Uz=z;s.p=p;s.Tgas=T;s.faceWeight=weight;s.faceOwner=owners;s.faceNeighbour=neighbours;
 s.gasSpecies.rho=species;s.gasSpecies.cellStatus=status;s.gasSpecies.faceStatus=status;
 ugkwp::gaswall::WallMatchingState<Real,2> m;m.velocity[0]=7;m.velocity[1]=-2;m.velocity[2]=3;m.k=4;m.omega=5;
 for(int legacy=0;legacy<2;++legacy){s.gasSpecies.mode=legacy?ugkwp::GasMode::SingleLegacy:ugkwp::GasMode::MixtureFrozen;
  for(Real w:{Real(-.2),Real(0),Real(.31),Real(.5),Real(1),Real(1.2)}){
   weight[0]=w;const Real t=clampRange(w,Real(0),Real(1));
   for(int c=0;c<2;++c){auto f=riemannFacePrimitiveForGradient(s,c,0);const Real wc=c==0?t:1-t;
    const Real left=c==0?200:300,right=c==0?300:200;
    assert(f.T==wc*left+(1-wc)*right);
    assert(f.T==ugkwp::physicalFaceTemperature(left,right,wc));
    ugkwp::sampleConservativeWallThermocomposition(s,0,f.T,m);
    assert(m.temperature==f.T);
    for(int j=0;j<2;++j)assert(m.massFraction[j]==t*species[2*j]/rho[0]+(1-t)*species[2*j+1]/rho[1]);
    assert(m.velocity[0]==7&&m.velocity[1]==-2&&m.velocity[2]==3&&m.k==4&&m.omega==5);
   }
  }
 }
 // The disclosed .8/.84 static interface gains the common physical trace;
 // this is not a BVP or full nullspace accuracy assertion.
 weight[0]=.5;T[0]=T[1]=300;s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;
 auto f=riemannFacePrimitiveForGradient(s,0,0);ugkwp::sampleConservativeWallThermocomposition(s,0,f.T,m);
 assert(m.temperature==300&&m.massFraction[0]==Real(.5)*Real(.8)+Real(.5)*Real(.84));
 assert(m.massFraction[0]>species[0]&&m.massFraction[0]<species[1]);
}
''', production=True)


    def test_actual_production_matching_slot_changes_only_T_Y(self):
        # Reuse the existing ordinary mixture fixture without importing pytest.
        tree=ast.parse((HERE/'test_mixture_transport.py').read_text())
        factory=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='fixture')
        environment={'re':re,'VALIDATION_ROOT':ROOT/'validation'}
        exec(compile(ast.Module(body=[factory],type_ignores=[]),'<mixture-fixture>','exec'),environment)
        fixture=environment['fixture']()
        extra=r"""
using MatchingModel=ugkwp::GasBoundaryLayerModelState<double,2,128>;
"""
        fixture=extra+fixture.replace('ugkwp::GasSpeciesState<Real,2> gasSpecies;};',
            'ugkwp::GasSpeciesState<Real,2> gasSpecies; MatchingModel gasBoundaryLayerModel;};')
        path=ROOT/'applications/gasUGKP/private_backend/ConservativeWallProjection.cuh'
        text=path.read_text();start=text.index('template <class State>\n__global__ void prepareConservativeWallMatching')
        end=text.index('\ntemplate <class State>',start+1)
        actual_producer=text[start:end]
        self.compile_run(r"""
#include "gasTransport/GasBuildConfig.H"
#include "gasTransport/GasConservativeMatching.H"
#include "gasTransport/GasBoundaryLayerModelState.H"
#include <cassert>
"""+fixture+actual_producer+r"""
int main(){State s;initialise(s);s.gasBoundaryLayer.count=1;
 ugkwp::ConservativeWallColumn columns[1];columns[0].cells[0]=0;columns[0].faces[2]=0;
 ugkwp::gaswall::WallMatchingState<double,2> matching[1];double pressure[1]{};
 double history=0;int storage=0;ugkwp::gaswall::WallHistoryFrame<2> frame;
 ugkwp::gaswall::WallReferenceStats stats;
 auto& model=s.gasBoundaryLayerModel;model.conservative.enabled=true;
 model.conservative.columns=columns;model.conservative.matching=matching;model.conservative.pressure=pressure;
 model.coherentState=&storage;model.referenceScratch=&storage;
 model.acceptedHistory=model.candidateHistory=&history;model.acceptedFrame=model.candidateFrame=&frame;
 model.referenceStats=&stats;model.config.model=ugkwp::gaswall::BoundaryLayerModel::ReactingSst;
 s.rho[0]=1;s.rho[1]=2;s.Tgas[0]=300;s.Tgas[1]=310;s.faceWeight[0]=.31;
 s.gasSpecies.rho[0]=.8;s.gasSpecies.rho[1]=1.68;s.gasSpecies.rho[2]=.2;s.gasSpecies.rho[3]=.32;
 s.k[0]=7;s.k[1]=11;s.omega[0]=13;s.omega[1]=17;s.Ux[0]=4;s.Ux[1]=6;
 for(int c=0;c<2;++c){Real Y[2]={s.gasSpecies.rho[c]/s.rho[c],s.gasSpecies.rho[2+c]/s.rho[c]};
  s.p[c]=s.rho[c]*ugkwp::mixtureGasConstant(Y,s.gasSpecies.thermo)*s.Tgas[c];}
 // Disabled, fully owned coherent route, then enabled but missing worker scratch.
 for(int route=0;route<3;++route)for(int donor=0;donor<2;++donor){
  const bool coherent=route==1;model.coherentEnabled=route!=0;
  model.coherentScratch=route==2?nullptr:&storage;
  assert(ugkwp::gasBoundaryLayerUsesCoherent(model,0)==coherent);
  columns[0].cells[1]=donor;
  GasPrimDevice old{};Real y[2]{};ck(reconstructGasMixtureFace(s,donor,0,old,y,false),"matching control failed");
  const auto physical=riemannFacePrimitiveForGradient(s,donor,0);
  blockIdx.x=threadIdx.x=0;blockDim.x=1;prepareConservativeWallMatching(&s);
  const auto& d=s.gasBoundaryLayerModel.conservative;const auto& m=d.matching[0];
  if(coherent)assert(m.temperature==physical.T&&m.temperature!=old.T);
  else assert(m.temperature==old.T);
  assert(m.velocity[0]==old.ux&&m.velocity[1]==old.uy&&m.velocity[2]==old.uz&&d.pressure[0]==old.p);
  assert(m.k==.31*s.k[0]+(1-.31)*s.k[1]);assert(m.omega==.31*s.omega[0]+(1-.31)*s.omega[1]);
  for(int j=0;j<2;++j){
   if(coherent)assert(m.massFraction[j]==.31*s.gasSpecies.rho[2*j]/s.rho[0]+(1-.31)*s.gasSpecies.rho[2*j+1]/s.rho[1]);
   else assert(m.massFraction[j]==y[j]);
  }
 }
}
""",production=True)


if __name__=='__main__':
    unittest.main()
