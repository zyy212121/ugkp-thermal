"""Exercise the production input adapter and projected macro source together."""
from test_mixture_state import compile_probe
from test_mixture_transport import fixture
from test_boundary_layer_operators import SETUP


def model_fixture():
    return fixture().replace(
        'struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2> gasSpecies;};',
        '#include "gasTransport/GasBoundaryLayerModelState.H"\n'
        'struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{'
        'ugkwp::GasSpeciesState<Real,2> gasSpecies;'
        'ugkwp::GasBoundaryLayerModelState<Real,2> gasBoundaryLayerModel;};') + SETUP + r'''
struct ModelStorage {
 ugkwp::gaswall::WallInput<Real,2> input[1];
 ugkwp::gaswall::WallOutput<Real,2> output[1];
 ugkwp::gaswall::WallStatus status[1];
 int faces[1]={1},offsets[2]={0,1},matching[1]={1};
 Real weights[1]={1},height[1]={.5},volume[1]={1},diffusion[2]={.01,.01};
 unsigned char workspace[1];
 void bind(State& s){
  wall(s);s.gasMu=.01;s.gasSpecies.diffusivity=diffusion;s.sstCoefficients=ugkwp::defaultSstCoefficients();
  for(int i=0;i<2;++i){s.rhoK[i]=1;s.rhoOmega[i]=2;}
  s.riemannBoundaryT[1]=500;
  auto& m=s.gasBoundaryLayerModel;m.config.enableSst=true;
  m.config.model=ugkwp::gaswall::BoundaryLayerModel::ReactingSst;m.volumeSstClosure=true;
  m.input=input;m.output=output;m.status=status;m.faces=faces;
  m.matchingOffsets=offsets;m.matchingCells=matching;m.matchingWeights=weights;
  m.workspace=workspace;m.workspaceCount=1;
  input[0].normal[0]=.6;input[0].normal[1]=.8;
  input[0].matchingDistance=1;input[0].ownerDistance=.5;
  input[0].quadrature={height,volume,1,1,.5};
 }
};
void close(Real actual,Real expected){
 if(std::abs(actual-expected)>=1e-12*(1+std::abs(expected))){std::fprintf(stderr,"tensor component: %.17g expected %.17g\n",double(actual),double(expected));std::abort();}
}
'''


def test_prepare_input_rotates_physical_gradient_at_actual_owner(tmp_path):
    compile_probe(tmp_path, model_fixture() + r'''
int main(){State s;initialise(s);ModelStorage storage;storage.bind(s);
 // n=(.6,.8,0), t0=(0,0,1), t1=(.8,-.6,0). Global tensor below is
 // Q^T*[[1,2,3],[4,5,6],[7,8,9]]*Q, independently expanded by hand.
 s.faceOwner[1]=1;
 s.gradUxX[1]=13.16;s.gradUxY[1]=2.88;s.gradUxZ[1]=7.4;
 s.gradUyX[1]=4.88;s.gradUyY[1]=.84;s.gradUyZ[1]=3.2;
 s.gradUzX[1]=3.4;s.gradUzY[1]=1.2;s.gradUzZ[1]=1;
 Real sweep=99;ck(ugkwp::prepareGasBoundaryLayerInput(s,0,0,sweep),"input rejected");
 const Real expected[3][2]={{1,2},{4,5},{7,8}};
 for(int i=0;i<3;++i)for(int j=0;j<2;++j)close(storage.input[0].sstTangentialGradient[i][j],expected[i][j]);
 close(sweep,0);
 // Rotated local diagonal trace and all cross terms reach the real source.
 auto& in=storage.input[0];in.model.sstCoefficients.a1=100;in.model.sstCoefficients.c1=100000;
 ugkwp::gaswall::detail::LayerPoint<Real,2> p;p.rho=1;p.k=1;p.omega=1;p.z=1;
 ugkwp::gaswall::detail::applySstGradients(in,Real(1),Real(3),Real(6),Real(9),Real(0),Real(0),p);
 // 2*(1+25+81)+(2+4)^2+(3+7)^2+(6+8)^2 = 546;
 // G=546-2/3*(1+5+9)^2=396, compression=10.
 close(p.nut,1);close(p.sourceK,Real(396)-10-in.model.sstCoefficients.betaStar);
 // Stage refresh replaces every component; disabling the volume closure
 // clears the borrowed tensor and no longer requires gradient storage.
 s.gasBoundaryLayerModel.volumeSstClosure=false;s.gradUxX=nullptr;
 ck(ugkwp::prepareGasBoundaryLayerInput(s,0,0,sweep),"non-volume input demanded gradient storage");
 for(int i=0;i<3;++i)for(int j=0;j<2;++j)close(storage.input[0].sstTangentialGradient[i][j],0);
}
''')


def test_full_projected_source_is_not_augmented_by_macro_production_or_compression(tmp_path):
    compile_probe(tmp_path, model_fixture() + r'''
int main(){State s;initialise(s);ModelStorage storage;storage.bind(s);
 auto& closure=s.gasBoundaryLayer.sst[0];closure.kFluxCount=3;
 closure.nonDestructiveKSource=17;closure.ownerOmega=5;
 s.gasBoundaryLayer.exchange[0].ready=true;
 s.rho[0]=2;s.k[0]=3;s.rhoK[0]=6;s.omega[0]=99;s.nut[0]=2;
 for(Real gradient:{Real(-100),Real(0),Real(100)}){
  s.gradUxX[0]=gradient;s.gradUxY[0]=gradient;s.gradUyX[0]=-2*gradient;
  for(Real div:{Real(-77),Real(0),Real(77)}){
   Real sk,so;ck(sstSourcesForCell(s,0,div,sk,so),"macro source rejected");
   close(sk,Real(17)-s.sstCoefficients.betaStar*Real(6)*5);
  }
 }
}
''')


def test_volume_input_missing_face_owner_is_rejected_without_dereference(tmp_path):
    compile_probe(tmp_path, model_fixture() + r'''
int main(){State s;initialise(s);ModelStorage storage;storage.bind(s);
 s.faceOwner=nullptr;Real sweep=0;
 ck(!ugkwp::prepareGasBoundaryLayerInput(s,0,0,sweep),"missing faceOwner accepted");
 ck(s.gasBoundaryLayer.status[0]==int(ugkwp::GasTransportCode::InvalidStorage),"wrong missing-storage status");
}
''')
