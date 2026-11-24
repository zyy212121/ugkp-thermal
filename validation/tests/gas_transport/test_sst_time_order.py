import pytest
from test_mixture_state import compile_probe, ROOT
from test_legacy_ale_transport import fixture



def source_fixture():
    text = fixture()
    if "class Geometry = Real" in (ROOT / "common/gasTransport/GasStateView.H").read_text():
        text = text.replace("GasStateView<Real,Time,ugkwp::SstCoefficients>",
                            "GasStateView<Real,Time,ugkwp::SstCoefficients,Real,double,double>")
    return text[:text.index("void moving")]


@pytest.mark.parametrize('order', [1, 2, 3])
def test_lowre_decay_matches_requested_time_order(tmp_path, order):
    compile_probe(tmp_path, source_fixture()+r'''
double evolve(int steps){
 State s;initialise(s);s.sstConfigured=1;s.sstWallTreatment=0;
 for(int c=0;c<2;++c){s.rho[c]=2;s.k[c]=3;s.omega[c]=4;
  s.rhoK[c]=6;s.rhoOmega[c]=8;s.sstF1[c]=1;s.sstF2[c]=1;}
 const double end=1,dt=end/steps;
 auto stage=[&](){
  applySstFluxAndSourceKernel(&s,dt,false,ORDER==1);
  recoverSstPrimitivesKernel(&s);
 };
 auto blend=[&](double a,double b){
  blendGasConservativeStateKernel(&s,a,b);
  recoverSstPrimitivesKernel(&s);
 };
 for(int step=0;step<steps;++step){
  saveGasConservativeStateKernel(&s);
  stage();
  if(ORDER>1){stage();blend(ORDER==2?.5:.75,ORDER==2?.5:.25);}
  if(ORDER==3){stage();blend(1./3,2./3);}
 }
 const double omegaExact=4/(1+s.sstCoefficients.beta1*4*end);
 const double kExact=3*std::pow(1+s.sstCoefficients.beta1*4*end,
                             -s.sstCoefficients.betaStar/s.sstCoefficients.beta1);
 ck(std::abs(s.omega[0]-omegaExact)<.1,"omega inconsistent");
 return std::hypot(double(s.k[0])-kExact,double(s.omega[0])-omegaExact);
}
int main(){
 const double coarse=evolve(4),fine=evolve(8);
 const double observed=std::log(coarse/fine)/std::log(2.);
 std::printf("SSPRK/Euler order %d errors %.9g %.9g observed %.6f",ORDER,coarse,fine,observed);
 ck(observed>ORDER-.3 && observed<ORDER+.4,"low-Re time order degraded");
}
'''.replace('ORDER',str(order)))


def test_rk_ale_source_uses_explicit_inventory_update(tmp_path):
    compile_probe(tmp_path, source_fixture()+r'''
int main(){
 State s;initialise(s);s.sstConfigured=1;s.sstWallTreatment=0;
 using Geometry=std::remove_const_t<std::remove_pointer_t<decltype(s.gasGeometry.oldVolume)>>;
 Geometry oldV[2]={1,1},newV[2]={Geometry(1.1),Geometry(.9)},sweeps[3]={Geometry(.1),0,0};
 const double dt=.001;
 s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;
 s.gasGeometry.faceSweptVolume=sweeps;s.gasGeometry.interval=dt;
 s.gasGeometry.absoluteGeometryTolerance=Real(32)*std::numeric_limits<Real>::epsilon();
 s.gasGeometry.relativeGeometryTolerance=Real(32)*std::numeric_limits<Real>::epsilon();
 s.gasPhiRho[0]=-s.rho[0]*sweeps[0]/dt;
 const Real initialK=s.rhoK[0],initialOmega=s.rhoOmega[0];
 Real sk=0,so=0;ck(sstSourcesForCell(s,0,Real(0),sk,so),"source evaluation");
 applySstFluxAndSourceKernel(&s,dt,false,false);
 near(s.rhoK[0]*newV[0],initialK*oldV[0]+dt*oldV[0]*sk,"RK moving k source");
 near(s.rhoOmega[0]*newV[0],initialOmega*oldV[0]+dt*oldV[0]*so,"RK moving omega source");
}
''')
