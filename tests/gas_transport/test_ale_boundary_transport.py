"""Production boundary helpers must use mesh-relative normal motion."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('legacy',[False,True])
def test_moving_wall_gradients_pressure_work_and_inlet_outlet_direction(tmp_path,bits,legacy):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);if(LEGACY)s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;
Real oldV[2]={1,1},newV[2]={1.1,1.1},sweep[3]={0,.1,.1};
s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.001;s.gasGeometry.absoluteGeometryTolerance=1e-7;s.gasGeometry.relativeGeometryTolerance=1e-6;
s.Ux[0]=-50;s.Uy[0]=2;s.riemannBoundaryUFix[1]=2;s.riemannBoundaryUx[1]=7;s.riemannBoundaryUy[1]=3;
auto owner=gasCellPrimitive(s,0);ck(useRiemannBoundaryVelocity(s,1,owner),"inletOutlet ignored mesh-relative inflow");
s.riemannBoundaryKind[1]=1;auto slip=riemannFacePrimitiveForGradient(s,0,1);
ck(std::abs(slip.ux+100)<Real(2e-5)&&slip.uy==2,"slip wall normal velocity is not mesh velocity");
s.riemannBoundaryKind[1]=2;auto wall=riemannFacePrimitiveForGradient(s,0,1);
ck(std::abs(wall.ux+100)<Real(2e-5)&&wall.uy==3,"no-slip wall lost tangential prescription or normal motion");
threadIdx.x=1;computeGasInternalFaceFluxKernel<false>(&s,.001);
ck(s.gasSpecies.faceStatus[1]==0&&s.gasPhiRho[1]==0,"moving wall leaks mass");
ck(std::abs(s.gasPhiRhoE[1]-s.p[0]*100)<Real(2e-6)*std::abs(s.p[0]*100),"wall pressure work is not p times sweep rate");
s.gasGeometry.faceSweptVolume=nullptr;s.gasSpecies.faceStatus[1]=0;riemannFacePrimitiveForGradient(s,0,1);ck(s.gasSpecies.faceStatus[1]!=0,"invalid boundary geometry unreported");
}
'''.replace('LEGACY','true' if legacy else 'false'),bits)
