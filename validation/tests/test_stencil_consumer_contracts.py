"""Run real producer/consumer headers; no hand-written substitute operators."""
import pytest
from test_final_diagnostic_integration import compile_run, mixture_fixture


@pytest.mark.parametrize('native', [False, True])
def test_unsigned_courant_needs_face_ids_not_topology(tmp_path, native):
    compile_run(tmp_path, mixture_fixture() + r"""
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.faceOwner=nullptr;s.faceNeighbour=nullptr;s.facePeriodicPair=nullptr;
 s.gasPhiRhoE[0]=2;s.gasPhiRhoE[1]=4;s.gasFluxPositivityScale[0]=-99;
 threadIdx.x=0;computeGasConvectiveCourantByCellKernel(&s,.1,NATIVE);
 ck(!ugkp::diagnostics::hasFatal(&d)&&s.gasSpecies.cellStatus[0]==0,"unsigned Courant read unrelated topology");
 ck(std::abs(s.gasFluxPositivityScale[0]-.3)<1e-14,"unsigned spectral sum changed");}
""".replace('NATIVE', 'true' if native else 'false'))


@pytest.mark.parametrize('bad_face', [-1, 3])
def test_unsigned_courant_rejects_real_face_array_oob(tmp_path, bad_face):
    compile_run(tmp_path, mixture_fixture() + r"""
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.cellFaceId[0]=BAD_FACE;s.gasFluxPositivityScale[0]=-99;
 threadIdx.x=0;computeGasConvectiveCourantByCellKernel(&s,.1);
 ck(ugkp::diagnostics::hasFatal(&d)&&d.firstFatal.sourceSiteId==30000,"face OOB lost diagnostic");
 ck(s.gasFluxPositivityScale[0]==-99,"invalid face published Courant");}
""".replace('BAD_FACE',str(bad_face)))


@pytest.mark.parametrize('native', [False, True])
def test_local_boundary_primitive_does_not_need_owner_neighbour(tmp_path,native):
    compile_run(tmp_path, mixture_fixture() + r"""
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.faceOwner=nullptr;s.faceNeighbour=nullptr;
 auto q=riemannFacePrimitiveForGradient(s,0,1,NATIVE);
 ck(!ugkp::diagnostics::hasFatal(&d)&&s.gasSpecies.cellStatus[0]==0,"local boundary required unrelated owner/neighbour");
 ck(q.rho==s.rho[0]&&q.T==s.Tgas[0],"local boundary primitive changed");}
""".replace('NATIVE','true' if native else 'false'))


@pytest.mark.parametrize('boundary', [1,3,4])
def test_zero_sst_face_channel_needs_no_cell_topology(tmp_path,boundary):
    compile_run(tmp_path, mixture_fixture() + r"""
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.sstConfigured=1;s.riemannBoundaryKind[1]=BOUNDARY;
 s.faceOwner=nullptr;s.faceNeighbour=nullptr;s.sstPhiRhoK[1]=7;s.sstPhiRhoOmega[1]=9;
 threadIdx.x=1;computeSstFaceFluxKernel(&s);
 ck(!ugkp::diagnostics::hasFatal(&d),"zero SST channel required unrelated topology");
 ck(s.sstPhiRhoK[1]==0&&s.sstPhiRhoOmega[1]==0,"zero SST channel did not publish zeros");}
""".replace('BOUNDARY',str(boundary)))




def test_enabled_cell_read_still_rejects_bad_neighbour(tmp_path):
    compile_run(tmp_path, mixture_fixture() + r"""
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.faceNeighbour[0]=s.nCells;
 auto q=riemannFacePrimitiveForGradient(s,0,0);
 ck(ugkp::diagnostics::hasFatal(&d)&&d.firstFatal.sourceSiteId==30002,"enabled neighbour OOB not diagnosed");
 ck(std::isnan(q.rho),"invalid cell read returned usable primitive");}
""")


@pytest.mark.parametrize("limiter",[0,1])
def test_mixture_limiter_local_boundary_uses_no_owner_neighbour(tmp_path,limiter):
    compile_run(tmp_path, mixture_fixture() + r"""
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.cellPlaneCount[0]=1;s.cellFaceId[0]=1;s.faceOwner=nullptr;s.faceNeighbour=nullptr;s.gasLimiter=LIMITER;
 threadIdx.x=0;computeGasGradientLimiterKernel(&s);
 ck(!ugkp::diagnostics::hasFatal(&d)&&s.gasSpecies.cellStatus[0]==0,"local limiter read unrelated topology");
 ck(s.gasSpecies.limiter[0]==1&&s.gasGradientLimiterRho[0]==1,"uniform limiter changed");}
""".replace("LIMITER",str(limiter)))




def test_wall_classification_ignores_nonwall_cell_topology(tmp_path):
    compile_run(tmp_path, mixture_fixture() + r"""
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.cellPlaneCount[0]=1;s.cellFaceId[0]=1;s.riemannBoundaryKind[1]=1;
 s.sstConfigured=1;s.sstWallTreatment=0;s.omega[0]=3;
 s.faceOwner=nullptr;s.faceNeighbour=nullptr;
 threadIdx.x=0;applySstWallFunctionStateKernel(&s);
 ck(!ugkp::diagnostics::hasFatal(&d)&&s.gasSpecies.cellStatus[0]==0,"nonwall classification read unrelated topology");
 ck(s.omega[0]==3,"nonwall classification changed omega");}
""")
