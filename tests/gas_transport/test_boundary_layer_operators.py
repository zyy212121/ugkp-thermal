"""Production gas/SST operators consume exactly one physical wall closure."""
import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture
from test_sst_ale_audit import AUDIT

SETUP=r'''
void wall(State&s){s.sstWallTreatment=2;s.sstConfigured=1;s.turbulenceModel=3;s.riemannBoundaryKind[1]=2;
auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;w.faceSlot=new int[3]{-1,0,-1};w.ownerSlot=new int[2]{0,-1};w.status=new int[1]{};
w.exchange=new ugkwp::GasBoundaryLayerExchange<Real>[1];w.sst=new ugkwp::GasBoundaryLayerSstClosure<Real>[1];w.speciesFlux=new Real[2]{};
w.exchange[0].matchingPressure=s.p[0];w.exchange[0].ready=true;w.exchange[0].massReady=true;w.exchange[0].mass=-.4;w.exchange[0].momentumX=-17;w.exchange[0].momentumY=3;w.exchange[0].energy=-31;w.exchange[0].k=-2;
w.sst[0]={-3,11,1};}
'''

@pytest.mark.parametrize('bits',[32,64])
def test_selected_wall_replaces_k_once_and_keeps_six_channel_budget(tmp_path,bits):
    compile_probe(tmp_path,fixture()+AUDIT+SETUP+r'''
int main(){State s;initialise(s);wall(s);audit(s);
threadIdx.x=0;applySstWallFunctionStateKernel(&s);near(s.omega[0],11,"new owner omega not applied");near(s.rhoK[0],6,"k was projected");
threadIdx.x=1;computeSstFaceFluxKernel(&s);near(s.sstPhiRhoK[1],-2,"physical wall k flux not used");
s.sstPhiRhoK[0]=.5;s.sstPhiRhoOmega[0]=.7;
threadIdx.x=0;applySstFluxAndSourceKernel(&s,.1);near(s.rhoK[0],Real(5.85),"complete k source was not replaced exactly once");recoverSstPrimitivesKernel(&s);
near(s.rhoOmega[0],22,"finite owner constraint not retained");near(s.gasSstAudit.sourceK[0],Real(-.3),"physical volume source audit wrong");
checkBudget(s,6,8,0);ck(s.gasSstAudit.transportOmega[0]!=0,"replaced omega transport row omitted");ck(s.gasSstAudit.sourceOmega[0]!=0,"replaced omega source row omitted");
Real saved=s.rhoOmega[0];recoverSstPrimitivesKernel(&s);near(s.rhoOmega[0],saved,"omega projection not idempotent");checkBudget(s,6,8,0);
}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_wall_flux_and_mass_predictor_have_distinct_readiness(tmp_path,bits):
    compile_probe(tmp_path,fixture()+SETUP+r'''
int main(){State s;initialise(s);wall(s);Real m,x,y,z,e;
ck(computeRiemannGasFaceFluxDevice<true>(s,1,m,x,y,z,e),"ready physical wall rejected");near(m,Real(-.4),"wall mass missing");near(x,-17,"wall momentum missing");near(e,-31,"wall energy missing");
s.gasBoundaryLayer.exchange[0].ready=false;
ck(computeRiemannGasFaceFluxDevice<true,true>(s,1,m,x,y,z,e),"known prescribed mass forced thermal solve");near(m,Real(-.4),"mass predictor ignored prescribed Js");
ck(!computeRiemannGasFaceFluxDevice<true>(s,1,m,x,y,z,e),"unprepared wall silently fell back");ck(s.gasSpecies.faceStatus[1]!=0,"unprepared wall did not reject trial");
}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_owner_geometry_mismatch_rejects_before_inventory_write(tmp_path,bits):
    compile_probe(tmp_path,fixture()+SETUP+r'''
int main(){State s;initialise(s);wall(s);s.gasBoundaryLayer.sst[0].volume=2;
threadIdx.x=0;applySstFluxAndSourceKernel(&s,.01);ck(s.gasSpecies.cellStatus[0]!=0,"wrong stage volume accepted");near(s.rhoK[0],6,"bad source modified inventory");near(s.rhoOmega[0],8,"bad source modified omega");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_new_profile_source_exceeding_explicit_limit_rejects_before_update(tmp_path,bits):
    compile_probe(tmp_path,fixture()+SETUP+r'''
int main(){State s;initialise(s);wall(s);s.gasBoundaryLayer.sst[0].integratedKSource=-100;
threadIdx.x=0;applySstFluxAndSourceKernel(&s,.1);ck(s.gasSpecies.cellStatus[0]!=0,"new profile source timestep limit ignored");near(s.rhoK[0],6,"failed source limit wrote k");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_prescribed_wall_flux_is_rejected_rather_than_partially_scaled(tmp_path,bits):
    from test_mixture_transport import fixture as mixture
    compile_probe(tmp_path,mixture()+SETUP+r'''
int main(){State s;initialise(s);wall(s);s.gasPhiRho[1]=0;s.gasPhiRhoE[1]=17;
s.gasSpecies.flux[1]=.5;s.gasSpecies.flux[4]=-.5;s.gasSpecies.positivityScale[0]=.2;
threadIdx.x=1;applyGasFluxPositivityScaleKernel(&s);
ck(s.gasSpecies.faceStatus[1]==int(ugkwp::GasTransportCode::NegativeInventory),"prescribed counterflow donor excess did not reject trial");
ck(s.gasSpecies.flux[1]==.5&&s.gasSpecies.flux[4]==-.5&&s.gasPhiRhoE[1]==17,"fixed physical exchange was partially scaled");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_outer_source_estimate_reuses_accepted_closure_but_trial_requires_current_profile(tmp_path,bits):
    compile_probe(tmp_path,fixture()+SETUP+r'''
int main(){State s;initialise(s);wall(s);threadIdx.x=0;
s.gasBoundaryLayer.exchange[0].ready=false;
computeSstStabilityNumberKernel(&s,Real(.1),Real(.5),false);
ck(s.gasSpecies.cellStatus[0]==0&&s.sstSourceNumber[0]==0,"cold outer estimate demanded an extra BVP");
computeSstStabilityNumberKernel(&s,Real(.1),Real(.5),true);
ck(s.gasSpecies.cellStatus[0]==int(ugkwp::GasTransportCode::InvalidStorage),"authoritative trial accepted missing profile");
s.gasSpecies.cellStatus[0]=0;s.gasBoundaryLayer.exchange[0].ready=true;
s.gasBoundaryLayer.sst[0].integratedKSource=-12;
computeSstStabilityNumberKernel(&s,Real(.1),Real(.5),false);
near(s.sstSourceNumber[0],Real(.4),"outer estimate ignored accepted source in equivalent-Courant units");
computeSstStabilityNumberKernel(&s,Real(.1),Real(.5),true);
near(s.sstSourceNumber[0],Real(.4),"trial source bound differs from published integral in equivalent-Courant units");}
''',bits)


@pytest.mark.parametrize('bits',[32,64])
def test_optional_full_budget_audit_does_not_change_physical_wall_update(tmp_path,bits):
    compile_probe(tmp_path,fixture()+AUDIT+SETUP+r'''
int main(){Real accepted[2][2]{};for(int enabled=0;enabled<2;++enabled){State s;initialise(s);wall(s);audit(s);s.gasSstAudit.enabled=enabled;
threadIdx.x=0;applySstWallFunctionStateKernel(&s);threadIdx.x=1;computeSstFaceFluxKernel(&s);
s.sstPhiRhoK[0]=.5;s.sstPhiRhoOmega[0]=.7;threadIdx.x=0;applySstFluxAndSourceKernel(&s,.1);recoverSstPrimitivesKernel(&s);
near(s.rhoK[0],Real(5.85),"physical wall k source changed with audit");near(s.rhoOmega[0],22,"owner omega changed with audit");
accepted[enabled][0]=s.rhoK[0];accepted[enabled][1]=s.rhoOmega[0];if(enabled)checkBudget(s,6,8,0);
}ck(accepted[0][0]==accepted[1][0]&&accepted[0][1]==accepted[1][1],"diagnostic audit changed physical result");}
''',bits)
