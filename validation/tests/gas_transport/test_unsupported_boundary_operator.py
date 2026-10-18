"""Unsupported mixture boundary operators must latch an actual fatal at their producer."""
from test_mixture_transport import fixture
from test_mixture_state import compile_probe


def test_missing_mixture_boundary_operator_is_fatal_in_both_execution_modes(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){
 for(bool native:{false,true})for(int mode=0;mode<4;++mode){
  State s;initialise(s);ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;
  const int face=1;s.gasBoundaryKind[face]=s.riemannBoundaryKind[face]=0;
  s.riemannBoundaryUFix[face]=mode==0?3:1;
  s.riemannBoundaryPWave[face]=mode==1?7:0;
  s.nScheduledInletFaces=mode==2?2:0;
  s.riemannBoundaryUx[face]=-2;s.riemannBoundaryUy[face]=s.riemannBoundaryUz[face]=0;
  s.riemannBoundaryPFix[face]=s.riemannBoundaryTFix[face]=1;
  s.riemannBoundaryP[face]=101325;s.riemannBoundaryT[face]=700;
  const auto owner=gasCellPrimitive(s,s.faceOwner[face]);
  const auto boundary=riemannBoundaryState(s,face,owner,nullptr,native);
  if(mode<3){
   ck(diagnostics.fatalClaim,"missing boundary operator did not latch fatal");
   const auto& record=diagnostics.firstFatal;
   ck(record.code==ugkp::diagnostics::Code::MissingOperator&&record.sourceSiteId==30420,"wrong producer diagnostic");
   ck(record.entityKind==ugkp::diagnostics::EntityKind::Face&&record.entityIndex==face,"missing face provenance");
   ck(record.component==mode&&record.observed==(mode==0?3:mode==1?7:2),"missing boundary selector evidence");
   ck(s.gasSpecies.faceStatus[face]==int(ugkwp::GasTransportCode::UnsupportedConfiguration),"coarse rejection lost");
   ck(boundary.rho==owner.rho&&boundary.p==owner.p&&boundary.T==owner.T&&boundary.ux==owner.ux,"rejected boundary continued evaluation");
  }else{
   ck(!diagnostics.fatalClaim&&s.gasSpecies.faceStatus[face]==0,"normal mixture boundary rejected");
   ck(boundary.ux==-2&&boundary.T==700&&boundary.p==101325,"normal boundary result changed");
  }
  diagnostics={};s.gasSpecies.faceStatus[face]=0;threadIdx.x=face;
  computeGasInternalFaceFluxKernel<false>(&s,1e-5,native);
  if(mode<3)ck(diagnostics.fatalClaim&&diagnostics.firstFatal.code==ugkp::diagnostics::Code::MissingOperator,"zero flux escaped missing-operator fatal");
  else ck(!diagnostics.fatalClaim&&s.gasSpecies.faceStatus[face]==0&&std::isfinite(s.gasPhiRho[face]),"normal face kernel failed");
 }
}
''')


def test_single_gas_total_inlet_operator_remains_available(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){
 for(bool native:{false,true}){
  State s;initialise(s);ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;
  s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;
  const int face=1;s.gasBoundaryKind[face]=s.riemannBoundaryKind[face]=0;
  s.riemannBoundaryUFix[face]=3;s.riemannBoundaryP[face]=101325;s.riemannBoundaryT[face]=900;
  const auto owner=gasCellPrimitive(s,0);
  const auto expected=ugkpboundary::totalConditionInletState(
   ugkpboundary::Primitive{owner.rho,owner.ux,owner.uy,owner.uz,owner.p,owner.T},
   Real(-1),Real(0),Real(0),Real(101325),Real(900),s.gammaGas,s.Rgas,s.rhoMin,s.TgasMin);
  const auto result=riemannBoundaryState(s,face,owner,nullptr,native);
  ck(!diagnostics.fatalClaim&&s.gasSpecies.faceStatus[face]==0,"legacy total inlet rejected");
  ck(result.rho==expected.rho&&result.ux==expected.ux&&result.p==expected.p,"legacy total inlet algebra changed");
  ck(std::abs(result.T-expected.T)<1e-12*expected.T,"legacy total inlet temperature changed");
  threadIdx.x=face;computeGasInternalFaceFluxKernel<false>(&s,1e-5,native);
  ck(!diagnostics.fatalClaim&&std::isfinite(s.gasPhiRho[face]),"legacy total inlet face flux failed");
 }
}
''')
