"""Physical face failures latch diagnostics and do not publish invalid fluxes."""
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


def test_face_nonfinite_stops_before_bulk_publication(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};
 ugkp::diagnostics::reset(d,9,5,17,.3);s.diagnostics=&d;
 s.rhoE[0]=std::numeric_limits<Real>::quiet_NaN();Real out[5]={};
 ck(!computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]),"invalid result accepted");
 ck(ugkp::diagnostics::hasFatal(&d),"face failure not sticky");
 ck(d.firstFatal.step==17&&d.firstFatal.operatorId!=0,"face provenance lost");
 for(int k=0;k<5;++k)ck(std::isfinite(out[k]),"invalid bulk flux published");
}
''')


def test_generic_status_preserves_first_error(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 ugkwp::gasRecordFaceFailure(s,0,ugkwp::GasTransportCode::NonFiniteState);
 ck(ugkp::diagnostics::hasFatal(&d),"nonfinite status not mapped");
 const auto first=d.firstFatal;ugkwp::gasRecordFaceFailure(s,1,ugkwp::GasTransportCode::InvalidStorage);
 ck(d.firstFatal.entityIndex==first.entityIndex&&d.firstFatal.code==first.code,"first error overwritten");
}
''')


def test_unimplemented_mixture_selector_has_no_substitute_flux(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.gasFluxScheme=7;Real out[5]={};s.gasSpecies.flux[0]=123;
 ck(!computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]),"unimplemented selector remapped");
 ck(d.firstFatal.code==ugkp::diagnostics::Code::MissingOperator,"wrong selector diagnostic");
 ck(s.gasSpecies.flux[0]==123,"substitute species flux published");
}
''')


def test_boundary_finite_composition_warns_without_fatal(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext d{};s.diagnostics=&d;
 s.gasSpecies.compositionBoundaryFixed[1]=1;s.gasSpecies.boundaryMassFraction[1]=1.1;s.gasSpecies.boundaryMassFraction[4]=-.1;
 const auto out=riemannBoundaryState(s,1,gasCellPrimitive(s,0));
 ck(!ugkp::diagnostics::hasFatal(&d),"finite boundary fractions fatal");
 ck(d.warnings[2].count==2,"boundary clipping unreported");ck(std::isfinite(out.rho),"boundary result unavailable");
}
''')
