"""SST wall transport reports physical errors and retains diagnostic-only warnings."""
from test_mixture_state import compile_probe
from test_mixture_wall_function import wall_fixture


def test_wall_temperature_errors_record_original_value_and_stop(tmp_path):
    compile_probe(tmp_path,wall_fixture()+r'''
int main(){using namespace ugkp::diagnostics;
 for(bool trial:{false,true})for(bool mixture:{false,true})for(int fault=0;fault<3;++fault){
  State s;wall(s);if(!mixture)s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;
  DeviceContext d{};reset(d,19,8,23,.75);s.diagnostics=&d;
  const Real bad=fault==0?Real(-12):(fault==1?Real(0):std::numeric_limits<Real>::quiet_NaN());
  s.riemannBoundaryT[1]=bad;Real mu=77,k=78,q=79;int active=1;bool evaluated=true;
  gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active,trial,&evaluated);
  ck(hasFatal(&d),"wall temperature failure was not sticky");
  ck(d.firstFatal.code==(fault==0?Code::NegativeTemperature:(fault==1?Code::ZeroDenominator:Code::NonFinite)),"wrong temperature classification");
  ck(d.firstFatal.quantityId==Quantity::Temperature&&d.firstFatal.entityKind==EntityKind::Face&&d.firstFatal.entityIndex==1,"temperature location missing");
  ck(d.firstFatal.sourceSiteId>=2650&&d.firstFatal.sourceSiteId<=2669&&d.firstFatal.operatorId==26&&d.firstFatal.step==23,"wall provenance missing");
  ck(fault==2?std::isnan(d.firstFatal.observed):d.firstFatal.observed==bad,"wall observed temperature replaced");
  ck(!evaluated&&mu==0&&k==0&&q==0&&active==0,"failed wall result published");
 }
}
''')


def test_wall_finite_fractions_clip_and_continue_without_renormalization(tmp_path):
    compile_probe(tmp_path,wall_fixture()+r'''
int main(){using namespace ugkp::diagnostics;
 for(bool trial:{false,true})for(bool clip:{false,true}){
  State s;wall(s);DeviceContext d{};reset(d,20,8,24,.8);s.diagnostics=&d;
  s.gasSpecies.rho[0]=clip?Real(-1):Real(.2);s.gasSpecies.rho[2]=clip?Real(2):Real(.3);
  Real mu=0,k=0,q=0;int active=0;bool evaluated=false;
  gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active,trial,&evaluated);
  ck(!hasFatal(&d)&&evaluated&&active==1&&s.gasSpecies.faceStatus[1]==0,"finite composition blocked wall transport");
  ck(d.warnings[clip?2:0].count==(clip?2u:1u),"wall composition warning missing");
  const Real ya=clip?Real(0):Real(.2),yb=clip?Real(1):Real(.3);
  const Real cp=ya*1100+yb*1600,R=ugkwp::universalGasConstant<Real>()*(ya/Real(.028)+yb/Real(.04))/(ya+yb);
  const auto expected=ugkpwall::sstJayatillekeThermalTransport(s.p[0]/(R*500),cp,s.gasMu,s.gasPrClamped,s.turbulentPrandtl,
   s.sstWallCmu,s.sstWallKappa,s.sstWallE,s.sstJayatillekeP,s.sstThermalYPlus,s.k[0],Real(.5),Real(10),Real(2),Real(-400));
  near(q,expected.heatFlux,"wall composition was not clipped componentwise or was renormalized");
  ck(s.gasSpecies.rho[0]==(clip?Real(-1):Real(.2))&&s.gasSpecies.rho[2]==(clip?Real(2):Real(.3)),"wall closure rewrote transported species");
 }
}
''')


def test_wall_table_range_is_warning_and_uses_original_temperature(tmp_path):
    compile_probe(tmp_path,wall_fixture()+r'''
int main(){using namespace ugkp::diagnostics;
 for(bool trial:{false,true}){
  State s;wall(s);DeviceContext d{};s.diagnostics=&d;s.riemannBoundaryT[1]=4000;
  Real mu=0,k=0,q=0;int active=0;bool evaluated=false;
  gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active,trial,&evaluated);
  ck(!hasFatal(&d)&&evaluated&&active==1&&std::isfinite(q),"finite table extrapolation rejected");
  ck(d.warnings[1].count==2&&d.warnings[1].sample.observed==4000,"table accuracy warning missing");
 }
}
''')


def test_wall_property_faults_are_classified_at_wall_producer(tmp_path):
    compile_probe(tmp_path,wall_fixture()+r'''
int main(){using namespace ugkp::diagnostics;
 for(bool missing:{false,true}){
  State s;wall(s);DeviceContext d{};reset(d,21,8,25,.9);s.diagnostics=&d;s.gasSpecies.thermo.diagnostics=&d;
  if(missing)s.gasSpecies.thermo.coefficients=nullptr;
  else const_cast<Real*>(s.gasSpecies.thermo.coefficients)[3]=std::numeric_limits<Real>::quiet_NaN();
  Real mu=0,k=0,q=0;int active=0;bool evaluated=false;
  gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active,false,&evaluated);
  ck(hasFatal(&d)&&!evaluated&&active==0,"wall property fault continued");
  ck(d.firstFatal.code==(missing?Code::MissingOperator:Code::NonFinite),"property failure misclassified");
  ck(d.firstFatal.sourceSiteId>=2650&&d.firstFatal.sourceSiteId<=2669&&d.firstFatal.operatorId==26&&d.firstFatal.step==25,"property producer provenance missing");
  if(!missing)ck(d.firstFatal.component==1&&std::isnan(d.firstFatal.observed),"nonfinite species property value lost");
 }
}
''')


def test_wall_zero_viscosity_stops_before_division_and_nan_velocity_is_not_repaired(tmp_path):
    compile_probe(tmp_path,wall_fixture()+r'''
#include <cfenv>
int main(){using namespace ugkp::diagnostics;
 for(bool trial:{false,true})for(bool velocity:{false,true}){
  State s;wall(s);DeviceContext d{};s.diagnostics=&d;
  if(velocity)s.riemannBoundaryUy[1]=std::numeric_limits<Real>::quiet_NaN();else s.gasMu=0;
  Real mu=77,k=78,q=79;int active=1;bool evaluated=true;
  if(!velocity)feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);
  gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,active,trial,&evaluated);
  fedisableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);
  ck(hasFatal(&d)&&!evaluated&&mu==0&&k==0&&q==0&&active==0,"invalid wall arithmetic proceeded");
  ck(d.firstFatal.code==(velocity?Code::NonFinite:Code::ZeroDenominator),"wrong wall arithmetic classification");
  if(velocity)ck(d.firstFatal.quantityId==Quantity::Velocity&&d.firstFatal.component==1&&std::isnan(d.firstFatal.observed),"velocity failure lost");
 }
}
''')


def test_wall_and_bulk_transport_do_not_hide_nonfinite_inputs(tmp_path):
    compile_probe(tmp_path,wall_fixture()+r'''
int main(){using namespace ugkp::diagnostics;
 for(int fault=0;fault<7;++fault){
  State s;wall(s);DeviceContext d{};s.diagnostics=&d;const Real nan=std::numeric_limits<Real>::quiet_NaN();
  if(fault==0)s.k[0]=nan;
  if(fault==1)s.turbulentPrandtl=nan;
  if(fault==2)s.sstWallDistance[0]=nan;
  if(fault==3)s.gasThermalConductivity=0;
  if(fault==4)s.gasSpecies.heatCapacity[0]=nan;
  if(fault==5)s.nut[0]=nan;
  if(fault==6){s.turbulenceModel=1;s.gasCp=nan;s.gasSpecies.mode=ugkwp::GasMode::SingleLegacy;}
  Real mu=77,k=78,q=79;int active=1;bool evaluated=true;
  gasFaceSubgridTransportProperties(s,fault==5?0:1,0,fault==5?1:-1,fault==5?0:2,Real(1),mu,k,q,active,false,&evaluated);
  ck(hasFatal(&d)&&!evaluated&&mu==0&&k==0&&q==0&&active==0,"nonfinite transport input was masked or published");
  ck(d.firstFatal.code==(fault==3?Code::ZeroDenominator:Code::NonFinite),"transport input classification missing");
 }
 State s;wall(s);DeviceContext d{};s.diagnostics=&d;s.nut[0]=s.nut[1]=-3;
 Real mu=77,k=78,q=79;int active=1;bool evaluated=false;
 gasFaceSubgridTransportProperties(s,0,0,1,0,Real(1),mu,k,q,active,false,&evaluated);
 ck(!hasFatal(&d)&&evaluated&&mu==0&&k==0,"original finite eddy-viscosity limiter changed");
}
''')
