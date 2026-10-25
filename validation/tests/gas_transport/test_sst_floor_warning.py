"""Actual SST update/recovery clipping warns without changing continuation.

Serial host execution of production operators; no CUDA execution claim.
"""
import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture


@pytest.mark.parametrize('moving_mesh', [False, True])
@pytest.mark.parametrize('component', [0, 1])
@pytest.mark.parametrize('target', ['-2.206', '0', '1e-13', '3'])
def test_update_floor_records_raw_value_and_existing_print_switch(tmp_path, moving_mesh, component, target):
    compile_probe(tmp_path, fixture()+r'''
#include "DiagnosticsCatalogue.H"
int main(){
 using namespace ugkp::diagnostics;
 State s;initialise(s);DeviceContext d{};s.diagnostics=&d;s.sstConfigured=1;
 const Real dt=.001,target=TARGET;const int component=COMPONENT;
 Real oldV[2]={1,1},newV[2]={1.1,.9},sweeps[3]={.1,0,0};
 if(MOVING)moving(s,oldV,newV,sweeps,dt);
 // Set a finite finalized transport flux so the actual source+flux update
 // reaches the chosen raw inventory, independently of its subsequent floor.
 Real sk,so;ck(sstSourcesForCell(s,0,MOVING?Real(100):Real(0),sk,so),"source evaluation");
 const Real ov=1,nv=MOVING?newV[0]:Real(1);
 const Real initial=component?s.rhoOmega[0]:s.rhoK[0];
 const Real source=component?so:sk;
 const Real flux=(initial*ov+dt*ov*source-target*nv)/dt;
 if(component)s.sstPhiRhoOmega[0]=flux;else s.sstPhiRhoK[0]=flux;
 applySstFluxAndSourceKernel(&s,dt);
 const Real floor=s.rho[0]*(component?s.sstOmegaMin:s.sstKMin)*ov/nv;
 const Real result=component?s.rhoOmega[0]:s.rhoK[0];
 near(result,std::max(target,floor),"floor changed");
 ck(!hasFatal(&d)&&s.gasSpecies.cellStatus[0]==0,"warning changed success status");
 auto snapshot=takeSnapshot(d,sourceCatalogue());
 if(target<floor){
  ck(snapshot.warnings.size()==1,"missing clipping warning");
  const auto&w=snapshot.warnings[0];ck(w.count==1,"unexpected warning count");
  ck(w.sample.severity==Severity::Warning&&w.sample.code==Code::AccuracyResidual,"classification");
  ck(w.sample.sourceSiteId==2623&&w.sample.entityIndex==0&&w.sample.component==component,"provenance");
  near(w.sample.observed,target,"raw value lost");near(w.sample.reference,floor,"floor lost");near(w.sample.corrected,floor,"correction lost");
 }else ck(snapshot.warnings.empty(),"healthy update warned");
 std::string printed;HostSink sink;sink.user=&printed;
 sink.warning=[](void*p,const WarningSummary&w,const SourceDescriptor*source){*static_cast<std::string*>(p)+=formatRecord(w.sample,source);};
 sink.warningsEnabled=false;dispatch(snapshot,sink);ck(printed.empty(),"warning off printed");
 sink.warningsEnabled=true;dispatch(snapshot,sink);ck(printed.empty()==snapshot.warnings.empty(),"warning on missing output");
 ck((component?s.rhoOmega[0]:s.rhoK[0])==result&&s.gasSpecies.cellStatus[0]==0,"dispatch changed state");
}
'''.replace('TARGET',target).replace('COMPONENT',str(component)).replace('MOVING','true' if moving_mesh else 'false'))


@pytest.mark.parametrize('component', [0, 1])
def test_recovery_floor_warns_and_nonfinite_stays_fatal(tmp_path,component):
    compile_probe(tmp_path, fixture()+r'''
#include "DiagnosticsHost.H"
int main(){using namespace ugkp::diagnostics;
 for(int bad=0;bad<3;++bad){State s;initialise(s);s.sstConfigured=1;DeviceContext d{};s.diagnostics=&d;
 const int component=COMPONENT;Real& value=component?s.rhoOmega[0]:s.rhoK[0];
 value=bad==0?Real(-2):bad==1?std::numeric_limits<Real>::quiet_NaN():std::numeric_limits<Real>::infinity();
 recoverSstPrimitiveCell(s,0);
 if(bad==0){auto snap=takeSnapshot(d);ck(!snap.hasFatal&&s.gasSpecies.cellStatus[0]==0,"finite clip failed");ck(snap.warnings.size()==1&&snap.warnings[0].sample.sourceSiteId==2624,"recovery warning absent");ck(snap.warnings[0].sample.observed==-2,"recovery original lost");}
 else ck(hasFatal(&d),"nonfinite demoted to warning");
 }
}
'''.replace('COMPONENT',str(component)))


def test_acoustic_cfl_does_not_bound_explicit_lowre_sst_source(tmp_path):
    compile_probe(tmp_path, fixture()+r'''
#include "DiagnosticsHost.H"
int main(){using namespace ugkp::diagnostics;
 State s;initialise(s);DeviceContext d{};s.diagnostics=&d;s.sstConfigured=1;
 // Independent synthetic fine-cell LowRe source witness, not a replay of
 // a user case: two faces of unit area, height 2e-8 m, rho=2, p=1e5.
 const Real height=2e-8,nu=1e-5; s.V[0]=height;s.cellLength[0]=height;
 s.sstWallDistance[0]=height/2;s.gasMu=nu*s.rho[0];
 s.omega[0]=6*nu/(.075*s.sstWallDistance[0]*s.sstWallDistance[0]);
 s.rhoOmega[0]=s.rho[0]*s.omega[0];
 const Real a=std::sqrt(s.gammaGas*s.p[0]/s.rho[0]);
 const Real dt=.7*height/a;
 s.riemannBoundaryKind[1]=2;s.sstWallTreatment=0;
 for(int f=0;f<2;++f){threadIdx.x=f;computeGasCourantFieldKernel(&s,dt);}
 threadIdx.x=0;computeGasConvectiveCourantByCellKernel(&s,dt);
 near(s.gasFluxPositivityScale[0],Real(.7),"production acoustic CFL");
 Real sk,so;ck(sstSourcesForCell(s,0,Real(0),sk,so),"source calculation");
 const Real raw=s.rhoK[0]+dt*sk;ck(raw<0,"source witness no longer unstable");
 applySstFluxAndSourceKernel(&s,dt);
 auto snap=takeSnapshot(d);ck(!snap.hasFatal&&s.gasSpecies.cellStatus[0]==0,"source warning stopped solver");
 ck(!snap.warnings.empty()&&snap.warnings[0].sample.observed==raw,"source clipping silent");
 ck(s.rhoK[0]==2e-12,"production floor changed");
}
''')


def test_constrained_wall_omega_is_not_spuriously_advanced_or_warned(tmp_path):
    compile_probe(tmp_path, fixture()+r'''
#include "DiagnosticsHost.H"
int main(){using namespace ugkp::diagnostics;State s;initialise(s);DeviceContext d{};s.diagnostics=&d;s.sstConfigured=1;
 s.sstWallTreatment=0;s.riemannBoundaryKind[1]=2;s.sstPhiRhoOmega[0]=1e20;
 const Real original=s.rhoOmega[0];applySstFluxAndSourceKernel(&s,Real(.001));
 ck(s.rhoOmega[0]==original,"wall omega RHS changed");ck(takeSnapshot(d).warnings.empty(),"suppressed RHS warned");
}
''')


def test_legacy_gas_existing_floors_warn_without_changing_recovery(tmp_path):
    compile_probe(tmp_path, fixture()+r'''
#include "DiagnosticsCatalogue.H"
int main(){using namespace ugkp::diagnostics;
 for(int component=0;component<2;++component)for(int kind=0;kind<6;++kind){
 State s;initialise(s);DeviceContext d{};s.diagnostics=&d;
 const Real floor=component?Real(1e-6):s.rhoMin;
 if(component)s.TgasMin=floor*(s.gammaGas-1)/(s.rho[0]*s.Rgas);
 const Real value=kind==0?-floor:kind==1?Real(0):kind==2?floor/2:kind==3?floor*2:kind==4?std::numeric_limits<Real>::quiet_NaN():std::numeric_limits<Real>::infinity();
 if(component)s.rhoE[0]=value;else s.rho[0]=value;
 recoverGasPrimitiveCell(s,0);
 auto snapshot=takeSnapshot(d,sourceCatalogue());
 if(kind>=4){ck(snapshot.hasFatal,"nonfinite gas recovery demoted");continue;}
 ck(!snapshot.hasFatal&&s.gasSpecies.cellStatus[0]==0,"finite gas flooring changed status");
 ck((component?s.rhoE[0]:s.rho[0])==std::max(value,floor),"finite gas recovery changed value");
 if(kind<3){ck(snapshot.warnings.size()==1,"missing gas floor warning");const auto&r=snapshot.warnings[0].sample;
 ck(r.sourceSiteId==unsigned(component?2626:2625)&&r.quantityId==unsigned(component?Quantity::InternalEnergy:Quantity::Density),"gas warning provenance");
 ck(r.observed==value,"gas original lost");near(r.reference,floor,"gas floor lost");near(r.corrected,floor,"gas correction lost");}
 else ck(snapshot.warnings.empty(),"healthy gas inventory warned");
 std::string printed;HostSink sink;sink.user=&printed;
 sink.warning=[](void*p,const WarningSummary&w,const SourceDescriptor*source){*static_cast<std::string*>(p)+=formatRecord(w.sample,source);};
 sink.warningsEnabled=false;dispatch(snapshot,sink);ck(printed.empty(),"disabled gas warning printed");
 sink.warningsEnabled=true;dispatch(snapshot,sink);ck(printed.empty()==snapshot.warnings.empty(),"gas warning missing");
 }
}
''')
