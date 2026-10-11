"""Actual production operators; host CUDA shim, not GPU execution evidence."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


def audited_fixture():
    return fixture().replace('ugkwp::GasSpeciesState<Real,2> gasSpecies;',
        'ugkwp::GasSpeciesState<Real,2> gasSpecies;ugkwp::GasBudgetAuditState gasBudgetAudit;')+r'''
void enable(State&s){s.gasBudgetAudit.enabled=true;s.gasBudgetAudit.speciesCount=2;
 s.gasBudgetAudit.transport=new double[8]{};s.gasBudgetAudit.initialTransport=new double[8]{};
 s.gasBudgetAudit.chemistrySpecies=new double[4]{};}
void stage(State&s,double dt,double rate){
 for(int f=0;f<3;++f){s.gasPhiRho[f]=0;s.gasPhiRhoE[f]=0;s.gasSpecies.flux[f]=0;s.gasSpecies.flux[3+f]=0;}
 s.gasPhiRho[1]=-rate;s.gasPhiRhoE[1]=-100*rate;
 s.gasSpecies.flux[1]=Real(-.4)*rate;s.gasSpecies.flux[4]=Real(-.6)*rate;
 for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,dt);}
}
'''


@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('rk',[1,2,3])
def test_actual_rk_weights_propagate_final_boundary_flux_budget(tmp_path,bits,rk):
    compile_probe(tmp_path,audited_fixture()+r'''
int main(){State s;initialise(s);enable(s);const double dt=.001;
if(RK!=1)for(int c=0;c<2;++c){threadIdx.x=c;saveGasConservativeStateKernel(&s);}
stage(s,dt,1);
if(RK!=1){stage(s,dt,2);for(int c=0;c<2;++c){threadIdx.x=c;blendGasConservativeStateKernel(&s,Real(RK==2?.5:.75),Real(RK==2?.5:.25));}}
if(RK==3){stage(s,dt,4);for(int c=0;c<2;++c){threadIdx.x=c;blendGasConservativeStateKernel(&s,Real(1)/Real(3),Real(2)/Real(3));}}
const double expected=RK==1?dt:RK==2?1.5*dt:double(Real(2)/Real(3))*(.25*3+4)*dt;
ck(std::abs(s.gasBudgetAudit.transport[0]-expected)<1e-12,"actual RK flux weight missing");
ck(std::abs(s.gasBudgetAudit.transport[2]-100*expected)<1e-9,"physical energy budget missing");
ck(std::abs(s.gasBudgetAudit.transport[4]-.4*expected)<1e-9,"species boundary budget missing");
ck(std::abs(s.gasBudgetAudit.transport[6]-.6*expected)<1e-9,"species second component missing");
ck(s.gasBudgetAudit.transport[1]==0,"unforced cell gained external flux budget");}
'''.replace('RK',str(rk)),bits)


def test_budget_reads_limited_flux_and_excludes_internal_periodic_faces(tmp_path):
    compile_probe(tmp_path,audited_fixture()+r'''
int main(){State s;initialise(s);enable(s);s.gasSpecies.rho[0]=.001;s.gasSpecies.rho[2]=.999;
s.gasSpecies.flux[0]=2;s.gasSpecies.flux[3]=-2;s.gasPhiRho[0]=0;s.gasPhiRhoE[0]=123;
for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,.1);}
threadIdx.x=0;applyGasFluxPositivityScaleKernel(&s);
for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,.1);}
for(int i=0;i<8;++i)ck(s.gasBudgetAudit.transport[i]==0,"internal flux counted as external");
s.facePeriodicPair[1]=2;s.facePeriodicPair[2]=1;stage(s,.00001,1);
for(int i=0;i<8;++i)ck(s.gasBudgetAudit.transport[i]==0,"periodic interface counted as external");}
''')


def test_budget_uses_limited_external_counterflow_without_bulk_mass_source(tmp_path):
    compile_probe(tmp_path,audited_fixture()+r'''
int main(){State s;initialise(s);enable(s);const double dt=.1;
s.gasSpecies.rho[0]=.001;s.gasSpecies.rho[2]=.999;
s.gasSpecies.flux[1]=2;s.gasSpecies.flux[4]=-2;s.gasPhiRho[1]=0;s.gasPhiRhoE[1]=123;
for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,dt);}
threadIdx.x=1;applyGasFluxPositivityScaleKernel(&s);
const double limitedSpecies=s.gasSpecies.flux[1],limitedEnergy=s.gasPhiRhoE[1];
ck(limitedSpecies>0&&limitedSpecies<2,"external limiter was not active");
threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,dt);
ck(s.gasSpecies.cellStatus[0]==0,"legal external counterflow failed");
ck(s.gasBudgetAudit.transport[0]==0,"counterflow invented bulk mass");
ck(s.gasBudgetAudit.transport[2]==-dt*limitedEnergy,"energy audit read unscaled flux");
ck(s.gasBudgetAudit.transport[4]==-dt*limitedSpecies,"species audit read unscaled flux");
ck(s.gasBudgetAudit.transport[6]==dt*limitedSpecies,"counterflow species audit missing");}
''')


@pytest.mark.parametrize('bits',[32,64])
def test_budget_records_final_physical_wall_override(tmp_path,bits):
    from test_boundary_layer_operators import SETUP
    compile_probe(tmp_path,audited_fixture()+SETUP+r'''
int main(){State s;initialise(s);enable(s);wall(s);const double dt=.0001;
s.gasBoundaryLayer.speciesFlux[0]=Real(-.16);s.gasBoundaryLayer.speciesFlux[1]=Real(-.24);
threadIdx.x=1;computeGasInternalFaceFluxKernel<false>(&s,dt);applyGasFluxPositivityScaleKernel(&s);
ck(s.gasSpecies.faceStatus[1]==0,"physical wall flux failed");
const double finalMass=s.gasPhiRho[1],finalEnergy=s.gasPhiRhoE[1],finalSpecies=s.gasSpecies.flux[1];
ck(std::abs(finalMass+.4)<1e-7&&std::abs(finalEnergy+31)<1e-7,"physical wall override missing");
threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,dt);
ck(s.gasSpecies.cellStatus[0]==0,"physical wall update failed");
ck(s.gasBudgetAudit.transport[0]==-dt*finalMass,"wall mass budget read pre-override flux");
ck(s.gasBudgetAudit.transport[2]==-dt*finalEnergy,"wall energy budget read pre-override flux");
ck(s.gasBudgetAudit.transport[4]==-dt*finalSpecies,"wall species budget read pre-override flux");}
''',bits)
