"""Test-local conservation checks of actual operators, without runtime ledgers."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


def physical_fixture():
    return fixture()+r'''
void stage(State&s,double dt,double rate){
 for(int f=0;f<3;++f){s.gasPhiRho[f]=0;s.gasPhiRhoE[f]=0;s.gasSpecies.flux[f]=0;s.gasSpecies.flux[3+f]=0;}
 s.gasPhiRho[1]=-rate;s.gasPhiRhoE[1]=-100*rate;
 s.gasSpecies.flux[1]=Real(-.4)*rate;s.gasSpecies.flux[4]=Real(-.6)*rate;
 for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,dt);}
}
'''


@pytest.mark.parametrize('rk',[1,2,3])
def test_actual_rk_weights_propagate_final_boundary_flux_budget(tmp_path,rk):
    compile_probe(tmp_path,physical_fixture()+r'''
int main(){State s;initialise(s);const double dt=.001;
const Real mass=s.rho[0],energy=s.rhoE[0],a=s.gasSpecies.rho[0],b=s.gasSpecies.rho[2];
const Real other=s.rho[1],otherE=s.rhoE[1];
if(RK!=1)for(int c=0;c<2;++c){threadIdx.x=c;saveGasConservativeStateKernel(&s);}
stage(s,dt,1);
if(RK!=1){stage(s,dt,2);for(int c=0;c<2;++c){threadIdx.x=c;blendGasConservativeStateKernel(&s,Real(RK==2?.5:.75),Real(RK==2?.5:.25));}}
if(RK==3){stage(s,dt,4);for(int c=0;c<2;++c){threadIdx.x=c;blendGasConservativeStateKernel(&s,Real(1)/Real(3),Real(2)/Real(3));}}
const double expected=RK==1?dt:RK==2?1.5*dt:double(Real(2)/Real(3))*(.25*3+4)*dt;
const double eps=64*std::numeric_limits<Real>::epsilon();
ck(std::abs(double(s.rho[0]-mass)-expected)<eps,"actual mass RK flux weight missing");
ck(std::abs(double(s.rhoE[0]-energy)-100*expected)<eps*std::abs(energy),"actual energy boundary flux missing");
ck(std::abs(double(s.gasSpecies.rho[0]-a)-.4*expected)<eps,"species first component missing");
ck(std::abs(double(s.gasSpecies.rho[2]-b)-.6*expected)<eps,"species second component missing");
ck(std::abs(s.rho[1]-other)<eps&&std::abs(s.rhoE[1]-otherE)<eps*std::abs(otherE),"unforced cell changed");}
'''.replace('RK',str(rk)))


def test_mixed_energy_budget_uses_actual_energy_stage_weights(tmp_path):
    source=physical_fixture().replace('GasStateView<Real,Time,ugkwp::SstCoefficients>',
        'GasStateView<Real,Time,ugkwp::SstCoefficients,double>').replace('s.rhoE=new Real[8]{};','s.rhoE=new double[8]{};').replace('s.rhoENext=new Real[8]{};','s.rhoENext=new double[8]{};')
    compile_probe(tmp_path,source+r'''
int main(){State s;initialise(s);threadIdx.x=0;saveGasConservativeStateKernel(&s);
s.rhoENext[0]=1000.;s.rhoE[0]=1030.;blendGasConservativeStateKernel(&s,1./3.,2./3.);
ck(s.rhoE[0]==1020.,"physical mixed energy stage mismatch");}
''')


def test_limited_external_counterflow_changes_actual_inventory(tmp_path):
    compile_probe(tmp_path,physical_fixture()+r'''
int main(){State s;initialise(s);const double dt=.1;
s.gasSpecies.rho[0]=.001;s.gasSpecies.rho[2]=.999;
const Real a=s.gasSpecies.rho[0],b=s.gasSpecies.rho[2],energy=s.rhoE[0],mass=s.rho[0];
s.gasSpecies.flux[1]=2;s.gasSpecies.flux[4]=-2;s.gasPhiRho[1]=0;s.gasPhiRhoE[1]=123;
for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,dt);}
threadIdx.x=1;applyGasFluxPositivityScaleKernel(&s);
const Real limited=s.gasSpecies.flux[1],limitedE=s.gasPhiRhoE[1];
ck(limited>0&&limited<2,"external limiter inactive");threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,dt);
const double eps=64*std::numeric_limits<Real>::epsilon();
ck(s.rho[0]==mass,"counterflow invented bulk mass");
ck(std::abs(double(s.gasSpecies.rho[0]-a)+dt*limited)<eps,"limited outgoing species missing");
ck(std::abs(double(s.gasSpecies.rho[2]-b)-dt*limited)<eps,"limited incoming species missing");
ck(std::abs(double(s.rhoE[0]-energy)+dt*limitedE)<eps*std::abs(energy),"limited physical energy missing");}
''')


def test_final_physical_wall_override_updates_actual_inventory(tmp_path):
    from test_boundary_layer_operators import SETUP
    compile_probe(tmp_path,physical_fixture()+SETUP+r'''
int main(){State s;initialise(s);wall(s);const double dt=.0001;
const Real mass=s.rho[0],energy=s.rhoE[0],species=s.gasSpecies.rho[0];
s.gasBoundaryLayer.speciesFlux[0]=Real(-.16);s.gasBoundaryLayer.speciesFlux[1]=Real(-.24);
threadIdx.x=1;computeGasInternalFaceFluxKernel<false>(&s,dt);applyGasFluxPositivityScaleKernel(&s);
const Real fm=s.gasPhiRho[1],fe=s.gasPhiRhoE[1],fs=s.gasSpecies.flux[1];
ck(std::abs(fm+.4)<1e-7&&std::abs(fe+31)<1e-7,"physical wall override missing");
threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&s,dt);const double eps=64*std::numeric_limits<Real>::epsilon();
ck(s.gasSpecies.cellStatus[0]==0,"physical wall update failed");
ck(std::abs(double(s.rho[0]-mass)+dt*fm)<eps,"wall mass update missing");
ck(std::abs(double(s.rhoE[0]-energy)+dt*fe)<eps*std::abs(energy),"wall energy update missing");
ck(std::abs(double(s.gasSpecies.rho[0]-species)+dt*fs)<eps,"wall species update missing");}
''')


def test_internal_flux_preserves_global_inventory(tmp_path):
    compile_probe(tmp_path,physical_fixture()+r'''
int main(){State s;initialise(s);s.gasSpecies.rho[0]=.001;s.gasSpecies.rho[2]=.999;
const Real a=s.gasSpecies.rho[0]+s.gasSpecies.rho[1],b=s.gasSpecies.rho[2]+s.gasSpecies.rho[3],e=s.rhoE[0]+s.rhoE[1];
s.gasSpecies.flux[0]=2;s.gasSpecies.flux[3]=-2;s.gasPhiRho[0]=0;s.gasPhiRhoE[0]=123;
for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,.1);}threadIdx.x=0;applyGasFluxPositivityScaleKernel(&s);
for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,.1);}
const double eps=64*std::numeric_limits<Real>::epsilon();
ck(std::abs(s.gasSpecies.rho[0]+s.gasSpecies.rho[1]-a)<eps,"internal first species not conserved");
ck(std::abs(s.gasSpecies.rho[2]+s.gasSpecies.rho[3]-b)<eps,"internal second species not conserved");
ck(std::abs(s.rhoE[0]+s.rhoE[1]-e)<eps*std::abs(e),"internal energy not conserved");}
''')
