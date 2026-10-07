"""Shared stability bounds use actual boundary calorics and turbulent diffusion."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

@pytest.mark.parametrize('bits',[32,64])
def test_courant_uses_hot_fixed_boundary_mixture_sound_speed(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.riemannBoundaryKind[2]=0;s.riemannBoundaryPFix[2]=1;s.riemannBoundaryTFix[2]=1;s.riemannBoundaryP[2]=s.p[1];s.riemannBoundaryT[2]=1400;
s.gasSpecies.compositionBoundaryFixed[2]=1;s.gasSpecies.boundaryMassFraction[2]=.4;s.gasSpecies.boundaryMassFraction[5]=.6;
const Real y[2]={Real(.4),Real(.6)};const Real R=ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo),cv=ugkwp::mixtureHeatCapacity(y,Real(1400),s.gasSpecies.thermo);const Real expected=sqrt((cv+R)/cv*R*Real(1400));
threadIdx.x=2;computeGasCourantFieldKernel(&s,1e-4);ck(s.gasSpecies.faceStatus[2]==0,"valid hot boundary rejected");ck(std::abs(s.gasPhiRho[2]-expected)<Real(1e-5)*expected,"boundary Courant reused cold owner sound speed");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_mixture_diffusion_bound_keeps_eddy_heat_and_species_terms(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.turbulenceModel=1;s.turbulentPrandtl=.7;s.gasMu=.003;s.gasCp=7;s.gasSpecies.turbulentSchmidt=.5;s.nut[0]=s.nut[1]=.02;s.maxDiffusionNumber=.25;
// One internal face isolates the conservative coefficient against an analytic bound.
s.cellPlaneCount[0]=1;s.cellPlaneStart[0]=0;Real D[2]={.01,.03};s.gasSpecies.diffusivity=D;
Real mu=0,kappa=0,q=0;int direct=0;gasFaceSubgridTransportProperties(s,0,0,1,0,Real(1),mu,kappa,q,direct);
const Real cp=Real(.5)*(s.gasSpecies.heatCapacity[0]+s.gasSpecies.heatCapacity[1]);const Real R=Real(.5)*(s.gasSpecies.gasConstant[0]+s.gasSpecies.gasConstant[1]);
const Real thermal=(s.gasMu*cp/s.gasPrClamped+kappa)/(cp-R);const Real species=Real(2)*(.03+mu/s.gasSpecies.turbulentSchmidt);const Real nu=s.gasMu+mu;const Real expected=Real(.5)*Real(.01)*fmax(fmax(nu,thermal),species)/s.maxDiffusionNumber;
threadIdx.x=0;computeGasDiffusionNumberKernel(&s,.01,.5);ck(std::abs(s.gasDiffusionNumber[0]-expected)<Real(1e-5)*expected,"mixture turbulent diffusion bound lost conductivity or eddy species diffusion");
s.gasSpecies.diffusivity=nullptr;s.gasSpecies.turbulentSchmidt=1000;const Real expectedHeat=Real(.5)*Real(.01)*fmax(nu,thermal)/s.maxDiffusionNumber;computeGasDiffusionNumberKernel(&s,.01,.5);ck(std::abs(s.gasDiffusionNumber[0]-expectedHeat)<Real(1e-5)*expectedHeat,"mixture molecular cp overwrite removed eddy heat bound");}
''',bits)
