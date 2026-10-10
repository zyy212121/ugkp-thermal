"""Mixture closure extension in shared viscosity/SST/transport operators."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

@pytest.mark.parametrize('bits',[32,64])
def test_mixture_eddy_thermal_and_species_transport(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State a,b;initialise(a);initialise(b);b.turbulenceModel=1;b.turbulentPrandtl=.7;b.gasSpecies.turbulentSchmidt=.5;
b.nut[0]=b.nut[1]=.02;b.gasCp=7; // scalar bootstrap Cp must have no mixture effect
for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&a);computeGasPrimitiveGradientsKernel(&b);}
Real mu=0,kappa=0,q=0;int direct=0;gasFaceSubgridTransportProperties(b,0,0,1,0,Real(1),mu,kappa,q,direct);
ck(std::abs(mu-Real(.02))<Real(1e-7),"eddy viscosity absent");ck(std::abs(kappa-Real(1000)*mu/Real(.7))<Real(1e-4),"mixture eddy conductivity uses scalar Cp");
threadIdx.x=0;computeGasInternalFaceFluxKernel<false>(&a,1e-6);computeGasInternalFaceFluxKernel<true>(&b,1e-6);
ck(b.gasSpecies.faceStatus[0]==0,"mixture eddy transport rejected");Real j0=b.gasSpecies.flux[0]-a.gasSpecies.flux[0];
ck(std::abs(j0-Real(.032))<Real(3e-5),"turbulent species flux absent without molecular D");
ck(std::abs((b.gasPhiRhoE[0]-a.gasPhiRhoE[0])-Real(-96000))<Real(100),"turbulent formation enthalpy omitted");
ck(b.gasSpecies.flux[0]+b.gasSpecies.flux[3]==b.gasPhiRho[0],"turbulent total mass closure");
b.turbulenceModel=3;b.sstWallTreatment=0;mu=kappa=99;gasFaceSubgridTransportProperties(b,1,0,-1,2,Real(1),mu,kappa,q,direct);ck(mu==0&&kappa==0,"low-Re wall turbulent flux nonzero");
}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_low_re_sst_runs_the_actual_mixture_transport_chain(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.turbulenceModel=3;s.sstConfigured=1;s.sstWallTreatment=0;s.sstCoefficients=ugkwp::defaultSstCoefficients();s.gasMu=.001;s.turbulentPrandtl=.9;s.sstKMin=1e-12;s.sstOmegaMin=1e-8;s.sstMaxSourceNumber=.25;s.sstWallKappa=.41;s.sstWallE=9.8;s.sstWallCmu=.09;
for(int c=0;c<2;++c){s.k[c]=.1+.02*c;s.omega[c]=10;s.sstWallDistance[c]=.5;threadIdx.x=c;initialiseSstConservativeStateKernel(&s);recoverSstPrimitivesKernel(&s);}
const Real initialE=s.rhoE[0]+s.rhoE[1];
for(int step=0;step<20;++step){
for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);computeSstGradientsKernel(&s);computeGasGradientLimiterKernel(&s);computeGasEddyViscosityKernel(&s);ck(s.nut[c]>0,"SST eddy viscosity absent");}
for(int f=0;f<3;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<true>(&s,1e-7);ck(s.gasSpecies.faceStatus[f]==0,"SST mixture face rejected");}
for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,1e-7);}
for(int f=0;f<3;++f){threadIdx.x=f;applyGasFluxPositivityScaleKernel(&s);computeSstFaceFluxKernel(&s);}
for(int c=0;c<2;++c){threadIdx.x=c;applySstFluxAndSourceKernel(&s,1e-7);applyGasFluxDivergenceByCellKernel(&s,1e-7);recoverGasPrimitivesKernel(&s);recoverSstPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0&&s.k[c]>0&&s.omega[c]>0,"SST coupled update invalid");ck(std::abs(s.Tgas[c]-700)<Real(.05),"SST formation contact heated");}}
ck(std::abs(s.rhoE[0]+s.rhoE[1]-initialE)<Real(3e-6)*std::abs(initialE),"SST gas energy not conservative");
}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_laminar_inviscid_wall_skips_turbulent_wall_algebra(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
#include <cfenv>
int main(){State s;initialise(s);s.turbulenceModel=0;s.gasMu=0;s.Ux[0]=1;
Real mu=9,k=8,q=7;int direct=6;
feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW);
gasFaceSubgridTransportProperties(s,1,0,-1,2,Real(1),mu,k,q,direct);
ck(mu==0&&k==0&&q==0&&direct==0,"laminar wall gained turbulent transport");}
''',bits)
