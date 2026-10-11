"""Exercise mixture extensions in the original gas operator chain."""
from pathlib import Path
import re
import pytest
from test_mixture_state import compile_probe

HERE=Path(__file__).resolve().parent

def fixture():
    fields=re.findall(r'    ([\w:]+\*) (\w+)\{\};',(HERE/'legacy_gas_fixture.hpp').read_text())
    allocations='\n'.join(f's.{name}=new {ty[:-1]}[8]{{}};' for ty,name in fields)
    return r'''
struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2> gasSpecies;};
void ck(bool good,const char*message){if(!good){std::fprintf(stderr,"%s\n",message);std::abort();}}
void initialise(State&s){
ALLOCATE
s.nCells=2;s.nFaces=3;s.nInternalFaces=1;s.gasFluxScheme=1;s.gasReconstruction=0;s.gasLimiter=0;s.gasMu=0;s.gasPrClamped=.7;s.gammaGas=1.4;s.Rgas=287;s.gasCp=1004.5;s.rhoMin=1e-12;s.TgasMin=1e-9;
s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;
auto&g=s.gasSpecies;g.rho=new Real[4]{};g.initial=new Real[4]{};g.flux=new Real[6]{};g.gradX=new Real[4]{};g.gradY=new Real[4]{};g.gradZ=new Real[4]{};g.limiter=new Real[2]{};g.boundaryMassFraction=new Real[6]{};g.compositionBoundaryFixed=new int[3]{};g.positivityScale=new Real[4]{};g.soundSpeed=new Real[2]{};g.heatCapacity=new Real[2]{};g.gasConstant=new Real[2]{};g.cellStatus=new int[2]{};g.faceStatus=new int[3]{};
auto*meta=new ugkwp::SpeciesThermoData<Real>[2];auto*coeff=new Real[6]{1000,0,-1e6,1000,0,2e6};g.thermo.species=meta;g.thermo.coefficients=coeff;g.thermo.coefficientCount=6;g.thermoControls.relativeEnergyTolerance=Real(2e-6);g.thermoControls.relativeTemperatureTolerance=Real(2e-6);
for(int k=0;k<2;++k){meta[k].molarMass=.028;meta[k].minTemperature=100;meta[k].maxTemperature=3000;meta[k].referencePressure=101325;meta[k].coefficientOffset=3*k;}
g.rho[0]=.9;g.rho[1]=.1;g.rho[2]=.1;g.rho[3]=.9;
for(int c=0;c<2;++c){s.V[c]=1;s.Cx[c]=c;s.rho[c]=1;Real rhos[2]={g.rho[c],g.rho[2+c]};s.rhoE[c]=ugkwp::mixtureEnergy(rhos,Real(700),g.thermo);s.Tgas[c]=700;s.cellPlaneStart[c]=2*c;s.cellPlaneCount[c]=2;}
s.cellFaceId[0]=0;s.cellFaceId[1]=1;s.cellFaceId[2]=0;s.cellFaceId[3]=2;
for(int f=0;f<3;++f){s.faceOwner[f]=f==2?1:0;s.faceNeighbour[f]=f==0?1:-1;s.facePeriodicPair[f]=-1;s.magSf[f]=1;s.Sfx[f]=f==1?-1:1;s.faceCx[f]=f==0?.5:f==1?-.5:1.5;s.faceWeight[f]=.5;s.deltaCoeffs[f]=f==0?1:2;s.gasBoundaryKind[f]=s.riemannBoundaryKind[f]=f==0?0:1;}
blockDim.x=1;blockIdx.x=0;
for(int c=0;c<2;++c){threadIdx.x=c;recoverGasPrimitivesKernel(&s);}
}
void transport(State&s,double dt){
for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);computeGasGradientLimiterKernel(&s);computeGasEddyViscosityKernel(&s);}
for(int f=0;f<3;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<false>(&s,dt);}
for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,dt);}
for(int f=0;f<3;++f){threadIdx.x=f;applyGasFluxPositivityScaleKernel(&s);}
for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,dt);recoverGasPrimitivesKernel(&s);}
}
'''.replace('ALLOCATE',allocations)

@pytest.mark.parametrize('bits',[32,64])
def test_actual_chain_preserves_formation_contact_temperature(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);Real beforeE=s.rhoE[0]+s.rhoE[1];transport(s,1e-5);
for(int c=0;c<2;++c){ck(s.gasSpecies.cellStatus[c]==0,"trial status");ck(std::abs(s.Tgas[c]-700)<Real(.03),"formation diffusion heated stationary contact");}
ck(s.gasSpecies.rho[0]<Real(.9)&&s.gasSpecies.rho[1]>Real(.1),"species numerical dissipation missing");
ck(std::abs(s.rhoE[0]+s.rhoE[1]-beforeE)<Real(1e-6)*std::abs(beforeE),"total energy changed");
for(int k=0;k<2;++k)ck(std::abs(s.gasSpecies.rho[2*k]+s.gasSpecies.rho[2*k+1]-1)<Real(2e-6),"species not conserved");
ck(std::abs(s.gasSpecies.flux[0]+s.gasSpecies.flux[3]-s.gasPhiRho[0])<Real(2e-6),"mass flux closure");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_zero_mass_flux_species_donors_scale_all_components(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.gasSpecies.rho[0]=Real(.001);s.gasSpecies.rho[2]=Real(.999);
s.gasSpecies.flux[0]=2;s.gasSpecies.flux[3]=-2;s.gasPhiRho[0]=0;s.gasPhiRhoE[0]=123;
for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,.1);}
threadIdx.x=0;applyGasFluxPositivityScaleKernel(&s);
ck(s.gasSpecies.flux[0]>0&&s.gasSpecies.flux[0]<Real(.01),"zero-mass diffusion donor ignored");
ck(s.gasSpecies.flux[3]==-s.gasSpecies.flux[0],"scale broke species closure");
ck(std::abs(s.gasPhiRhoE[0]/123-s.gasSpecies.flux[0]/2)<Real(1e-7),"energy and species scaled differently");
for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,.1);}
ck(s.gasSpecies.rho[0]>=0,"negative depleted species");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_species_runge_kutta_and_periodic_use_existing_kernels(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);for(int c=0;c<2;++c){threadIdx.x=c;saveGasConservativeStateKernel(&s);}
s.gasSpecies.rho[0]=.5;s.gasSpecies.rho[2]=.5;threadIdx.x=0;blendGasConservativeStateKernel(&s,Real(.75),Real(.25));
ck(std::abs(s.gasSpecies.rho[0]-Real(.8))<Real(1e-7),"species RK missing");
s.nFaces=2;s.nInternalFaces=0;s.facePeriodicPair[0]=1;s.facePeriodicPair[1]=0;s.gasSpecies.flux[0]=2;s.gasSpecies.flux[1]=-4;s.gasSpecies.flux[2]=-2;s.gasSpecies.flux[3]=4;
threadIdx.x=0;enforcePeriodicGasFluxAntisymmetryKernel(&s);ck(s.gasSpecies.flux[0]==3&&s.gasSpecies.flux[1]==-3&&s.gasSpecies.flux[2]==-3&&s.gasSpecies.flux[3]==3,"periodic species antisymmetry missing");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_species_diffusion_is_zero_sum_and_carries_formation_enthalpy(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
#include "gasTransport/SpeciesDiffusion.H"
int main(){State s;initialise(s);Real Y[2]={.25,.75},grad[2]={1,-1},D[2]={.01,.03},flux[2]={99,98},energy=97;
ck(ugkwp::correctedSpeciesDiffusion(Y,grad,Real(2),D,Real(.004),Real(.8),Real(700),s.gasSpecies.thermo,flux,energy),"valid corrected diffusion rejected");
Real raw0=-2*(D[0]+Real(.005)),raw1=2*(D[1]+Real(.005));
ck(std::abs(flux[0]-(raw0-Y[0]*(raw0+raw1)))<Real(1e-7),"unequal D correction");ck(flux[0]+flux[1]==0,"diffusion total mass nonzero");
Real expected=ugkwp::speciesH(0,Real(700),s.gasSpecies.thermo)*flux[0]+ugkwp::speciesH(1,Real(700),s.gasSpecies.thermo)*flux[1];ck(std::abs(energy-expected)<Real(1e-6)*std::abs(expected),"enthalpy diffusion omitted formation energy");
Real prior[]={flux[0],flux[1],energy};D[0]=-1;ck(!ugkwp::correctedSpeciesDiffusion(Y,grad,Real(2),D,Real(0),Real(.8),Real(700),s.gasSpecies.thermo,flux,energy),"invalid diffusion accepted");ck(flux[0]==prior[0]&&flux[1]==prior[1]&&energy==prior[2],"failed diffusion published");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_actual_face_adds_unequal_diffusion_and_periodic_gradients(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State a,b;initialise(a);initialise(b);Real D[2]={.01,.03};b.gasSpecies.diffusivity=D;
for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&a);computeGasPrimitiveGradientsKernel(&b);}
threadIdx.x=0;computeGasInternalFaceFluxKernel<false>(&a,1e-6);computeGasInternalFaceFluxKernel<false>(&b,1e-6);
Real j0=b.gasSpecies.flux[0]-a.gasSpecies.flux[0];Real j1=b.gasSpecies.flux[3]-a.gasSpecies.flux[3];
ck(std::abs(j0-Real(.016))<Real(5e-5),"actual corrected Fick flux missing");ck(std::abs(j0+j1)<Real(5e-5),"diffusion changes total species mass");
ck(a.gasPhiRho[0]==b.gasPhiRho[0],"bulk mass gained diffusion");Real h0=ugkwp::speciesH(0,Real(700),b.gasSpecies.thermo),h1=ugkwp::speciesH(1,Real(700),b.gasSpecies.thermo);Real expected=(h0-h1)*Real(.016);
ck(std::abs((b.gasPhiRhoE[0]-a.gasPhiRhoE[0])-expected)<Real(2e-3)*std::abs(expected),"actual energy diffusion missing");
ck(b.gasSpecies.gradX[0]!=0&&b.gasSpecies.gradX[0]+b.gasSpecies.gradX[2]==0,"common species Green-Gauss gradients missing");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_boundary_uses_composition_eos_and_rejects_redundant_inputs(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);auto*metadata=const_cast<ugkwp::SpeciesThermoData<Real>*>(s.gasSpecies.thermo.species);metadata[1].molarMass=.044;
s.riemannBoundaryKind[1]=0;s.riemannBoundaryPFix[1]=1;s.riemannBoundaryTFix[1]=1;s.riemannBoundaryP[1]=101325;s.riemannBoundaryT[1]=500;s.gasSpecies.compositionBoundaryFixed[1]=1;s.gasSpecies.boundaryMassFraction[1]=.25;s.gasSpecies.boundaryMassFraction[4]=.75;
GasPrimDevice owner{s.rho[0],0,0,0,s.p[0],s.Tgas[0]};auto b=riemannBoundaryState(s,1,owner);Real Y[2]={.25,.75};Real expectedR=ugkwp::mixtureGasConstant(Y,s.gasSpecies.thermo);
ck(s.gasSpecies.faceStatus[1]==0&&std::abs(b.rho-101325/(expectedR*500))<Real(1e-6),"boundary used scalar R");
s.Rgas=1;s.gammaGas=9;auto again=riemannBoundaryState(s,1,owner);ck(again.rho==b.rho&&again.T==b.T,"bootstrap EOS leaked into boundary");
s.riemannBoundaryRhoFix[1]=1;s.riemannBoundaryRho[1]=100;riemannBoundaryState(s,1,owner);ck(s.gasSpecies.faceStatus[1]!=0,"inconsistent redundant rho/p/T accepted");}
''',bits)

def test_explicit_conductivity_is_independent_of_caloric_cp(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.gasThermalConductivity=.42;s.gasMu=0;
ck(molecularGasConductivity(s)==Real(.42),"explicit conductivity ignored");s.gasCp=99999;
ck(molecularGasConductivity(s)==Real(.42),"direct conductivity changed with cp");s.gasThermalConductivity=-1;s.gasMu=.01;s.gasCp=1000;s.gasPrClamped=.5;
ck(molecularGasConductivity(s)==20,"legacy conductivity fallback changed");}
''')

@pytest.mark.parametrize('bits',[32,64])
def test_fixed_composition_boundary_has_corrected_diffusion(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State a,b;initialise(a);initialise(b);Real D[2]={.01,.03};b.gasSpecies.diffusivity=D;
for(auto*s:{&a,&b}){s->riemannBoundaryKind[1]=0;s->riemannBoundaryPFix[1]=1;s->riemannBoundaryTFix[1]=1;s->riemannBoundaryP[1]=s->p[0];s->riemannBoundaryT[1]=700;s->gasSpecies.compositionBoundaryFixed[1]=1;s->gasSpecies.boundaryMassFraction[1]=.25;s->gasSpecies.boundaryMassFraction[4]=.75;}
for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&a);computeGasPrimitiveGradientsKernel(&b);}threadIdx.x=1;
computeGasInternalFaceFluxKernel<false>(&a,1e-6);computeGasInternalFaceFluxKernel<false>(&b,1e-6);
ck(b.gasSpecies.faceStatus[1]==0,"boundary rejected");Real j=b.gasSpecies.flux[1]-a.gasSpecies.flux[1];ck(std::abs(j-Real(.0195))<Real(2e-5),"fixed boundary species diffusion omitted");
ck(std::abs((b.gasPhiRhoE[1]-a.gasPhiRhoE[1])-Real(-58500))<Real(100),"fixed boundary enthalpy diffusion omitted");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_hll_shared_scheme_uses_matching_species_and_energy_waves(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.gasFluxScheme=2;transport(s,1e-5);
for(int c=0;c<2;++c){ck(s.gasSpecies.cellStatus[c]==0,"HLL mixture rejected");ck(std::abs(s.Tgas[c]-700)<Real(.03),"HLL formation-contact heating");}
ck(s.gasSpecies.rho[0]<Real(.9),"HLL species waves absent");ck(s.gasSpecies.flux[0]+s.gasSpecies.flux[3]==s.gasPhiRho[0],"HLL mass closure");
using namespace ugkpriemann;Primitive l{1,2000,0,0,100000},r{.8,2100,0,0,90000};
auto ul=conservative(l,Real(1.4)),ur=conservative(r,Real(1.4));Real al=std::sqrt(Real(1.4)*l.p/l.rho),ar=std::sqrt(Real(1.4)*r.p/r.rho);
auto old=hllKurganovFluxUnitNormal(l,r,Real(1),Real(0),Real(0),Real(1.4),false);
auto now=hllKurganovFluxUnitNormal(l,ul,al,r,ur,ar,Real(1),Real(0),Real(0),false);
for(int k=0;k<5;++k)ck(old.flux[k]==now.flux[k],"HLL legacy reduction");
Real species=hllConservativeComponentFlux(Real(.3)*l.ux,Real(.2)*r.ux,Real(.3),Real(.2),l.ux,r.ux,al,ar);
ck(std::abs(species-Real(.3)*l.ux)<Real(1e-4),"HLL supersonic scalar upwind");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_euler_ale_uniform_state_satisfies_gcl_and_conservation(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);for(int c=0;c<2;++c){s.gasSpecies.rho[c]=.4;s.gasSpecies.rho[2+c]=.6;Real partial[2]={.4,.6};s.rhoE[c]=ugkwp::mixtureEnergy(partial,Real(700),s.gasSpecies.thermo);threadIdx.x=c;recoverGasPrimitivesKernel(&s);}
// Two moving transmissive endpoints with a uniform state: outward sweeps sum
// to .1 in each cell; relative advective flux exactly balances this dilation.
s.riemannBoundaryKind[1]=s.riemannBoundaryKind[2]=0;
Real oldV[2]={1,1},newV[2]={1.1,1.1},sweep[3]={0,.1,.1};
s.gasGeometry.enabled=true;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.001;s.gasGeometry.absoluteGeometryTolerance=Real(1e-7);s.gasGeometry.relativeGeometryTolerance=Real(1e-6);
Real oldE=s.rhoE[0];transport(s,.001);
for(int c=0;c<2;++c){ck(s.gasSpecies.cellStatus[c]==0,"ALE rejected valid geometry");ck(std::abs(s.rho[c]-1)<Real(2e-6),"ALE uniform density GCL");ck(std::abs(s.Tgas[c]-700)<Real(.03),"ALE uniform thermal GCL");ck(std::abs(s.rhoE[c]-oldE)<Real(2e-6)*std::abs(oldE),"ALE uniform energy GCL");}
ck(s.V[0]==1,"operator unexpectedly published endpoint geometry");
State bad;initialise(bad);bad.gasGeometry=s.gasGeometry;Real wrongV[2]={1.2,1.1};bad.gasGeometry.newVolume=wrongV;Real before=bad.rho[0];threadIdx.x=0;applyGasFluxDivergenceByCellKernel(&bad,.001);ck(bad.gasSpecies.cellStatus[0]!=0&&bad.rho[0]==before,"bad GCL modified conserved state");}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_single_legacy_explicit_conductivity_drives_actual_face_flux(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State a,b;initialise(a);initialise(b);
for(auto* s:{&a,&b}){s->gasSpecies.mode=ugkwp::GasMode::SingleLegacy;s->gasMu=0;s->gasThermalConductivity=0;
for(int c=0;c<2;++c){s->Tgas[c]=600+200*c;s->p[c]=s->rho[c]*s->Rgas*s->Tgas[c];s->rhoE[c]=s->p[c]/(s->gammaGas-1);threadIdx.x=c;computeGasPrimitiveGradientsKernel(s);}}
b.gasThermalConductivity=.42;threadIdx.x=0;computeGasInternalFaceFluxKernel<false>(&a,1e-6);computeGasInternalFaceFluxKernel<false>(&b,1e-6);
ck(std::abs((b.gasPhiRhoE[0]-a.gasPhiRhoE[0])-Real(-84))<Real(4),"single legacy explicit conductivity missing when viscosity zero");
ck(a.gasPhiRho[0]==b.gasPhiRho[0]&&a.gasPhiRhoUx[0]==b.gasPhiRhoUx[0],"conductivity changed nonthermal fluxes");}
''',bits)
