"""Composition-aware MUSCL exercises production gradients/limiters/face fluxes."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('scheme',[1,2])
@pytest.mark.parametrize('limiter',[0,1,2])
def test_muscl_reconstructs_composition_and_formation_energy(tmp_path,bits,scheme,limiter):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.gasFluxScheme=SCHEME;s.gasReconstruction=1;s.gasLimiter=LIMITER;
for(int c=0;c<2;++c){s.gasSpecies.rho[c]=c?Real(.3):Real(.7);s.gasSpecies.rho[2+c]=1-s.gasSpecies.rho[c];Real r[2]={s.gasSpecies.rho[c],s.gasSpecies.rho[2+c]};s.rhoE[c]=ugkwp::mixtureEnergy(r,Real(700),s.gasSpecies.thermo);threadIdx.x=c;recoverGasPrimitivesKernel(&s);}
for(int f=1;f<3;++f){s.riemannBoundaryKind[f]=0;s.gasSpecies.compositionBoundaryFixed[f]=1;s.gasSpecies.boundaryMassFraction[f]=f==1?Real(.8):Real(.2);s.gasSpecies.boundaryMassFraction[3+f]=1-s.gasSpecies.boundaryMassFraction[f];}
for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);computeGasGradientLimiterKernel(&s);}
threadIdx.x=0;computeGasInternalFaceFluxKernel<false>(&s,1e-5);
ck(s.gasSpecies.faceStatus[0]==0,"mixture MUSCL rejected");
const Real muscl=s.gasSpecies.flux[0];const Real energy=s.gasPhiRhoE[0];
s.gasReconstruction=0;computeGasInternalFaceFluxKernel<false>(&s,1e-5);
ck(std::abs(muscl)<Real(.8)*std::abs(s.gasSpecies.flux[0]),"MUSCL still transports cell-centre composition");
ck(std::abs(energy-Real(-3e6)*muscl)<Real(2e-5)*(1+std::abs(energy)),"MUSCL formation energy does not follow species flux");
s.gasReconstruction=1;transport(s,1e-6);for(int c=0;c<2;++c){ck(s.gasSpecies.cellStatus[c]==0,"MUSCL trial invalid");ck(std::abs(s.Tgas[c]-700)<Real(.03),"MUSCL stationary formation contact heated");}
}
'''.replace('SCHEME',str(scheme)).replace('LIMITER',str(limiter)),bits)

def test_chemistry_and_muscl_use_the_shared_transport_capabilities(tmp_path):
    compile_probe(tmp_path,r'''
#include "gasTransport/GasCapabilities.H"
int main(){ugkwp::GasCapabilityRequest r;r.mode=ugkwp::GasMode::MixtureChemistry;
for(int scheme=1;scheme<=2;++scheme)for(int limiter=0;limiter<=2;++limiter)for(int rk=1;rk<=3;++rk){r.fluxScheme=scheme;r.limiter=limiter;r.reconstruction=1;r.timeIntegrator=rk;if(!ugkwp::validateGasCapabilities(r))return 1;}
r.movingGeometry=true;r.timeIntegrator=1;if(!ugkwp::validateGasCapabilities(r))return 2;
r.timeIntegrator=2;if(ugkwp::validateGasCapabilities(r))return 3;}
''')

@pytest.mark.parametrize('bits',[32,64])
def test_muscl_face_simplex_and_eos_are_jointly_admissible(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.gasReconstruction=1;s.gasLimiter=0;
auto* m=const_cast<ugkwp::SpeciesThermoData<Real>*>(s.gasSpecies.thermo.species);m[1].molarMass=.044;
for(int c=0;c<2;++c){Real r[2]={s.gasSpecies.rho[c],s.gasSpecies.rho[2+c]};s.rhoE[c]=ugkwp::mixtureEnergy(r,Real(700),s.gasSpecies.thermo);threadIdx.x=c;recoverGasPrimitivesKernel(&s);computeGasPrimitiveGradientsKernel(&s);}
// Overshoots in Y and p must contract one face state rather than repair cells.
s.gasSpecies.gradX[0]=-20;s.gasSpecies.gradX[2]=20;s.gradPx[0]=-100*s.p[0];
threadIdx.x=0;computeGasGradientLimiterKernel(&s);Real before[3]={s.rho[0],s.rhoE[0],s.gasSpecies.rho[0]};
Real y[2];GasPrimDevice p;ck(reconstructGasMixtureFace(s,0,1,p,y),"joint admissibility failed");
ck(y[0]>=0&&y[1]>=0&&y[0]+y[1]==1,"reconstructed simplex violated");
ck(p.T>=100&&p.T<=3000&&p.p>0&&p.rho>0,"reconstructed thermodynamic range violated");
Real R=ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo);ck(std::abs(p.p-p.rho*R*p.T)<Real(1e-6)*p.p,"face EOS inconsistent");
ck(s.rho[0]==before[0]&&s.rhoE[0]==before[1]&&s.gasSpecies.rho[0]==before[2],"face reconstruction repaired conserved inputs");
// Extrapolated boundary composition follows reconstructed owner composition.
s.riemannBoundaryKind[1]=0;s.riemannBoundaryUFix[1]=0;s.gasSpecies.compositionBoundaryFixed[1]=0;
s.gradPx[0]=0;s.gasSpecies.gradX[0]=-.2;s.gasSpecies.gradX[2]=.2;computeGasGradientLimiterKernel(&s);
ck(reconstructGasMixtureFace(s,0,1,p,y),"boundary reconstruction failed");threadIdx.x=1;computeGasInternalFaceFluxKernel<false>(&s,1e-6);
ck(s.gasSpecies.faceStatus[1]==0,"extrapolated boundary rejected");
ck(std::abs(s.gasPhiRho[1])<Real(.002),"extrapolated boundary created spurious mass dissipation");
}
''',bits)

@pytest.mark.parametrize('bits',[32,64])
def test_mixture_gradients_ignore_scalar_eos_bootstrap(tmp_path,bits):
    compile_probe(tmp_path,fixture()+r'''
int main(){State a,b;initialise(a);initialise(b);b.Rgas=1e8;b.gammaGas=9;b.gasCp=1;b.TgasMin=100;
for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&a);computeGasPrimitiveGradientsKernel(&b);ck(a.gradPx[c]==b.gradPx[c]&&a.gradTX[c]==b.gradTX[c]&&a.gradRhoX[c]==b.gradRhoX[c],"mixture gradients use bootstrap scalar EOS");}
}
''',bits)

@pytest.mark.parametrize('scheme',[1,2])
def test_smooth_periodic_muscl_has_refinement_gain_over_first_order(tmp_path,scheme):
    source=fixture().replace('new Real[8]{}','new Real[132]{}').replace('new int[8]{}','new int[132]{}')
    for field in ['rho','initial','gradX','gradY','gradZ','positivityScale']:
        source=source.replace(f'g.{field}=new Real[4]{{}}',f'g.{field}=new Real[256]{{}}')
    for field in ['flux','boundaryMassFraction']:
        source=source.replace(f'g.{field}=new Real[6]{{}}',f'g.{field}=new Real[258]{{}}')
    for field in ['limiter','soundSpeed','heatCapacity','gasConstant']:
        source=source.replace(f'g.{field}=new Real[2]{{}}',f'g.{field}=new Real[128]{{}}')
    source=source.replace('g.compositionBoundaryFixed=new int[3]{}','g.compositionBoundaryFixed=new int[129]{}').replace('g.faceStatus=new int[3]{}','g.faceStatus=new int[129]{}').replace('g.cellStatus=new int[2]{}','g.cellStatus=new int[128]{}')
    compile_probe(tmp_path,source+r'''
Real error(int n,int reconstruction){State s;initialise(s);s.nCells=n;s.nFaces=n+1;s.nInternalFaces=n-1;s.gasFluxScheme=SCHEME;s.gasReconstruction=reconstruction;
const Real pi=std::acos(Real(-1)),dx=Real(1)/n,velocity=.5,average=std::sin(pi/n)/(pi/n);
for(int c=0;c<n;++c){s.Cx[c]=(c+Real(.5))*dx;s.V[c]=dx;s.rho[c]=1;s.rhoUx[c]=velocity;s.rhoUy[c]=s.rhoUz[c]=0;
s.gasSpecies.rho[c]=Real(.5)+Real(.1)*average*std::sin(2*pi*s.Cx[c]);s.gasSpecies.rho[n+c]=1-s.gasSpecies.rho[c];Real r[2]={s.gasSpecies.rho[c],s.gasSpecies.rho[n+c]};s.rhoE[c]=ugkwp::mixtureEnergy(r,Real(700),s.gasSpecies.thermo)+Real(.5)*velocity*velocity;
s.cellPlaneStart[c]=2*c;s.cellPlaneCount[c]=2;s.cellFaceId[2*c]=c==0?n-1:c-1;s.cellFaceId[2*c+1]=c==n-1?n:c;threadIdx.x=c;recoverGasPrimitivesKernel(&s);}
for(int f=0;f<n+1;++f){s.faceOwner[f]=f<n-1?f:f==n-1?0:n-1;s.faceNeighbour[f]=f<n-1?f+1:f==n-1?n-1:0;s.facePeriodicPair[f]=f<n-1?-1:f==n-1?n:n-1;s.facePeriodicDx[f]=f<n-1?0:f==n-1?-1:1;s.Sfx[f]=f==n-1?-1:1;s.magSf[f]=1;s.faceCx[f]=f<n-1?(f+1)*dx:f==n-1?0:1;s.faceWeight[f]=.5;s.deltaCoeffs[f]=1/dx;s.riemannBoundaryKind[f]=0;}
for(int c=0;c<n;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);computeGasGradientLimiterKernel(&s);}
for(int f=0;f<n+1;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<false>(&s,1e-6);ck(s.gasSpecies.faceStatus[f]==0,"periodic MUSCL face rejected");}
for(int f=n-1;f<n+1;++f){threadIdx.x=f;enforcePeriodicGasFluxAntisymmetryKernel(&s);}
Real norm=0;for(int c=0;c<n;++c){Real derivative=0;for(int j=0;j<2;++j){int f=s.cellFaceId[2*c+j];derivative+=(s.faceOwner[f]==c?-1:1)*s.gasSpecies.flux[f]/dx;}
Real exact=-velocity*Real(.1)*2*pi*average*std::cos(2*pi*s.Cx[c]);norm+=std::abs(derivative-exact)*dx;}return norm;}
int main(){Real first16=error(16,0),first32=error(32,0),muscl16=error(16,1),muscl32=error(32,1),muscl64=error(64,1);
ck(first16/first32>1.8&&first16/first32<2.2,"first-order control refinement unexpected");
ck(muscl16/muscl32>3.5&&muscl32/muscl64>3.5,"MUSCL lost second-order spatial refinement");
ck(muscl32<Real(.02)*first32,"MUSCL did not improve smooth periodic transport");}
'''.replace('SCHEME',str(scheme)))

@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('scheme',[1,2])
def test_muscl_boundary_validates_only_with_reconstructed_composition(tmp_path,bits,scheme):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.gasReconstruction=1;s.gasFluxScheme=SCHEME;
auto* m=const_cast<ugkwp::SpeciesThermoData<Real>*>(s.gasSpecies.thermo.species);m[0].molarMass=.002;m[1].molarMass=.044;
auto* coefficients=const_cast<Real*>(s.gasSpecies.thermo.coefficients);coefficients[0]=15000;coefficients[3]=1000;
for(int c=0;c<2;++c){s.gasSpecies.rho[c]=.1;s.gasSpecies.rho[2+c]=.9;Real r[2]={.1,.9};s.rhoE[c]=ugkwp::mixtureEnergy(r,Real(700),s.gasSpecies.thermo);threadIdx.x=c;recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"invalid setup");s.gasGradientLimiterP[c]=s.gasGradientLimiterRho[c]=s.gasGradientLimiterUx[c]=s.gasGradientLimiterUy[c]=s.gasGradientLimiterUz[c]=1;s.gasSpecies.limiter[c]=1;}
s.gasSpecies.gradX[0]=-.2;s.gasSpecies.gradX[2]=.2;
s.riemannBoundaryKind[1]=0;s.riemannBoundaryPFix[1]=s.riemannBoundaryRhoFix[1]=1;s.riemannBoundaryRho[1]=1;
Real desiredY[2]={.2,.8};s.riemannBoundaryP[1]=ugkwp::mixtureGasConstant(desiredY,s.gasSpecies.thermo)*2000;
GasPrimDevice left;Real faceY[2];ck(reconstructGasMixtureFace(s,0,1,left,faceY),"valid reconstruction rejected");
auto right=riemannBoundaryState(s,1,left,faceY);ck(s.gasSpecies.faceStatus[1]==0&&std::abs(right.T-2000)<Real(.01),"valid composition-specific boundary rejected");
Real mass,mx,my,mz,energy;
ck(computeRiemannGasFaceFluxDevice<false>(s,1,mass,mx,my,mz,energy),"premature cell-composition boundary EOS rejected valid MUSCL face");
ck(s.gasSpecies.faceStatus[1]==0&&std::isfinite(energy),"valid MUSCL boundary left failure status");
// The table bound still applies to the actual reconstructed composition.
s.riemannBoundaryP[1]=ugkwp::mixtureGasConstant(desiredY,s.gasSpecies.thermo)*4000;
ck(!computeRiemannGasFaceFluxDevice<false>(s,1,mass,mx,my,mz,energy)&&s.gasSpecies.faceStatus[1]!=0,"invalid actual boundary temperature accepted");
}
'''.replace('SCHEME',str(scheme)),bits)
