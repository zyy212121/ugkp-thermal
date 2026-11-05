"""A nonzero trace keeps its own Gauss stencil and conservative scalar flux."""
import pytest
from test_mixture_state import compile_probe
from test_populated_species_closure_f3 import closure_fixture

@pytest.mark.parametrize("scheme", [1, 2, 4])
@pytest.mark.parametrize("reconstruction", [0, 1])
def test_trace_does_not_close_bulk_gradient_or_flux(tmp_path, scheme, reconstruction):
    compile_probe(tmp_path, closure_fixture()+r"""
int main(){
 for(int trace=0;trace<3;++trace)for(int exponent: {40,120,240}){
  State s;initialise(s);s.gasFluxScheme=SCHEME;s.gasReconstruction=RECONSTRUCTION;
  const int carrier=(trace+1)%3;const Real tiny=std::pow(Real(10),-exponent);
  for(int c=0;c<2;++c){
   for(int k=0;k<3;++k)s.gasSpecies.rho[k*2+c]=0;
   s.gasSpecies.rho[carrier*2+c]=1;s.gasSpecies.rho[trace*2+c]=(c+1)*tiny;
   s.Ux[c]=10;recover(s,c);
  }
  for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);computeGasGradientLimiterKernel(&s);
   const Real expected=(c?s.faceWeight[0]:1-s.faceWeight[0])*tiny;
   ck(std::abs((s.gasSpecies.gradX[trace*2+c]-expected)/expected)<Real(1e-13),"carrier roundoff replaced trace Gauss gradient");
  }
  face(s);
  const Real traceFlux=s.gasSpecies.flux[trace*3];
  ck(std::isfinite(traceFlux)&&std::abs(traceFlux)<Real(1000)*tiny,"carrier-sized convective remainder assigned to trace");
  const Real mass=s.gasPhiRho[0],pressure=s.gasPhiRhoUx[0],energy=s.gasPhiRhoE[0];
  for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,1e-6);}
  threadIdx.x=0;applyGasFluxPositivityScaleKernel(&s);
  ck(s.gasPhiRho[0]==mass&&s.gasPhiRhoUx[0]==pressure&&s.gasPhiRhoE[0]==energy,"trace roundoff shut down bulk transport");
 }
}
""".replace("SCHEME",str(scheme)).replace("RECONSTRUCTION",str(reconstruction)))

@pytest.mark.parametrize("limiter", [1, 2])
def test_steep_trace_front_limits_bulk_outflow_independently_of_scale(tmp_path, limiter):
    compile_probe(tmp_path, closure_fixture()+r"""
int main(){
 for(int trace=0;trace<3;++trace)for(int exponent: {40,120,240}){
  State s;initialise(s);s.gasFluxScheme=4;s.gasReconstruction=1;s.gasLimiter=LIMITER;
  const int carrier=(trace+1)%3;const Real tiny=std::pow(Real(10),-exponent);
  for(int c=0;c<2;++c){
   for(int k=0;k<3;++k)s.gasSpecies.rho[k*2+c]=0;
   s.gasSpecies.rho[carrier*2+c]=1;s.gasSpecies.rho[trace*2+c]=tiny*(c?Real(1e20):Real(1));
   s.Ux[c]=10;recover(s,c);
  }
  for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s);}
  for(int c=0;c<2;++c){threadIdx.x=c;computeGasGradientLimiterKernel(&s);}
  ck(s.gasSpecies.limiter[0]<Real(1e-18),"tiny profile bypassed cell slope limiter");
  face(s);
  ck(s.gasSpecies.flux[trace*3]>=0&&s.gasSpecies.flux[trace*3]<=Real(21)*tiny,"trace outflow exceeds limited donor reconstruction");
  const Real mass=s.gasPhiRho[0],pressure=s.gasPhiRhoUx[0],energy=s.gasPhiRhoE[0];
  for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,1e-6);}
  threadIdx.x=0;applyGasFluxPositivityScaleKernel(&s);
  ck(s.gasPhiRho[0]==mass&&s.gasPhiRhoUx[0]==pressure&&s.gasPhiRhoE[0]==energy,"trace front shut down bulk transport");
 }
}
""".replace("LIMITER",str(limiter)))

def test_nonorthogonal_diffusion_uses_admissible_trace_reconstruction(tmp_path):
    compile_probe(tmp_path, closure_fixture()+r"""
int main(){
 for(int trace=0;trace<3;++trace){
  State s;initialise(s);s.gasFluxScheme=4;s.gasReconstruction=1;s.gasLimiter=2;
  const int carrier=(trace+1)%3;
  Real diffusion[3]={1e-4,2e-4,3e-4};s.gasSpecies.diffusivity=diffusion;
  s.Cy[1]=.1;s.faceCy[0]=.05;s.faceCy[1]=-.05;s.faceCy[2]=.15;
  for(int c=0;c<2;++c){
   for(int k=0;k<3;++k)s.gasSpecies.rho[k*2+c]=0;
   s.gasSpecies.rho[carrier*2+c]=1;s.Ux[c]=10;recover(s,c);
   // Tangential slopes supplied by the wider multidimensional stencil.
   s.gasSpecies.gradY[trace*2+c]=1e-100*(c+1);
   s.gasSpecies.gradY[carrier*2+c]=-s.gasSpecies.gradY[trace*2+c];
   threadIdx.x=c;computeGasGradientLimiterKernel(&s);
   ck(s.gasSpecies.limiter[c]==0,"zero trace has an inadmissible reconstructed slope");
  }
  face(s);ck(s.gasSpecies.flux[trace*3]==0,"nonorthogonal diffusion exported absent trace");
  const Real mass=s.gasPhiRho[0],momentum=s.gasPhiRhoUx[0],energy=s.gasPhiRhoE[0];
  for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,1e-6);}
  threadIdx.x=0;applyGasFluxPositivityScaleKernel(&s);
  ck(s.gasPhiRho[0]==mass&&s.gasPhiRhoUx[0]==momentum&&s.gasPhiRhoE[0]==energy,"absent trace correction suppressed bulk transport");
 }
}
""")

def test_admissible_linear_nonorthogonal_diffusion_and_enthalpy(tmp_path):
    compile_probe(tmp_path, closure_fixture()+r"""
int main(){
 for(int trace=0;trace<3;++trace){
  State s;initialise(s);s.gasFluxScheme=4;s.gasReconstruction=0;
  const int carrier=(trace+1)%3;Real D[3]={1e-4,2e-4,3e-4};s.gasSpecies.diffusivity=D;
  s.Cy[1]=.1;s.faceCy[0]=.05;
  const Real gx=.01,gy=.02;
  for(int c=0;c<2;++c){
   for(int k=0;k<3;++k)s.gasSpecies.rho[k*2+c]=0;
   s.gasSpecies.rho[trace*2+c]=.2+gx*s.Cx[c]+gy*s.Cy[c];
   s.gasSpecies.rho[carrier*2+c]=1-s.gasSpecies.rho[trace*2+c];s.Ux[c]=10;recover(s,c);
   s.gasSpecies.gradX[trace*2+c]=gx;s.gasSpecies.gradY[trace*2+c]=gy;
   s.gasSpecies.gradX[carrier*2+c]=-gx;s.gasSpecies.gradY[carrier*2+c]=-gy;
   s.gasSpecies.limiter[c]=1;
  }
  s.gasSpecies.diffusivity=nullptr;face(s);Real conv[3];for(int k=0;k<3;++k)conv[k]=s.gasSpecies.flux[k*3];
  const Real baseEnergy=s.gasPhiRhoE[0];s.gasSpecies.diffusivity=D;face(s);
  Real Y[3],normal[3]={},j[3],enthalpy;
  for(int k=0;k<3;++k)Y[k]=s.faceWeight[0]*s.gasSpecies.rho[k*2]+(1-s.faceWeight[0])*s.gasSpecies.rho[k*2+1];
  normal[trace]=gx;normal[carrier]=-gx;
  ck(ugkwp::correctedSpeciesDiffusion(Y,normal,Real(1),D,Real(0),Real(.7),Real(700),s.gasSpecies.thermo,j,enthalpy),"analytic diffusion rejected");
  for(int k=0;k<3;++k)ck(std::abs((s.gasSpecies.flux[k*3]-conv[k])-j[k])<1e-14,"linear nonorthogonal diffusion changed");
  ck(std::abs((s.gasPhiRhoE[0]-baseEnergy)-enthalpy)<1e-8,"diffusion enthalpy inconsistent with species flux");
 }
}
""")

def test_limited_skew_flux_preserves_cell_pair_mass_species_and_energy(tmp_path):
    compile_probe(tmp_path, closure_fixture()+r"""
int main(){
 for(int trace=0;trace<3;++trace)for(int exponent: {0,40,240}){
  State s;initialise(s);s.gasFluxScheme=4;s.gasReconstruction=1;s.gasLimiter=2;
  const int carrier=(trace+1)%3;const Real tiny=std::pow(Real(10),-exponent);
  Real D[3]={1e-4,2e-4,3e-4};s.gasSpecies.diffusivity=D;
  s.Cy[1]=.1;s.faceCy[0]=.05;s.faceCy[1]=-.05;s.faceCy[2]=.15;
  Real inventory[3]={};
  for(int c=0;c<2;++c){
   for(int k=0;k<3;++k)s.gasSpecies.rho[k*2+c]=0;
   s.gasSpecies.rho[trace*2+c]=(.2+.1*c)*tiny;
   s.gasSpecies.rho[carrier*2+c]=1-s.gasSpecies.rho[trace*2+c];s.Ux[c]=10;recover(s,c);
   s.gasSpecies.gradY[trace*2+c]=10*tiny;
   s.gasSpecies.gradY[carrier*2+c]=-10*tiny;
   threadIdx.x=c;computeGasGradientLimiterKernel(&s);
   for(int k=0;k<3;++k)inventory[k]+=s.gasSpecies.rho[k*2+c];
  }
  const Real energy=s.rhoE[0]+s.rhoE[1],mass=s.rho[0]+s.rho[1],dt=1e-6;
  for(int f=0;f<3;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<false>(&s,dt);}
  for(int c=0;c<2;++c){threadIdx.x=c;computeGasFluxPositivityScaleKernel(&s,dt);}
  for(int f=0;f<3;++f){threadIdx.x=f;applyGasFluxPositivityScaleKernel(&s);}
  const Real boundaryEnergy=s.gasPhiRhoE[1]+s.gasPhiRhoE[2];
  const Real boundaryMass=s.gasPhiRho[1]+s.gasPhiRho[2];
  Real boundarySpecies[3];for(int k=0;k<3;++k)boundarySpecies[k]=s.gasSpecies.flux[k*3+1]+s.gasSpecies.flux[k*3+2];
  for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,dt);}
  const Real eps=std::numeric_limits<Real>::epsilon();
  ck(std::abs(s.rho[0]+s.rho[1]-mass+dt*boundaryMass)<64*eps*mass,"cell-pair mass balance");
  ck(std::abs(s.rhoE[0]+s.rhoE[1]-energy+dt*boundaryEnergy)<128*eps*std::max(std::abs(energy),Real(1)),"cell-pair energy balance");
  for(int k=0;k<3;++k){
   const Real actual=s.gasSpecies.rho[k*2]+s.gasSpecies.rho[k*2+1];
   ck(s.gasSpecies.rho[k*2]>=0&&s.gasSpecies.rho[k*2+1]>=0,"negative species inventory");
   ck(std::abs(actual-inventory[k]+dt*boundarySpecies[k])<=128*eps*std::max(inventory[k],std::abs(dt*boundarySpecies[k])),"cell-pair species balance");
  }
 }
}
""")
