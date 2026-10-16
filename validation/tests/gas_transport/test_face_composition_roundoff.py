"""Closure accepted by recovery must not create inadmissible face composition."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

@pytest.mark.parametrize('pure',[0,1])
@pytest.mark.parametrize('side',['reconstruction','boundary'])
def test_roundoff_density_closure_preserves_pure_species_face(tmp_path,pure,side):
    body=r'''
int main(){State s;initialise(s);s.gasReconstruction=1;
const int pure=PURE;
for(int c=0;c<2;++c){s.rho[c]=1;s.gasSpecies.rho[c]=pure==0?std::nextafter(Real(1),Real(2)):0;s.gasSpecies.rho[2+c]=pure==1?std::nextafter(Real(1),Real(2)):0;
Real r[2]={s.gasSpecies.rho[c],s.gasSpecies.rho[2+c]};s.rhoE[c]=ugkwp::mixtureEnergy(r,Real(700),s.gasSpecies.thermo);
threadIdx.x=c;recoverGasPrimitivesKernel(&s,true);ck(s.gasSpecies.cellStatus[c]==0,"closure-valid cell rejected");
s.gasSpecies.limiter[c]=1;s.gasGradientLimiterRho[c]=s.gasGradientLimiterP[c]=s.gasGradientLimiterUx[c]=s.gasGradientLimiterUy[c]=s.gasGradientLimiterUz[c]=1;}
Real before[4]={s.rho[0],s.rhoE[0],s.gasSpecies.rho[0],s.gasSpecies.rho[2]};
SIDE
ck(s.rho[0]==before[0]&&s.rhoE[0]==before[1]&&s.gasSpecies.rho[0]==before[2]&&s.gasSpecies.rho[2]==before[3],"face composition changed conserved inventory");
}
'''.replace('PURE',str(pure))
    call=r'''
GasPrimDevice face;Real y[2];bool available=false;
ck(reconstructGasMixtureFace(s,0,0,face,y,true,&available)&&available,"roundoff rejected reconstructed face");
ck(y[pure]==1&&y[1-pure]==0,"pure face gained/lost species");
ck(std::isfinite(face.T)&&face.T>0,"nonphysical temperature");
'''
    if side=='boundary':
        call=r'''
s.riemannBoundaryKind[1]=0;s.gasSpecies.compositionBoundaryFixed[1]=0;
auto p=gasCellPrimitive(s,0);auto face=riemannBoundaryState(s,1,p,nullptr,true);
ck(s.gasSpecies.faceStatus[1]==0,"roundoff rejected boundary face");
ck(std::isfinite(face.T)&&face.T>0,"nonphysical boundary temperature");
'''
    compile_probe(tmp_path,fixture()+body.replace('SIDE',call))

@pytest.mark.parametrize('ns',[1,2,3,10,37])
def test_normalized_composition_preserves_zeros_and_invalid_inventory(tmp_path,ns):
    compile_probe(tmp_path,r'''
#include "gasTransport/MixtureThermo.H"
void ck(bool good,const char* message){if(!good){std::fprintf(stderr,"%s\n",message);std::abort();}}
int main(){constexpr int N=NS;Real partial[N]{},y[N];const Real tol=64*std::numeric_limits<Real>::epsilon();
for(int dominant=0;dominant<N;++dominant){for(int k=0;k<N;++k){partial[k]=k==dominant?std::nextafter(Real(1),Real(2)):0;y[k]=-7;}
ck(ugkwp::mixtureMassFractionsFromDensities<Real,N>(partial,Real(1),tol,y),"closure roundoff rejected");
for(int k=0;k<N;++k)ck(y[k]==(k==dominant?Real(1):Real(0)),"pure composition/absent species changed");}
for(int k=0;k<N;++k){partial[k]=Real(k+1);y[k]=-7;}
const Real total=Real(N)*(N+1)/2;
ck(ugkwp::mixtureMassFractionsFromDensities<Real,N>(partial,std::nextafter(total,Real(0)),tol,y),"valid multispecies closure rejected");
Real sum=0;for(int k=0;k<N;++k){ck(y[k]>=0&&y[k]<=1,"simplex violation");sum+=y[k];}
ck(sum==1,"composition not closed");
for(int mode=0;mode<4;++mode){for(int k=0;k<N;++k){partial[k]=k==0?1:0;y[k]=-7;}
Real density=1;if(mode==0)partial[0]=-std::numeric_limits<Real>::epsilon();if(mode==1)partial[0]=std::numeric_limits<Real>::quiet_NaN();if(mode==2)density=2;if(mode==3)density=0;
ck(!ugkwp::mixtureMassFractionsFromDensities<Real,N>(partial,density,tol,y),"invalid inventory repaired");for(int k=0;k<N;++k)ck(y[k]==-7,"failure published partial composition");}
}
'''.replace('NS',str(ns)))


def test_external_boundary_composition_and_physical_temperature_still_reject(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.riemannBoundaryKind[1]=0;s.gasSpecies.compositionBoundaryFixed[1]=1;
s.gasSpecies.boundaryMassFraction[1]=std::nextafter(Real(1),Real(2));s.gasSpecies.boundaryMassFraction[4]=0;
auto p=gasCellPrimitive(s,0);riemannBoundaryState(s,1,p,nullptr,true);
ck(s.gasSpecies.faceStatus[1]!=0,"invalid external boundary composition repaired");
s.gasSpecies.faceStatus[1]=0;s.gasSpecies.boundaryMassFraction[1]=1;s.gasSpecies.boundaryMassFraction[4]=0;s.riemannBoundaryTFix[1]=1;s.riemannBoundaryT[1]=50;
riemannBoundaryState(s,1,p,nullptr,true);ck(s.gasSpecies.faceStatus[1]!=0,"physical thermo range relaxed");
}
''')

def test_native_pure_mixture_step_conserves_inventories_after_roundoff_closure(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.gasReconstruction=1;
for(int c=0;c<2;++c){s.rho[c]=1;s.gasSpecies.rho[c]=c?Real(1):std::nextafter(Real(1),Real(2));s.gasSpecies.rho[2+c]=0;
Real r[2]={s.gasSpecies.rho[c],0};s.rhoE[c]=ugkwp::mixtureEnergy(r,Real(700),s.gasSpecies.thermo);threadIdx.x=c;recoverGasPrimitivesKernel(&s,true);ck(s.gasSpecies.cellStatus[c]==0,"valid recovered inventory rejected");}
const Real mass=s.rho[0]+s.rho[1],energy=s.rhoE[0]+s.rhoE[1],species=s.gasSpecies.rho[0]+s.gasSpecies.rho[1];
for(int c=0;c<2;++c){threadIdx.x=c;computeGasPrimitiveGradientsKernel(&s,false,true);computeGasGradientLimiterKernel(&s,true);}
for(int f=0;f<3;++f){threadIdx.x=f;computeGasInternalFaceFluxKernel<false>(&s,1e-7,true);ck(s.gasSpecies.faceStatus[f]==0,"closure-valid native flux rejected");ck(s.gasSpecies.flux[3+f]==0,"absent species flux created");ck(s.gasSpecies.flux[f]==s.gasPhiRho[f],"pure mass flux not closed");}
for(int c=0;c<2;++c){threadIdx.x=c;applyGasFluxDivergenceByCellKernel(&s,1e-7,true);recoverGasPrimitivesKernel(&s,true);ck(s.gasSpecies.cellStatus[c]==0,"native update rejected");ck(s.gasSpecies.rho[2+c]==0,"absent species inventory created");}
const Real eps=std::numeric_limits<Real>::epsilon();
ck(std::abs(s.rho[0]+s.rho[1]-mass)<=4*eps*mass,"total mass changed");
ck(std::abs(s.rhoE[0]+s.rhoE[1]-energy)<=4*eps*std::abs(energy),"total energy changed");
ck(std::abs(s.gasSpecies.rho[0]+s.gasSpecies.rho[1]-species)<=4*eps*species,"species inventory changed");
}
''')
