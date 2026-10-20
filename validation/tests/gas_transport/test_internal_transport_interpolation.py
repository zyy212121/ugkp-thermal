"""Interior diffusion interpolates physical transport coefficients at the face.

Equal-weight nut averaging and products of interpolated rho/cp/nut fail these
linear-field and correlated-coefficient checks on the actual shared operators.
"""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


@pytest.mark.parametrize('reconstruction',[0,1])
@pytest.mark.parametrize('weight',[.25,.75])
def test_nonuniform_linear_shear_and_temperature_reproduce_face_transport(tmp_path,reconstruction,weight):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.turbulenceModel=3;s.gasMu=.001;s.turbulentPrandtl=.9;
s.gasFluxScheme=4;s.gasReconstruction=RECON;s.faceWeight[0]=WEIGHT;s.faceCx[0]=1-WEIGHT;
for(int c=0;c<2;++c){
 const Real temperature=500+10*c,velocity=2+3*c;Real partial[2]={.4,.6};
 s.rho[c]=1;s.rhoUx[c]=s.rhoUz[c]=0;s.rhoUy[c]=velocity;
 s.gasSpecies.rho[c]=.4;s.gasSpecies.rho[2+c]=.6;
 s.rhoE[c]=ugkwp::mixtureEnergy(partial,temperature,s.gasSpecies.thermo)+.5*velocity*velocity;
 threadIdx.x=c;recoverGasPrimitivesKernel(&s);ck(s.gasSpecies.cellStatus[c]==0,"linear transport state");
 s.nut[c]=.1+.2*c;s.gradUyX[c]=3;s.gradTX[c]=10;s.gradPx[c]=10*s.gasSpecies.gasConstant[c];
 s.gasGradientLimiterRho[c]=s.gasGradientLimiterP[c]=s.gasGradientLimiterUx[c]=s.gasGradientLimiterUy[c]=s.gasGradientLimiterUz[c]=s.gasSpecies.limiter[c]=1;
}
Real full[5],molecular[5];
ck(computeRiemannGasFaceFluxDevice<true>(s,0,full[0],full[1],full[2],full[3],full[4]),"full transport face");
ck(computeRiemannGasFaceFluxDevice<false>(s,0,molecular[0],molecular[1],molecular[2],molecular[3],molecular[4]),"molecular transport face");
const Real mu=.1+.2*(1-WEIGHT),velocity=2+3*(1-WEIGHT);
const Real shear=full[2]-molecular[2],energy=full[4]-molecular[4];
if(std::abs(shear+3*mu)>1e-10)std::fprintf(stderr,"nonuniform face shear=%.17g expected=%.17g\n",shear,-3*mu);
ck(std::abs(shear+3*mu)<1e-10,"linear turbulent viscosity was not evaluated at the face");
ck(std::abs(energy+3*mu*velocity+1000*mu/.9*10)<1e-8,"linear turbulent conductivity/work was not evaluated at the face");
}
'''.replace('RECON',str(reconstruction)).replace('WEIGHT',str(weight)))


def test_correlated_density_nut_cp_interpolates_dynamic_coefficients(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.turbulenceModel=3;s.turbulentPrandtl=.9;s.faceWeight[0]=.25;
s.rho[0]=2;s.rho[1]=8;s.nut[0]=3;s.nut[1]=1;s.gasSpecies.heatCapacity[0]=1000;s.gasSpecies.heatCapacity[1]=2000;
for(Real reconstructedDensity:{Real(1),Real(5),Real(9)}){
 Real mu=0,kappa=0,q=0;int active=0;
 gasFaceSubgridTransportProperties(s,0,0,1,0,reconstructedDensity,mu,kappa,q,active);
 // .25*(2*3)+.75*(8*1)=7.5, (.25*6000+.75*16000)/.9=15000.
 ck(mu==7.5,"rho*nut was multiplied after interpolation or used reconstructed density");
 ck(kappa==15000,"cp*rho*nut was multiplied after interpolation");
 ck(!active&&q==0,"internal diffusion acquired a direct wall heat source");
}
// Reverse the face and interpolation weight: the physical coefficient is unchanged.
std::swap(s.rho[0],s.rho[1]);std::swap(s.nut[0],s.nut[1]);std::swap(s.gasSpecies.heatCapacity[0],s.gasSpecies.heatCapacity[1]);s.faceWeight[0]=.75;
Real mu=0,kappa=0,q=0;int active=0;
gasFaceSubgridTransportProperties(s,0,0,1,0,Real(5),mu,kappa,q,active);
ck(mu==7.5&&kappa==15000,"coefficient interpolation depends on face orientation");
}
''')

def test_diffusion_timestep_uses_same_weight_from_either_cell(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);s.turbulenceModel=3;s.gasMu=0;s.turbulentPrandtl=.9;s.faceWeight[0]=.25;s.maxDiffusionNumber=.25;
s.rho[0]=2;s.rho[1]=8;s.nut[0]=3;s.nut[1]=1;s.gasSpecies.heatCapacity[0]=1000;s.gasSpecies.heatCapacity[1]=2000;
s.gasSpecies.gasConstant[0]=s.gasSpecies.gasConstant[1]=300;s.gasSpecies.turbulentSchmidt=1e6;
for(int c=0;c<2;++c){s.cellPlaneStart[c]=0;s.cellPlaneCount[c]=1;threadIdx.x=c;computeGasDiffusionNumberKernel(&s,Real(.1),Real(.5));
 // Same face: conductivity 15000 / (rhoFace 5 * cvFace 1200) = 2.5.
 ck(std::abs(s.gasDiffusionNumber[c]-.5)<1e-14,"timestep reversed the geometric coefficient weight");}
}
''')

def test_internal_transport_instantiates_in_cuda_device_code(tmp_path):
    import os
    import shutil
    import subprocess
    from test_mixture_state import ROOT, HERE
    nvcc=os.environ.get('NVCC') or shutil.which('nvcc')
    if not nvcc:
        pytest.skip('CUDA compiler unavailable')
    pre=(HERE/'gas_state_probe.cpp').read_text().split('void hashBytes')[0]
    adapters=('#define __host__','#define __global__','#define __device__',
              '#define __forceinline__','#define asm(', 'struct ThreadIndex ')
    pre='\n'.join(line for line in pre.splitlines() if not line.startswith(adapters))
    body=r'''
struct State:ugkwp::GasStateView<Real,Time,ugkwp::SstCoefficients>{ugkwp::GasSpeciesState<Real,2>gasSpecies;};
__global__ void transport(State*s,Real*out){int active=0;
 gasFaceSubgridTransportProperties(*s,0,0,1,0,Real(1),out[0],out[1],out[2],active);
 out[3]=active;
}
'''
    source=tmp_path/'transport.cu';source.write_text('#include <cuda_runtime.h>\n'+pre+body)
    result=subprocess.run([nvcc,'-std=c++17','-O2','-arch=sm_89','--ptx',
        '-I'+str(HERE),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),
        str(source),'-o',str(tmp_path/'transport.ptx')],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr