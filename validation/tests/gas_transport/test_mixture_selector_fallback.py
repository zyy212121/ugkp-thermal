"""Expose existing fallback arithmetic without applying the legacy Cp*T channel."""
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

def test_existing_mixture_fallback_complete_fluxes_and_actual_scheme(tmp_path):
    code=fixture()+r'''
#include "operators/computeGasPrimitiveGradientsKernel.cuh"
int main(){State s;initialise(s);Real oldV[2]={1,1},newV[2]={1,1},sweep[3]={.001,0,0};
 s.rhoUx[0]=s.rho[0]*13;s.rhoUx[1]=s.rho[1]*7;s.Ux[0]=13;s.Ux[1]=7;
 s.gradTX[0]=s.gradTX[1]=300;s.gradUxX[0]=s.gradUxX[1]=11;
 s.rhoE[0]+=s.rho[0]*.5*13*13;s.rhoE[1]+=s.rho[1]*.5*7*7;
 s.sstConfigured=1;s.gasFluxScheme=7;s.gasHllcAdcSensor[0]=-123;
 blockIdx.x=threadIdx.x=0;blockDim.x=1;computeGasPrimitiveGradientsKernel(&s,true);
 ck(s.gasHllcAdcSensor[0]==-123,"mixture fallback computed unused ADC sensor");
 for(int moving:{0,1})for(int face:{0,2})for(int scheme=3;scheme<=9;++scheme)for(int recon:{0,2}){
  s.gasGeometry.enabled=moving;s.gasGeometry.oldVolume=oldV;s.gasGeometry.newVolume=newV;s.gasGeometry.faceSweptVolume=sweep;s.gasGeometry.interval=.01;
  s.gasSpecies.cellStatus[0]=s.gasSpecies.cellStatus[1]=0;for(int i=0;i<3;++i)s.gasSpecies.faceStatus[i]=0;
  Real expected[5],actual[5],species[2];s.gasFluxScheme=1;s.gasReconstruction=0;
  ck(computeRiemannGasFaceFluxDevice<false>(s,face,expected[0],expected[1],expected[2],expected[3],expected[4],true),"reference flux failed");
  for(int i=0;i<2;++i)species[i]=s.gasSpecies.flux[i*s.nFaces+face];
  s.gasFluxScheme=scheme;s.gasReconstruction=recon;s.gasCp=0;
  ugkwp::GasCapabilityRequest request;request.mode=s.gasSpecies.mode;request.fluxScheme=scheme;request.reconstruction=recon;
  ck(bool(ugkwp::validateGasCapabilities(request)),"existing fallback blocked by certificate");
  ck(computeRiemannGasFaceFluxDevice<false>(s,face,actual[0],actual[1],actual[2],actual[3],actual[4],true),"fallback flux rejected");
  for(int i=0;i<5;++i)ck(actual[i]==expected[i],"fallback changed actual conservative flux");
  for(int i=0;i<2;++i)ck(s.gasSpecies.flux[i*s.nFaces+face]==species[i],"fallback species flux differs");
  s.gasCp=1e10;
  ck(computeRiemannGasFaceFluxDevice<false>(s,face,actual[0],actual[1],actual[2],actual[3],actual[4],true),"Cp independent fallback rejected");
  for(int i=0;i<5;++i)ck(actual[i]==expected[i],"legacy Cp contaminated formation-inclusive mixture flux");
  ck((computeRiemannGasFaceFluxDevice<false,true>(s,face,actual[0],actual[1],actual[2],actual[3],actual[4],true)),"mass-only fallback rejected");
  ck(actual[0]==expected[0],"mass-only fallback differs");
 }
}
'''
    compile_probe(tmp_path,code)
