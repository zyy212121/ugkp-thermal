"""Capability requests admit only resident wall-aware particle paths."""
from test_mixture_state import compile_probe

def test_wall_requires_shared_thermo_and_particle_lifecycle(tmp_path):
    compile_probe(tmp_path,r'''
#include "gasTransport/GasCapabilities.H"
int main(){
 ugkwp::GasCapabilityRequest request;
 request.mode=ugkwp::GasMode::MixtureFrozen;
 request.sstWallTreatment=2;request.boundaryLayerModel=2;
 request.particleCoupling=true;request.pairedMixtureParticleCoupling=true;
 if(ugkwp::validateGasCapabilities(request))return 1;
 request.residentParticleWallCoupling=true;
 if(!ugkwp::validateGasCapabilities(request))return 2;
 request.mode=ugkwp::GasMode::SingleLegacy;
 if(ugkwp::validateGasCapabilities(request))return 3;
 request.mode=ugkwp::GasMode::MixtureFrozen;
 request.pairedMixtureParticleCoupling=false;
 if(ugkwp::validateGasCapabilities(request))return 4;
 return 0;
}
''')
