"""Public estimator publishes actual IEEE fields, retaining strict retry policy."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

@pytest.mark.parametrize('channel',['convective','diffusion','sst'])
def test_final_courant_field_is_not_a_finite_placeholder(tmp_path,channel):
    statements={
        'convective':'s.gasPhiRhoE[0]=INFINITY;computeGasConvectiveCourantByCellKernel(&s,1.);ck(std::isinf(s.gasFluxPositivityScale[0]),"actual convective infinity was replaced");',
        'diffusion':'s.gasMu=INFINITY;computeGasDiffusionNumberKernel(&s,1.,.5);ck(std::isinf(s.gasDiffusionNumber[0]),"actual diffusion infinity was replaced");',
        'sst':'s.sstConfigured=1;s.sstF1[0]=1;s.rhoK[0]=1;s.rhoOmega[0]=1;s.k[0]=1;s.omega[0]=1;s.gasMu=INFINITY;computeSstStabilityNumberKernel(&s,1.,.5,false);ck(!std::isfinite(s.sstSourceNumber[0]),"actual SST bound was replaced");',
    }
    compile_probe(tmp_path,fixture()+'\nint main(){State s;initialise(s);threadIdx.x=0;'+statements[channel]+'}')
