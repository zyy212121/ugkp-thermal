"""Public estimator publishes actual IEEE fields, retaining strict retry policy."""
import pytest
from test_mixture_state import compile_probe
from test_mixture_transport import fixture

@pytest.mark.parametrize('channel',['convective'])
def test_final_courant_field_is_not_a_finite_placeholder(tmp_path,channel):
    statements={
        'convective':'s.gasPhiRhoE[0]=INFINITY;computeGasConvectiveCourantByCellKernel(&s,1.);ck(std::isinf(s.gasFluxPositivityScale[0]),"actual convective infinity was replaced");',
    }
    compile_probe(tmp_path,fixture()+'\nint main(){State s;initialise(s);threadIdx.x=0;'+statements[channel]+'}')
