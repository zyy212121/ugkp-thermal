"""Acoustic queries leave physical mass flux untouched in laminar and SST modes."""
from validation.tests.gas_transport.test_mixture_state import compile_probe
from validation.tests.gas_transport.test_legacy_ale_transport import fixture

def test_acoustic_query_preserves_mass_flux(tmp_path):
    compile_probe(tmp_path, fixture()+r'''
int main(){State s;initialise(s);for(int mode:{0,1}){s.sstConfigured=mode;for(int f=0;f<3;++f){threadIdx.x=f;s.gasPhiRho[f]=12345;computeGasCourantFieldKernel(&s,.01);ck(s.gasPhiRho[f]==12345,"acoustic query overwrote physical mass flux");ck(std::isfinite(s.gasPhiRhoE[f]),"nonfinite acoustic scratch");}for(int c=0;c<2;++c){threadIdx.x=c;computeGasConvectiveCourantByCellKernel(&s,.01);ck(s.gasFluxPositivityScale[c]>=0,"negative acoustic Courant");}}}
''')
