"""Ordinary Q/PB wall operators must overwrite all species channels."""
import pytest
from test_mixture_state import compile_probe
from test_legacy_ale_transport import fixture
from test_actual_wall_producer_consumers import SETUP

@pytest.mark.parametrize("kind", [1, 2])
def test_reference_inviscid_wall_clears_prior_injection(tmp_path, kind):
    compile_probe(tmp_path, fixture()+SETUP+r'''
int main(){State s;initialise(s);ActualWall wall(s);wall.produce(s);
s.riemannBoundaryKind[1]=WALL_KIND;
// Deliberately retain the prior physical wall's total species injection.
s.gasSpecies.flux[1]=-.007;s.gasSpecies.flux[4]=-.003;
Real mass,mx,my,mz,energy;
ck(computeRiemannGasFaceFluxDevice<false>(s,1,mass,mx,my,mz,energy,false,true),"ordinary wall unavailable");
ck(mass==0&&s.gasSpecies.flux[1]==0&&s.gasSpecies.flux[4]==0,"inviscid PB subtraction retained injected species");
ck(s.gasSpecies.flux[0]==811&&s.gasSpecies.flux[5]==816,"unrelated face overwritten");
}
'''.replace("WALL_KIND",str(kind)))
