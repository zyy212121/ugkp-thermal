"""A missing mixture operator must not silently select another flux."""
from test_mixture_state import compile_probe
from test_mixture_transport import fixture


def test_missing_mixture_operator_reports_without_substitution(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;
 for(int scheme=1;scheme<=9;++scheme)for(int reconstruction:{0,1,2}){
  if((scheme==1 || scheme==2 || scheme==4) && reconstruction<=1)continue;
  ugkp::diagnostics::reset(diagnostics,1,1);s.gasFluxScheme=scheme;s.gasReconstruction=reconstruction;
  for(int f=0;f<3;++f)s.gasSpecies.faceStatus[f]=0;
  Real out[5]={};s.gasSpecies.flux[0]=123;
  ck(!computeRiemannGasFaceFluxDevice<false>(s,0,out[0],out[1],out[2],out[3],out[4]),"missing mixture operator substituted");
  ck(diagnostics.firstFatal.code==ugkp::diagnostics::Code::MissingOperator,"missing selected operator diagnostic");
  ck(s.gasSpecies.flux[0]==123,"substitute flux published");
 }
}
''')
