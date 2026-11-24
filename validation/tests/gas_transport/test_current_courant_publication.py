from validation.tools.paths import SOURCE_ROOT
from test_current_gas_courant import launches, CUDA
from test_legacy_ale_transport import fixture
from test_mixture_state import compile_probe


def test_actual_courant_helper_publishes_all_computed_maxima(tmp_path):
    helper = launches((SOURCE_ROOT / "common/GpuGasCourant.cuh").read_text())
    compile_probe(tmp_path, fixture() + CUDA + helper + r"""
#include <cassert>
int main(){
 Host state;initialise(state);
 const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
 for(double a:{0.,.2,2.,nan,inf,-inf}){
  int calls=0;GasCourantMaxima result;result.maximum=-777;
  auto reduce=[&](const double*,int,void*,double& value){++calls;value=a;return 0;};
  assert(computeGasCurrentCourant(&state,.1,0.,32,32,nullptr,result,reduce)==0&&calls==1);
  assert((std::isnan(a)&&std::isnan(result.convective))||a==result.convective);
  const double expected=a;
  assert((std::isnan(expected)&&std::isnan(result.maximum))||expected==result.maximum);
 }
 GasCourantMaxima result;result.maximum=-777;
 auto fail=[](const double*,int,void*,double&){return 1;};
 assert(computeGasCurrentCourant(&state,.1,0.,32,32,nullptr,result,fail)!=0&&result.maximum==-777);
 failLaunch=launches+1;int calls=0;
 auto reduce=[&](const double*,int,void*,double&){++calls;return 0;};
 assert(computeGasCurrentCourant(&state,.1,0.,32,32,nullptr,result,reduce)!=0&&calls==0&&result.maximum==-777);
}
""")
