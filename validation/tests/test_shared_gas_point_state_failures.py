"""Actual pointState body, real common EOS/calorics, native OpenFOAM fatal boundary."""
from pathlib import Path
import os
import subprocess
import shlex
import pytest
ROOT=Path(os.environ.get('UGKP_POINT_STATE_ROOT',Path(__file__).resolve().parents[2]))
COMMON_ROOT=Path(os.environ.get('UGKP_COMMON_ROOT',ROOT))
def toolchain_prefix():
    if os.environ.get('WM_PROJECT_DIR') and os.environ.get('FOAM_LIBBIN'):
        return ''
    script=Path(os.environ.get('UGKP_TEST_ENV_SCRIPT','/workspace/shared/toolchains/env.sh'))
    if script.is_file():
        return 'source '+shlex.quote(str(script))+' >/dev/null 2>&1; '
    pytest.skip('source OpenFOAM or set UGKP_TEST_ENV_SCRIPT to its environment script')

@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    prefix=toolchain_prefix()
    tmp=tmp_path_factory.mktemp('point_state')
    text=(ROOT/'applications/gasUGKP/gpu/SharedGasModelInput.H').read_text()
    method=text[text.index('    void pointState('):text.index('    void initialiseIdentity(')]
    code=r'''
#include "error.H"
#include "vector.H"
#include "gasTransport/MixtureThermo.H"
#include <cmath>
#include <limits>
#include <cassert>
#include <string>
namespace ugkwp{constexpr int compiledGasSpecies=2;}
namespace Foam {
struct Model {
 ugkwp::SpeciesThermoData<double> species[2];double coefficients[6]={1000,0,0,1000,0,0};
 Model(){for(int k=0;k<2;++k){species[k].molarMass=.028;species[k].coefficientOffset=3*k;species[k].minTemperature=100;species[k].maxTemperature=3000;}}
 template<int N>ugkwp::SpeciesThermoView<double,N>thermoView()const{
 ugkwp::SpeciesThermoView<double,N>v;v.species=species;v.coefficients=coefficients;v.coefficientCount=6;return v;}
};
struct Fixture{Model model_;
METHOD
};}
int main(int argc,char**argv){using namespace Foam;Fixture fixture;
 double y[2]={.5,.5};scalar p=101325,T=300,rho=-17,energy=-19;vector U(1,2,3);
 std::string mode=argc>1?argv[1]:"valid";
 const double nan=std::numeric_limits<double>::quiet_NaN();
 if(mode=="nan_pressure")p=nan;
 if(mode=="zero_temperature")T=0;
 if(mode=="negative_temperature")T=-1;
 if(mode=="nan_velocity")U.x()=nan;
 if(mode=="overflow_velocity")U.x()=1e308;
 if(mode=="nan_species")y[0]=nan;
 if(mode=="nonfinite_caloric")fixture.model_.coefficients[2]=nan;
 fixture.pointState(p,T,U,y,rho,energy);
 if(mode=="valid"){assert(std::isfinite(rho)&&std::isfinite(energy)&&rho>0);}
 return 0;
}
'''.replace('METHOD',method)
    source=tmp/'probe.C';source.write_text(code);binary=tmp/'probe'
    command=prefix+f'''g++ -std=c++17 -DWM_DP -DWM_LABEL_SIZE=32 \\
      -I{shlex.quote(str(COMMON_ROOT/'common'))} \\
      -I"$WM_PROJECT_DIR/src/OpenFOAM/lnInclude" -I"$WM_PROJECT_DIR/src/OSspecific/POSIX/lnInclude" \\
      {shlex.quote(str(source))} -L"$FOAM_LIBBIN" -L"$FOAM_LIBBIN/$FOAM_MPI" \\
      -lOpenFOAM -lPstream -o {shlex.quote(str(binary))}'''
    result=subprocess.run(['bash','-c',command],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return binary,prefix

@pytest.mark.parametrize('mode',['nan_pressure','zero_temperature','negative_temperature','nan_velocity','overflow_velocity','nan_species','nonfinite_caloric'])
def test_physical_failure_exits_before_publication(probe,mode):
    binary,prefix=probe
    result=subprocess.run(['bash','-c',prefix+shlex.quote(str(binary))+' '+shlex.quote(mode)],capture_output=True,text=True)
    assert result.returncode==1,result.stdout+result.stderr
    assert 'FOAM FATAL ERROR' in result.stdout+result.stderr


def test_valid_state_preserved(probe):
    binary,prefix=probe
    result=subprocess.run(['bash','-c',prefix+shlex.quote(str(binary))+' valid'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
