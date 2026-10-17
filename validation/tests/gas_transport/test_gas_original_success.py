"""Exact original-header comparisons at successful gas kernel boundaries.

Host execution, nontrapping floating point. These are actual kernel/source
bodies with revision-correct dependencies, not whole-native GPU witnesses.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import io
import re
import subprocess
import tarfile
import pytest
from test_authoritative_energy_sources import source_fixture
from test_legacy_ale_transport import fixture
from test_mixture_transport import fixture as mixture_fixture

ROOT = SOURCE_ROOT
HERE = (VALIDATION_ROOT / 'tests/gas_transport')
ORIGINAL = 'd6110c02745a8ad726208a426c4bb3a28233c1fd'

from test_native_original_success import original_native_root as original_root

def build_run(folder,label,root,body):
    pre=((VALIDATION_ROOT / 'tests/gas_transport/gas_state_probe.cpp')).read_text().split('void hashBytes')[0]
    if not (root/"common/Diagnostics.H").exists():
        body=re.sub(r"^s\.diagnostics=.*;\n", "", body, flags=re.MULTILINE)
    src=folder/(label+'.cpp');exe=folder/label;src.write_text(pre+body)
    cmd=['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(root/'common'),'-I'+str(root/'common/gasNumerics'),'-I'+str(HERE),str(src),'-o',str(exe)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    (folder/(label+'_build.log')).write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    (folder/(label+'.log')).write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    return result.stdout

PRINT=r'''
template<class S>void output(const S&s){
 for(int c=0;c<s.nCells;++c){
  std::printf("%d ",s.gasSpecies.cellStatus[c]);
  for(double v:{double(s.rho[c]),double(s.rhoUx[c]),double(s.rhoUy[c]),double(s.rhoUz[c]),double(s.rhoE[c]),double(s.Ux[c]),double(s.Uy[c]),double(s.Uz[c]),double(s.p[c]),double(s.Tgas[c]),double(s.rhoK[c]),double(s.rhoOmega[c]),double(s.gasFluxPositivityScale[c]),double(s.gasDiffusionNumber[c]),double(s.sstSourceNumber[c])})std::printf("%a ",v);
 }
 for(int f=0;f<s.nFaces;++f)std::printf("%d %a ",s.gasSpecies.faceStatus[f],double(s.gasPhiRhoE[f]));
 std::puts("");
}
'''

@pytest.mark.parametrize('source',['drag','heat','volume','recovery'])
def test_exact_bounded_energy_fallbacks(tmp_path,original_root,source):
    action={'gravity':'s.gravityX=1;applyGasGravityCell<true>(s,0,1.);',
            'drag':'drag<0>(&s,.25,0);','heat':'s.gasMu=1e-5;applyEulerianParticleMaterialHeatKernel(&s,.25);',
            'volume':'s.TgasMin=0;applyGasVolumeFractionSourceKernel(&s,.25);',
            'recovery':'recoverGasPrimitivesKernel(&s);'}[source]


    values='{1e300,std::nextafter(1e300,INFINITY),1.1e300,1e299}' if source=='volume' else '{-1.1e300,-1e300,std::nextafter(-1e300,0.),std::nextafter(1e300,0.),1e300,1.1e300,2e300}'
    body=source_fixture()+PRINT+r'''
int main(){for(double value:VALUES){DeviceState s;setup(s);s.cellPlaneCount[0]=0;s.rhoE[0]=value;ACTION
output(s);std::printf("particle %a %a %a %a %a\n",double(s.momRhoUPx[0]),double(s.momRhoUPy[0]),double(s.momRhoUPz[0]),double(s.momRhoEP[0]),double(s.momRhoHpP[0]));}}
'''.replace('VALUES',values).replace('ACTION',action)
    expected=build_run(tmp_path,'original',original_root,body)
    actual=build_run(tmp_path,'corrected',ROOT,body)
    assert actual==expected

def test_exact_ordinary_finite_gravity_arithmetic(tmp_path,original_root):
    body=source_fixture()+PRINT+r'''
int main(){for(double value:{-100.,0.,100.,1e299}){DeviceState s;setup(s);s.rhoE[0]=value;
s.gravityX=1;applyGasGravityCell<true>(s,0,1.);output(s);}}
'''
    assert build_run(tmp_path,'current-gravity',ROOT,body)==build_run(tmp_path,'original-gravity',original_root,body)

@pytest.mark.parametrize('kernel',['positivity','convective','diffusion','sst','sst_update','divergence'])
def test_exact_skipped_face_and_zero_trip_loops(tmp_path,original_root,kernel):
    action={'positivity':'computeGasFluxPositivityScaleKernel(&s,.001);',
            'convective':'computeGasConvectiveCourantByCellKernel(&s,.001);',
            'diffusion':'computeGasDiffusionNumberKernel(&s,.001,Real(.5));',
            'sst':'computeSstStabilityNumberKernel(&s,.001,Real(.5));',
            'sst_update':'applySstFluxAndSourceKernel(&s,.001);',
            'divergence':'applyGasFluxDivergenceByCellKernel(&s,.001);'}[kernel]
    body=fixture()+PRINT+r'''
int main(){for(int mode:{0,1,2,3}){State s;initialise(s);s.sstConfigured=1;
 s.gasPhiRho[0]=.125;s.gasPhiRho[1]=.0625;s.gasPhiRhoE[0]=2;s.gasPhiRhoE[1]=3;s.gasMu=.001;
 s.sstPhiRhoK[0]=.25;s.sstPhiRhoOmega[0]=.5;
 if(mode<2){s.cellPlaneCount[0]=4;s.cellFaceId[0]=0;s.cellFaceId[1]=mode?-1:3;s.cellFaceId[2]=1;s.cellFaceId[3]=mode?3:-1;}
 if(mode==2)s.cellPlaneCount[0]=-1;
 if(mode==3){s.cellPlaneCount[0]=1;s.cellFaceId[0]=2;}
 ACTION output(s);}}
'''.replace('ACTION',action)
    expected=build_run(tmp_path,'original',original_root,body)
    actual=build_run(tmp_path,'corrected',ROOT,body)
    assert actual==expected

def test_exact_courant_fallback_arithmetic(tmp_path,original_root):
    body=fixture()+PRINT+r'''
int main(){for(double value:{1e301,double(INFINITY),double(NAN),2.}){State s;initialise(s);s.cellPlaneCount[0]=1;s.gasPhiRhoE[0]=Real(value);computeGasConvectiveCourantByCellKernel(&s,1e-300);output(s);}
 for(int mode:{0,1,2,3}){State s;initialise(s);s.sstConfigured=mode==3;s.gasMu=1;
 if(mode==0)s.faceOwner[0]=-1;
 if(mode==1)s.faceNeighbour[0]=s.nCells;
 if(mode<2)computeGasCourantFieldKernel(&s,1.);
 if(mode==2)computeGasDiffusionNumberKernel(&s,1e300,Real(.5));
 if(mode==3)computeSstStabilityNumberKernel(&s,1e300,Real(.5));
 output(s);}}
'''
    expected=build_run(tmp_path,'original',original_root,body)
    strict=body.replace('computeGasConvectiveCourantByCellKernel(&s,1e-300)', 'computeGasConvectiveCourantByCellKernel(&s,1e-300,true)').replace('computeGasCourantFieldKernel(&s,1.)','computeGasCourantFieldKernel(&s,1.,true)').replace('computeGasDiffusionNumberKernel(&s,1e300,Real(.5))','computeGasDiffusionNumberKernel(&s,1e300,Real(.5),true)').replace('computeSstStabilityNumberKernel(&s,1e300,Real(.5))','computeSstStabilityNumberKernel(&s,1e300,Real(.5),true,true)')
    actual=build_run(tmp_path,'corrected',ROOT,strict)
    rows=expected.splitlines(keepends=True)
    for i in [6,7]:
        assert rows[i].startswith('0 ')
        rows[i]='6 '+rows[i][2:]
    assert actual==''.join(rows)

def test_exact_species_positivity_skips(tmp_path,original_root):
    body=mixture_fixture()+PRINT+r'''
int main(){for(int count:{4,-1}){State s;initialise(s);threadIdx.x=0;s.cellPlaneCount[0]=count;
s.cellFaceId[0]=0;s.cellFaceId[1]=-1;s.cellFaceId[2]=1;s.cellFaceId[3]=s.nFaces;
s.gasSpecies.flux[0]=2;s.gasSpecies.flux[3]=-2;computeGasFluxPositivityScaleKernel(&s,1.);output(s);
for(int k=0;k<4;++k)std::printf("%a ",double(s.gasSpecies.positivityScale[k]));std::puts("");}}
'''
    expected=build_run(tmp_path,'original',original_root,body)
    actual=build_run(tmp_path,'corrected',ROOT,body)
    assert actual==expected

@pytest.mark.parametrize('field',['energy','momentum','sst'])
def test_nonfinite_blend_records_without_publication(tmp_path,field):
    set_field={'energy':'s.rhoE[0]=s.rhoENext[0]=std::numeric_limits<double>::infinity();',
               'momentum':'s.rhoUx[0]=s.rhoUxNext[0]=std::numeric_limits<Real>::infinity();',
               'sst':'s.sstConfigured=1;s.rhoK[0]=s.rhoKInitial[0]=std::numeric_limits<Real>::infinity();'}[field]
    body=source_fixture()+PRINT+r'''
int main(){DeviceState s;setup(s);saveGasConservativeStateKernel(&s);SET_FIELD
ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;
const auto omega=s.rhoOmega[0];blendGasConservativeStateKernel(&s,.5,.5);
ck(ugkp::diagnostics::hasFatal(&diagnostics),"nonfinite blend not diagnosed");
ck(s.rhoOmega[0]==omega,"failed blend published later fields");}
'''.replace('SET_FIELD',set_field)
    build_run(tmp_path,'corrected',ROOT,body)

@pytest.mark.parametrize('field',['energy','momentum'])
def test_nonfinite_fixed_divergence_records_without_publication(tmp_path,field):
    set_field={'energy':'s.rhoE[0]=std::numeric_limits<double>::infinity();',
               'momentum':'s.rhoUx[0]=std::numeric_limits<Real>::infinity();'}[field]
    body=source_fixture()+PRINT+r'''
int main(){DeviceState s;setup(s);SET_FIELD
ugkp::diagnostics::DeviceContext diagnostics{};s.diagnostics=&diagnostics;
const auto density=s.rho[0];applyGasFluxDivergenceByCellKernel(&s,.01);
ck(ugkp::diagnostics::hasFatal(&diagnostics),"nonfinite divergence not diagnosed");
ck(s.rho[0]==density,"failed divergence published state");}
'''.replace('SET_FIELD',set_field)
    build_run(tmp_path,'corrected',ROOT,body)
