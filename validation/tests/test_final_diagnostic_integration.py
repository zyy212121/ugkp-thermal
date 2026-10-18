"""Host regressions for optional caloric context and gradient failure publication.

Runs production numerical headers; makes no CUDA runtime claim.
"""
import ast
import os
from pathlib import Path
import re
import subprocess
import pytest

ROOT = Path(os.environ.get('UGKP_SOURCE_ROOT', Path(__file__).resolve().parents[2]))
HERE = ROOT / 'validation/tests/gas_transport'
OVERLAY = os.environ.get('UGKP_REVIEW_OVERLAY')


def compile_run(tmp_path, source):
    cpp = tmp_path / 'probe.cpp'
    cpp.write_text(source)
    exe = tmp_path / 'probe'
    overlay_flags = []
    if OVERLAY:
        overlay = Path(OVERLAY)
        overlay_flags = ['-I' + str(overlay), '-I' + str(overlay / 'common'),
                         '-include', str(overlay / 'common/gasTransport/MixtureThermo.H')]
    built = subprocess.run(['g++', '-std=c++17', '-O0', *overlay_flags,
                            '-I' + str(ROOT), '-I' + str(HERE),
                            '-I' + str(ROOT / 'common'),
                            '-I' + str(ROOT / 'common/operators'),
                            '-I' + str(ROOT / 'common/gasNumerics'),
                            '-I' + str(ROOT / 'common/gasTransport'),
                            '-I' + str(ROOT / 'applications/CHMT'),
                            str(cpp), '-o', str(exe)], capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run([str(exe)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr


def mixture_fixture():
    source = (HERE / 'test_mixture_transport.py').read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'fixture')
    scope = {'re': re, 'VALIDATION_ROOT': ROOT / 'validation'}
    exec(ast.get_source_segment(source, function), scope)
    prelude = (HERE / 'gas_state_probe.cpp').read_text().split('void hashBytes')[0]
    return prelude + scope['fixture']()


@pytest.mark.parametrize('correction_overflow', [False, True])
@pytest.mark.parametrize('with_context', [False, True])
def test_caloric_nonfinite_intermediate_returns_failure(tmp_path, correction_overflow, with_context):
    source = r'''
#include "common/gasTransport/MixtureThermo.H"
#include <cassert>
using namespace ugkwp; using namespace ugkp::diagnostics;
struct Evaluator { ThermoStatus operator()(double,double& e,double& c)const {
 e=1e308; c=CAPACITY; return ThermoStatus::Success; } };
int main(){DeviceContext context{};double T=400;bool evaluated=true;
 ThermoInversionControls<double> controls;controls.maximumIterations=1;
 const auto status=invertCaloricEnergy(Evaluator{},TARGET,100.,3000.,controls,T,&evaluated,CONTEXT);
 assert(status==ThermoStatus::NonFiniteInput&&!evaluated&&T==400);
 if(HAS_CONTEXT){assert(hasFatal(&context));assert(context.firstFatal.sourceSiteId==SITE);
 assert(context.warnings[1].count==0);}
}
'''
    source = source.replace('CAPACITY', '1e-308' if correction_overflow else '1.')
    source = source.replace('TARGET', '0.' if correction_overflow else '-1e308')
    source = source.replace('HAS_CONTEXT', 'true' if with_context else 'false')
    source = source.replace('CONTEXT', '&context' if with_context else 'nullptr')
    source = source.replace('SITE', '30601' if correction_overflow else '30600')
    compile_run(tmp_path, source)


def test_chmt_optional_context_failure_reaches_owning_caller(tmp_path):
    compile_run(tmp_path, r'''
#include "materials/Thermodynamics.H"
#include <cassert>
int main(){chmt::PhysicsConfig p;chmt::SolidQ q;auto& t=p.condensed[0];
 t.rho=1;t.cp0=1e303;t.e0=1e308;t.Tmin=100;t.Tmax=3000;
 q.condensed[0]=1;q.energy=-1e308;double T=123;
 assert(!chmt::recoverSolid(q,p,T));assert(T==123);
}
''')


@pytest.mark.parametrize('component,site', [(0,30125),(1,30126),(2,30127),(3,30128),(4,30129),(5,30041)])
def test_gradient_face_failure_records_before_publication(tmp_path, component, site):
    compile_run(tmp_path, mixture_fixture() + r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext context{};s.diagnostics=&context;
 s.gradRhoX[0]=123;s.gradUxX[0]=124;s.gradUyX[0]=125;s.gradUzX[0]=126;s.gradPx[0]=127;s.gradTX[0]=128;
 Real* fields[]={s.rho,s.Ux,s.Uy,s.Uz,s.p,s.Tgas};
 fields[COMPONENT][0]=fields[COMPONENT][1]=NAN;
 threadIdx.x=0;computeGasPrimitiveGradientsKernel(&s);
 ck(ugkp::diagnostics::hasFatal(&context),"nonfinite face primitive lost");
 ck(context.firstFatal.sourceSiteId==SITE,"wrong producer site");
 ck(s.gradRhoX[0]==123&&s.gradUxX[0]==124&&s.gradUyX[0]==125&&s.gradUzX[0]==126&&s.gradPx[0]==127&&s.gradTX[0]==128,"failed face published gradients");
}
'''.replace('COMPONENT', str(component)).replace('SITE', str(site)))


def test_gradient_keeps_finite_uniform_large_density(tmp_path):
    compile_run(tmp_path, mixture_fixture() + r'''
int main(){State s;initialise(s);ugkp::diagnostics::DeviceContext context{};s.diagnostics=&context;
 s.rho[0]=s.rho[1]=1e300;s.gradRhoX[0]=123;threadIdx.x=0;
 computeGasPrimitiveGradientsKernel(&s);
 ck(!ugkp::diagnostics::hasFatal(&context),"finite face density was rejected");
 ck(s.gradRhoX[0]==0,"finite uniform gradient was not published");
}
''')
