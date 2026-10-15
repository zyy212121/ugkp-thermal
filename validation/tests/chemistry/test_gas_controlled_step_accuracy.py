"""Additional controlled-step accuracy, alongside unchanged default-step tests.

These tests do not replace the original default-step H2O2/analytic assertions.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest
from test_gas_rates import SOURCE as RATE_SOURCE
from test_h2o2_reactor import SOURCE as H2O2_SOURCE,REFERENCE

ROOT=SOURCE_ROOT


def build(tmp_path,name,source):
    src=tmp_path/(name+'.cpp');exe=tmp_path/name;src.write_text(source)
    run=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I',str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    return exe


def test_analytic_reactor_with_requested_maximum_step(tmp_path):
    source=RATE_SOURCE.split('int main(){')[0].replace('double','double')+r'''
#include "common/chemistry/LocalStiffIntegrator.H"
int main(){
 Model a;ClosedReactorInput<REAL,2> b;b.speciesMass[0]=REAL(.028);b.totalMass=REAL(.028);b.volume=REAL(.001);b.internalEnergy=mixtureEnergy(b.speciesMass,REAL(700),a.t);
 ChemistryControls<REAL> c;c.maximumStep=REAL(.001);c.relativeTolerance=REAL(1e-3);c.absoluteMassFractionTolerance=REAL(1e-7);c.absoluteTemperatureTolerance=REAL(.01);c.thermo.relativeEnergyTolerance=REAL(1e-5);c.thermo.relativeTemperatureTolerance=REAL(1e-5);
 ClosedReactorResult<REAL,2> out;assert(advanceClosedReactor(b,REAL(.25),a.t,a.m,c,out));
 if(sizeof(REAL)==8)assert(near(out.speciesMass[0],REAL(.028)*std::exp(REAL(-.5)),REAL(3e-6)));
 else assert(std::abs(out.speciesMass[0]-REAL(.028)*std::exp(REAL(-.5)))<REAL(2e-5));
}
'''
    exe=build(tmp_path,'analytic',source.replace('REAL','double'))
    run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=20)
    assert run.returncode==0,run.stderr


def test_h2o2_requested_step_refinement_preserves_original_accuracy_bounds(tmp_path):
    source=H2O2_SOURCE.replace('#include <iomanip>','#include <iomanip>\n#include <cstdlib>').replace('int main(){','int main(int argc,char**argv){').replace('controls.relativeTolerance=tolerance;','controls.relativeTolerance=tolerance;controls.maximumStep=argc>1?std::atof(argv[1]):0;')
    exe=build(tmp_path,'h2o2',source)
    case=next(c for c in REFERENCE['cases'] if c['name']=='nitrogen_1200K_1atm')
    initial=case['states'][0]
    inputs=[case['volume'],*initial['speciesMass'],initial['temperature'],1e-6,len(case['output_times']),*case['output_times']]
    results=[]
    for dt in [1e-7,1e-8]:
        run=subprocess.run([str(exe),str(dt)],input=' '.join(map(str,inputs)),capture_output=True,text=True,timeout=90)
        assert run.returncode==0,run.stderr
        results.append([list(map(float,line.split())) for line in run.stdout.splitlines()])
    def error(rows):
        return max(max(abs(row[1]-ref['temperature'])/3000,max(abs(y-e) for y,e in zip(row[3:13],ref['massFractions']))) for row,ref in zip(rows,case['states']))
    assert error(results[1])<.15*error(results[0])
    assert error(results[1])<1e-5
    for row,ref in zip(results[1],case['states']):
        assert row[1]==pytest.approx(ref['temperature'],rel=3e-5,abs=3e-3)
        assert row[2]==pytest.approx(ref['pressure'],rel=4e-5,abs=.03)
        for y,expected in zip(row[3:13],ref['massFractions']):
            assert y>=0
            assert y==pytest.approx(expected,rel=4e-4,abs=2e-7)
        assert sum(row[3:13])==pytest.approx(1,abs=2e-12)
        assert abs(row[13])<1e-18
        assert abs(row[14])<1e-16
        assert abs(row[15])<1e-10+1e-10*abs(initial['internalEnergy'])
