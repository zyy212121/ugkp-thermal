"""Pure common C++ reactor vs pinned, independently generated Cantera histories.

This is CPU math evidence, not native gasUGKP/CHMT or CUDA execution evidence.
"""
from pathlib import Path
import json
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
REFERENCE=json.loads((ROOT/'common/chemistry/mechanisms/h2o2.cantera-reactor-reference.json').read_text())
SOURCE=r'''
#include "common/chemistry/LocalStiffIntegrator.H"
#include "common/chemistry/mechanisms/GeneratedH2O2.H"
#include <iostream>
#include <iomanip>
int main(){using namespace ugkwp;GeneratedH2O2<double> table;const auto thermo=table.thermoView();const auto mechanism=table.mechanismView();
ClosedReactorInput<double,10> base;std::cin>>base.volume;for(double& x:base.speciesMass){std::cin>>x;base.totalMass+=x;}double T,tolerance;int count;std::cin>>T>>tolerance>>count;base.internalEnergy=mixtureEnergy(base.speciesMass,T,thermo);ChemistryControls<double> controls;controls.relativeTolerance=tolerance;controls.absoluteMassFractionTolerance=1e-14;controls.absoluteTemperatureTolerance=1e-6;controls.maximumSteps=100000;controls.maximumRejectedSteps=1000;double time=0;std::cout<<std::setprecision(17);
for(int k=0;k<count;++k){double target;std::cin>>target;ClosedReactorResult<double,10> out;ChemistryAudit<double,10> audit;auto status=advanceClosedReactor(base,target-time,thermo,mechanism,controls,out,audit);
if(!status){std::cerr<<"code="<<int(status.code)<<" reaction="<<status.reaction<<" target="<<target<<" attempted="<<status.attemptedSteps<<" rejected="<<status.rejectedSteps<<" nonlinear="<<status.nonlinearIterations<<'\n';return 1;}
double mol=0;for(int s=0;s<10;++s)mol+=out.speciesMass[s]/thermo.species[s].molarMass;double p=mol*universalGasConstant<double>()*out.temperature/base.volume;
std::cout<<target<<' '<<out.temperature<<' '<<p;for(double mass:out.speciesMass)std::cout<<' '<<mass/base.totalMass;std::cout<<' '<<audit.massResidual<<' '<<audit.maximumElementResidual<<' '<<audit.energyResidual<<' '<<audit.acceptedSteps<<' '<<audit.rejectedSteps<<'\n';
for(int s=0;s<10;++s)base.speciesMass[s]=out.speciesMass[s];
time=target;}
}
'''
@pytest.fixture(scope='module')
def runner(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('h2o2-reactor');src=tmp/'run.cpp';src.write_text(SOURCE);exe=tmp/'run'
    build=subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-pedantic','-I',str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    return exe

def integrate(runner,case,tolerance):
    initial=case['states'][0]
    inputs=[case['volume'],*initial['speciesMass'],initial['temperature'],tolerance,len(case['output_times']),*case['output_times']]
    run=subprocess.run([str(runner)],input=' '.join(map(str,inputs)),capture_output=True,text=True,timeout=180)
    assert run.returncode==0,f"{case['name']}: {run.stderr}"
    return [list(map(float,line.split())) for line in run.stdout.splitlines()]

@pytest.mark.parametrize('case',REFERENCE['cases'],ids=lambda c:c['name'])
def test_full_h2o2_histories_and_invariants(runner,case):
    rows=integrate(runner,case,1e-6)
    for row,ref in zip(rows,case['states']):
        assert row[0]==ref['time']
        # Includes published NASA7 midpoint energy discontinuity (~0.00012 K)
        # in the independent reference; our own U invariant remains strict.
        assert row[1]==pytest.approx(ref['temperature'],rel=3e-5,abs=3e-3)
        assert row[2]==pytest.approx(ref['pressure'],rel=4e-5,abs=.03)
        for y,expected in zip(row[3:13],ref['massFractions']):
            assert y>=0
            assert y==pytest.approx(expected,rel=4e-4,abs=2e-7)
        assert sum(row[3:13])==pytest.approx(1,abs=2e-12)
        assert abs(row[13])<1e-18
        assert abs(row[14])<1e-16
        assert abs(row[15])<1e-10+1e-10*abs(case['states'][0]['internalEnergy'])

def test_chemical_tolerance_refinement_reduces_history_error(runner):
    case=next(c for c in REFERENCE['cases'] if c['name']=='nitrogen_1200K_1atm')
    coarse=integrate(runner,case,1e-4)
    fine=integrate(runner,case,1e-7)
    def error(rows):
        return max(max(abs(row[1]-ref['temperature'])/3000,
                       max(abs(y-e) for y,e in zip(row[3:13],ref['massFractions'])))
                   for row,ref in zip(rows,case['states']))
    assert error(fine)<.15*error(coarse)
    assert error(fine)<1e-5
