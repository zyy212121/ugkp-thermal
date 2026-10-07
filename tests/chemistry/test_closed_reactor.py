"""Closed reactor invariants, stiff stability and transactional behavior."""
from pathlib import Path
import subprocess
import pytest
from test_gas_rates import SOURCE as RATE_SOURCE
ROOT=Path(__file__).resolve().parents[2]
MODEL=RATE_SOURCE.split('int main(){')[0]
SOURCE=MODEL+r'''
#include "common/chemistry/LocalStiffIntegrator.H"
#include <cstring>
int main(){
 ChemistryControls<double> invalid;invalid.minimumStep=1;invalid.maximumStep=.1;assert(!validateChemistryControls(invalid));invalid=ChemistryControls<double>();invalid.thermo.maximumIterations=0;assert(!validateChemistryControls(invalid));
 Model a;ClosedReactorInput<double,2> base;base.speciesMass[0]=.028;base.totalMass=.028;base.volume=.001;base.internalEnergy=mixtureEnergy(base.speciesMass,700.,a.t);
 ChemistryControls<double> controls;controls.relativeTolerance=1e-7;controls.absoluteMassFractionTolerance=1e-13;controls.absoluteTemperatureTolerance=1e-6;
 ClosedReactorResult<double,2> result;ChemistryAudit<double,2> audit;
 auto st=advanceClosedReactor(base,.25,a.t,a.m,controls,result,audit);assert(st);
 assert(near(result.speciesMass[0],.028*std::exp(-.5),3e-6));assert(near(result.speciesMass[0]+result.speciesMass[1],base.totalMass,1e-12));assert(near(result.temperature,700.,1e-11));assert(audit.integratedTime==.25);assert(audit.acceptedSteps>0);
 assert(near(mixtureEnergy(result.speciesMass,result.temperature,a.t),base.internalEnergy,1e-11));
 // Stiff rate and exact-zero product: donor remains nonnegative, no clipping.
 a.rx[0].highRate.preExponential=1e8;st=advanceClosedReactor(base,.001,a.t,a.m,controls,result,audit);assert(st);assert(result.speciesMass[0]>=0 && result.speciesMass[0]<1e-10);assert(near(result.speciesMass[1],.028,1e-7));
 // Exothermic formation-energy change drives temperature while U stays fixed.
 a.rx[0].highRate.preExponential=20;a.coef[5]=-5e5;st=advanceClosedReactor(base,.05,a.t,a.m,controls,result,audit);assert(st);assert(result.temperature>1000);assert(near(result.speciesMass[0],.028*std::exp(-1.),3e-6));assert(near(mixtureEnergy(result.speciesMass,result.temperature,a.t),base.internalEnergy,1e-11));
 // Fixed-energy reduced Jacobian must include thermal feedback.
 a.rx[0].highRate.temperatureExponent=.5;double z[2]={.1,0},f[2],j[4],T=700;assert(evaluateConservativeReactor(base,z,a.t,a.m,controls,f,T,j));
 double zp[2]={.100001,0},zm[2]={.099999,0},fp[2],fm[2],Tp=700,Tm=700;assert(evaluateConservativeReactor(base,zp,a.t,a.m,controls,fp,Tp,nullptr));assert(evaluateConservativeReactor(base,zm,a.t,a.m,controls,fm,Tm,nullptr));assert(near(j[0],(fp[0]-fm[0])/2e-6,1e-6));
 // Whole-call failure cannot publish partial state OR accepted diagnostics.
 const auto saved=result;const auto savedAudit=audit;controls.maximumSteps=1;st=advanceClosedReactor(base,1.,a.t,a.m,controls,result,audit);assert(!st);assert(std::memcmp(&result,&saved,sizeof result)==0);assert(std::memcmp(&audit,&savedAudit,sizeof audit)==0);
 controls.maximumSteps=10000;base.totalMass*=2;st=advanceClosedReactor(base,.1,a.t,a.m,controls,result,audit);assert(!st);assert(std::memcmp(&result,&saved,sizeof result)==0);assert(std::memcmp(&audit,&savedAudit,sizeof audit)==0);
 base.totalMass=.028;st=advanceClosedReactor(base,0.,a.t,a.m,controls,result,audit);assert(st);assert(result.speciesMass[0]==base.speciesMass[0] && result.speciesMass[1]==0 && audit.acceptedSteps==0);
}
'''
def test_conservative_stiff_reactor(tmp_path):
 src=tmp_path/'reactor.cpp';src.write_text(SOURCE);exe=tmp_path/'reactor'
 built=subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-pedantic','-I',str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
 assert built.returncode==0,built.stderr
 run=subprocess.run([str(exe)],capture_output=True,text=True)
 assert run.returncode==0,run.stderr


def test_cuda_common_reactor_compiles_when_toolchain_available(tmp_path):
    import shutil
    nvcc=shutil.which('nvcc')
    if nvcc is None:
        pytest.skip('nvcc unavailable: no CUDA compilation or device execution claimed')
    src=tmp_path/'reactor.cu'
    src.write_text(r'''
#include "common/chemistry/LocalStiffIntegrator.H"
__global__ void reactor(const ugkwp::ClosedReactorInput<double,10>* b,
 ugkwp::SpeciesThermoView<double,10> t,ugkwp::GasMechanismView<double,10> m,
 ugkwp::ChemistryControls<double> c,ugkwp::ClosedReactorResult<double,10>* r,
 ugkwp::ChemistryAudit<double,10>* a,ugkwp::ChemistryStatus* status) {
 *status=ugkwp::advanceClosedReactor(*b,1e-6,t,m,c,*r,*a);
}
''')
    run=subprocess.run([nvcc,'-std=c++14','-I',str(ROOT),'-c',str(src),'-o',str(tmp_path/'reactor.o')],capture_output=True,text=True)
    assert run.returncode==0,run.stderr


def test_float_reactor_uses_explicit_precision_tolerances(tmp_path):
    src=tmp_path/'float.cpp'
    src.write_text(MODEL.replace('double','float')+r'''
#include "common/chemistry/LocalStiffIntegrator.H"
int main(){Model a;ClosedReactorInput<float,2> b;b.speciesMass[0]=.028f;b.totalMass=.028f;b.volume=.001f;b.internalEnergy=mixtureEnergy(b.speciesMass,700.f,a.t);
ChemistryControls<float> c;c.relativeTolerance=1e-3f;c.absoluteMassFractionTolerance=1e-7f;c.absoluteTemperatureTolerance=.01f;c.relativeConservationTolerance=1e-5f;c.thermo.relativeEnergyTolerance=1e-5f;c.thermo.absoluteTemperatureTolerance=.01f;c.thermo.relativeTemperatureTolerance=1e-5f;
ClosedReactorResult<float,2> r;ChemistryAudit<float,2> audit;auto status=advanceClosedReactor(b,.25f,a.t,a.m,c,r,audit);assert(status);assert(std::abs(r.speciesMass[0]-.028f*std::exp(-.5f))<2e-5f);assert(std::abs(r.temperature-700.f)<.02f);}
''')
    run=subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT),str(src),'-o',str(tmp_path/'float')],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    run=subprocess.run([str(tmp_path/'float')],capture_output=True,text=True)
    assert run.returncode==0,run.stderr


@pytest.mark.parametrize('real',['double','float'])
def test_inactive_fast_mode_does_not_block_independent_slow_reaction(tmp_path,real):
    """A dimensionally large independent row cannot invalidate a unit pivot."""
    src=tmp_path/'empty_fast_mode.cpp'
    src.write_text(r'''
#include "common/chemistry/LocalStiffIntegrator.H"
#include <cassert>
#include <cmath>
using namespace ugkwp;
int main(){
 SpeciesThermoData<double> species[4];double coefficients[12],atoms[4]={1,1,1,1};
 double basis[8]={-1,0,1,0,0,-1,0,1};
 for(int s=0;s<4;++s){species[s].coefficientOffset=3*s;species[s].molarMass=.028;species[s].minTemperature=200;species[s].maxTemperature=4000;species[s].referencePressure=101325;coefficients[3*s]=1000;coefficients[3*s+1]=coefficients[3*s+2]=0;}
 SpeciesThermoView<double,4> t;t.species=species;t.coefficients=coefficients;t.coefficientCount=12;t.elementCount=1;t.elementComposition=atoms;
 GasReactionData<double> reactions[2];GasStoichTerm<double> reactants[2]={{0,1},{2,1}},products[2]={{1,1},{3,1}};
 for(int r=0;r<2;++r){reactions[r].reactantOffset=reactions[r].productOffset=r;reactions[r].reactantCount=reactions[r].productCount=1;reactions[r].highRate.preExponential=1;}
 GasMechanismView<double,4> m;m.reactions=reactions;m.reactionCount=2;m.reactants=reactants;m.reactantTermCount=2;m.products=products;m.productTermCount=2;m.referencePressure=101325;m.stoichiometricBasis=basis;m.independentRank=2;
 ClosedReactorInput<double,4> base;base.speciesMass[2]=.028;base.totalMass=.028;base.volume=.001;base.internalEnergy=mixtureEnergy(base.speciesMass,700.,t);
 ChemistryControls<double> controls;controls.relativeTolerance=1e-8;ClosedReactorResult<double,4> out;ChemistryAudit<double,4> audit;
 const double rates[4]={1,1e12,1e20,1e100};int baselineAttempts=0;
 for(double rate:rates){reactions[0].highRate.preExponential=rate;auto status=advanceClosedReactor(base,1.,t,m,controls,out,audit);assert(status);assert(out.speciesMass[0]==0 && out.speciesMass[1]==0);assert(std::abs(out.speciesMass[2]/base.totalMass-std::exp(-1.))<3e-6);assert(std::abs(out.temperature-700.)<1e-8);assert(audit.integratedTime==1.);if(rate==1)baselineAttempts=status.attemptedSteps;else assert(status.attemptedSteps==baselineAttempts);}
 // Row-relative pivot testing must still reject an actually singular system.
 double singular[16]={1,2,0,0,2,4,0,0},rhs[4]={1,2,0,0};assert((!gasChemistryDetail::solveDense<double,4>(2,singular,rhs)));
}
''')
    if real=='float':
        source=src.read_text().replace('double','float').replace('1e100','1e30')
        source=source.replace('controls.relativeTolerance=1e-8;', 'controls.relativeTolerance=1e-3f;controls.absoluteMassFractionTolerance=1e-7f;controls.absoluteTemperatureTolerance=.01f;controls.relativeConservationTolerance=1e-5f;controls.thermo.relativeEnergyTolerance=1e-5f;controls.thermo.absoluteTemperatureTolerance=.01f;controls.thermo.relativeTemperatureTolerance=1e-5f;')
        source=source.replace(',700.,',',700.f,').replace('(base,1.,','(base,1.f,').replace('<3e-6','<1e-3').replace('<1e-8','<.01')
        src.write_text(source)
    exe=tmp_path/'empty_fast_mode'
    built=subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-pedantic','-I',str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
