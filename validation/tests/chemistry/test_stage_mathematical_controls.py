"""Fixed physical interval: domain-safe stage Newton and final residual status."""
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT
from validation.tests.chemistry.test_gas_rates import SOURCE

@pytest.mark.parametrize('mode',[1,2])
def test_stage_controls(tmp_path,mode):
    prefix=SOURCE.split('int main(){')[0]
    source=tmp_path/'stage.cpp'
    source.write_text(prefix+r'''
#include "common/chemistry/LocalStiffIntegrator.H"
#include <iostream>
int main(int argc,char**argv){const int mode=std::atoi(argv[1]);Model model;
    model.re[0].coefficient=model.pr[0].coefficient=mode==0?.5:2.;model.rx[0].highRate.preExponential=1000.;
    ugkp::diagnostics::DeviceContext diagnostics{};model.t.diagnostics=&diagnostics;
    ClosedReactorInput<double,2> base;base.speciesMass[0]=.028;base.totalMass=.028;base.volume=.001;
    base.internalEnergy=mixtureEnergy(base.speciesMass,700.,model.t);
    ChemistryControls<double> controls;controls.relativeTolerance=1e-7;controls.absoluteMassFractionTolerance=1e-13;controls.maximumNewtonIterations=16;controls.maximumLineSearchIterations=24;
    gasChemistryDetail::ReactorContext<double,2> ctx(base,model.t,model.m,controls);assert(ctx.prepare());
    const double h=mode==0?1.:(mode==1?10.:100.);double rhs[2]{},z[2]{},T=700,f[2],mass[2],residual[2];
    auto status=gasChemistryDetail::implicitStage(ctx,rhs,h,1.,z,T);assert(!diagnostics.fatalClaim);
    auto checked=ctx.evaluate(z,f,T,nullptr,mass);assert(checked);for(int i=0;i<model.m.independentRank;++i)residual[i]=z[i]-h*f[i];
    const double norm=gasChemistryDetail::coordinateNorm(ctx,residual,mass);
    std::cerr<<"mode="<<mode<<" code="<<int(status.code)<<" norm="<<norm<<" observed="<<status.observed<<"\n";
    assert(mass[0]>=0&&mass[1]>=0);
    if(mode!=2){assert(status.code==ChemistryCode::Success);assert(norm<=controls.newtonTolerance);}
    else{assert(status.code==ChemistryCode::NewtonFailure);assert(norm>controls.newtonTolerance);assert(std::abs(status.observed-norm)<=1e-10*std::max(1.,norm));}
}''')
    executable=tmp_path/'stage'
    subprocess.run(['g++','-std=c++17','-O2','-I',str(SOURCE_ROOT),str(source),'-o',str(executable)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(executable),str(mode)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
