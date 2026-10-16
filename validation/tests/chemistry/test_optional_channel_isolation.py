"""Unused chemistry/thermo channels cannot refuse finite requested results.

Serial host probes use real production headers; they do not claim CUDA execution.
"""
from pathlib import Path
import os
import subprocess

import pytest
from validation.tools.paths import SOURCE_ROOT

ROOT = Path(os.environ.get("UGKP_SOURCE_ROOT", str(SOURCE_ROOT)))

ZERO_RANK = r'''

#include "common/chemistry/GasRates.H"
#include "common/chemistry/GasJacobian.H"

#include <cassert>
#include <cmath>
#include <limits>
using namespace ugkwp;
struct Model {
 SpeciesThermoData<double> sp[2]; double coef[6]={1000,0,0,1000,0,0};
 double elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<double> rx[2]; GasStoichTerm<double> re[2],pr[2];
 GasColliderEfficiency<double> eff[1];
 SpeciesThermoView<double,2> t; GasMechanismView<double,2> m;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 for(int r=0;r<2;++r){rx[r].reactantOffset=r;rx[r].reactantCount=1;rx[r].productOffset=r;rx[r].productCount=1;rx[r].sourceIndex=r;rx[r].highRate.preExponential=2;re[r].species=0;re[r].coefficient=1;pr[r].species=1;pr[r].coefficient=1;}
 m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=2;m.products=pr;m.productTermCount=2;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 }
};
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<1e-12+r*std::abs(b);}


#include "common/chemistry/LocalStiffIntegrator.H"
int main() {
    for (int mode=0; mode<4; ++mode) {
        Model model;
        ugkp::diagnostics::DeviceContext diagnostics{};
        model.t.diagnostics=&diagnostics;
        ChemistryControls<double> controls;
        ClosedReactorInput<double,2> base;
        base.volume=.001; base.totalMass=.028; base.speciesMass[0]=.028;
        model.m.independentRank=0; model.m.stoichiometricBasis=nullptr;
        if (mode==0) model.t.coefficients=nullptr;
        if (mode==1) model.coef[1]=model.coef[4]=1e307;
        if (mode==2) model.coef[0]=model.coef[3]=universalGasConstant<double>()/.028;
        if (mode==3) model.m.reactants=nullptr;
        gasChemistryDetail::ReactorContext<double,2> context(base,model.t,model.m,controls,false);
        assert(context.prepare());
        bool available=false;
        double jacobian[4]={-777,-778,-779,-780};
        const auto status=context.evaluateJacobian(base.speciesMass,700.,jacobian,&available);
        if (mode<3) {
            assert(status && available);
            assert(!ugkp::diagnostics::hasFatal(&diagnostics));
        } else {
            assert(!status && !available && status.code==ChemistryCode::RateFailure);
        }
        assert(jacobian[0]==-777 && jacobian[1]==-778 && jacobian[2]==-779 && jacobian[3]==-780);
    }
}
'''

ENTROPY_CHANNELS = r'''

#include "common/chemistry/GasRates.H"
#include "common/chemistry/GasJacobian.H"

#include <cassert>
#include <cmath>
#include <limits>
using namespace ugkwp;
struct Model {
 SpeciesThermoData<double> sp[2]; double coef[6]={1000,0,0,1000,0,0};
 double elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<double> rx[2]; GasStoichTerm<double> re[2],pr[2];
 GasColliderEfficiency<double> eff[1];
 SpeciesThermoView<double,2> t; GasMechanismView<double,2> m;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 for(int r=0;r<2;++r){rx[r].reactantOffset=r;rx[r].reactantCount=1;rx[r].productOffset=r;rx[r].productCount=1;rx[r].sourceIndex=r;rx[r].highRate.preExponential=2;re[r].species=0;re[r].coefficient=1;pr[r].species=1;pr[r].coefficient=1;}
 m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=2;m.products=pr;m.productTermCount=2;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 }
};
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<1e-12+r*std::abs(b);}


int main() {
    for (int mode=0; mode<4; ++mode) {
        Model model;
        model.coef[2]=model.coef[5]=1;
        ugkp::diagnostics::DeviceContext diagnostics{};
        model.t.diagnostics=&diagnostics;
        const double temperature=mode==0?0.:700.;
        if (mode==1) model.sp[0].referenceTemperature=model.sp[1].referenceTemperature=0;
        if (mode==2) model.sp[0].hasEntropyReference=model.sp[1].hasEntropyReference=false;
        const double masses[2]={1,1};
        const double cp=speciesCp(0,temperature,model.t);
        const double energy=speciesE(0,temperature,model.t);
        const double total=mixtureEnergy(masses,temperature,model.t);
        const double expected=1+(1000-universalGasConstant<double>()/.028)*temperature;
        assert(cp==1000 && near(energy,expected) && near(total,2*expected));
        assert(!ugkp::diagnostics::hasFatal(&diagnostics));
        const double entropy=speciesEntropyStandard(0,temperature,model.t);
        if (mode<3) {
            assert(!std::isfinite(entropy));
            assert(ugkp::diagnostics::hasFatal(&diagnostics));
        } else {
            assert(std::isfinite(entropy));
            assert(!ugkp::diagnostics::hasFatal(&diagnostics));
        }
    }
    // Zero T remains an error when the actual EOS denominator consumes it.
    Model model;
    ugkp::diagnostics::DeviceContext diagnostics{};
    model.t.diagnostics=&diagnostics;
    const double fractions[2]={1,0}; double density=-777;
    const auto status=mixtureDensityFromPressureTemperature(101325.,0.,fractions,model.t,density);
    assert(status!=ThermoStatus::Success);
    assert(ugkp::diagnostics::hasFatal(&diagnostics));
    assert(density==-777);
}
'''

AUDIT_OVERFLOW = r'''

#include "common/operators/advanceGasChemistryKernel.cuh"
#include <cstdio>
#include <cassert>
#include <limits>
using namespace ugkwp;
template<class R,int N>struct State:GasStateView<R,double,int,double>{GasSpeciesState<R,N> gasSpecies;};
int main(){State<double,3> state;double rho=1,zero=0,energy=0,partial[3];
 SpeciesThermoData<double> species[3];double elements[6]={},coefficients[9]={};
 for(int i=0;i<3;++i){partial[i]=std::numeric_limits<double>::max()/2;species[i].coefficientOffset=i*3;species[i].molarMass=1;species[i].minTemperature=200;species[i].maxTemperature=4000;species[i].referencePressure=101325;coefficients[3*i]=universalGasConstant<double>();elements[2*i]=std::numeric_limits<double>::max();}
 auto& g=state.gasSpecies;g.thermo.species=species;g.thermo.coefficients=coefficients;g.thermo.coefficientCount=9;g.thermo.elementCount=2;g.thermo.elementComposition=elements;g.mode=GasMode::MixtureChemistry;g.mechanism.referencePressure=101325;g.rho=partial;g.densityClosureTolerance=0;
 ChemistryStatus recorded;int cell=0;g.chemistryStatus=&recorded;g.cellStatus=&cell;
 state.nCells=1;state.rho=&rho;state.rhoUx=state.rhoUy=state.rhoUz=&zero;state.rhoE=&energy;
 ugkp::diagnostics::DeviceContext diagnostics{};state.diagnostics=&diagnostics;
 energy=mixtureEnergy(partial,700.,g.thermo);bool available=false;
 const auto status=advanceGasChemistryCell(state,0,1e-6,1.,false,&available);
 assert(status && available && !ugkp::diagnostics::hasFatal(&diagnostics));
 for (double value: partial) assert(value==1); // Authorized finite-Y clipping.

 std::printf("code=%d thermo=%d available=%d energy=%.17g densities=%.17g,%.17g,%.17g\n",int(status.code),int(status.thermoCode),int(available),energy,partial[0],partial[1],partial[2]);
 std::printf("fatal=%d code=%u source=%u observed=%.17g\n",int(ugkp::diagnostics::hasFatal(&diagnostics)),unsigned(diagnostics.firstFatal.code),unsigned(diagnostics.firstFatal.sourceSiteId),diagnostics.firstFatal.observed);

 // An actually nonfinite input remains unavailable and fatal.
 diagnostics=ugkp::diagnostics::DeviceContext{};
 partial[0]=std::numeric_limits<double>::quiet_NaN();
 available=true;
 const auto invalid=advanceGasChemistryCell(state,0,1e-6,1.,false,&available);
 assert(!invalid && !available && ugkp::diagnostics::hasFatal(&diagnostics));
}
'''

@pytest.mark.parametrize("name,source", [
    ("zero_rank", ZERO_RANK),
    ("entropy_channels", ENTROPY_CHANNELS),
    ("audit_overflow", AUDIT_OVERFLOW),
], ids=["zero_rank", "entropy_channels", "audit_overflow"])
def test_unused_channels_preserve_finite_results(tmp_path, name, source):
    cpp = tmp_path / (name + ".cpp")
    executable = tmp_path / name
    cpp.write_text(source)
    build = subprocess.run(
        ["g++", "-std=c++17", "-O2", "-ffp-contract=off", "-Wall", "-Wextra",
         "-Werror", "-I", str(ROOT), str(cpp), "-o", str(executable)],
        capture_output=True, text=True, timeout=120,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=20)
    assert run.returncode == 0, run.stdout + run.stderr
