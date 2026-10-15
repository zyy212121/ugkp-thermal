"""B1 review counterexamples: actual CHMT success and fully backed metadata.

Every coefficient/index/offset probe stays within real allocated arrays.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import io
import subprocess
import tarfile
import pytest

ROOT = SOURCE_ROOT
BASE = "82f9c016704c9822b16b87cdeeb11f5bbd44d06f"

MATERIAL = r'''
#include "materials/MaterialCaloric.H"
#include <cassert>
#include <cstdio>
#include <limits>
struct Counted {
 mutable int count=0;
 ugkwp::ThermoStatus operator()(double T,double&e,double&c)const{++count;e=T;c=1;return ugkwp::ThermoStatus::Success;}
};
int main(){
 using namespace chmt;PhysicsConfig p;
 for(auto&t:p.condensed){t.rho=1000;t.cp0=100;t.cp1=0;t.e0=7;t.conductivity=2;t.Tmin=100;t.Tmax=3000;}
 for(auto&t:p.species){t.R=300;t.cp0=1000;t.cp1=.5;t.e0=17;t.Tmin=100;t.Tmax=3000;}
 const Real nan=std::numeric_limits<Real>::quiet_NaN();SolidQ q;q.condensed[0]=1;q.energy=nan;
 Real T=777;assert(recoverSolid<true>(q,p,T,nullptr,-1));assert(T==100);std::printf("solid %a\n",T);
 MaterialPrimitive out;assert(recoverMaterial<true>(q,.001,p,out));assert(out.temperature==100);std::printf("material %a %a %a %a\n",out.temperature,out.porosity,out.pressure,out.conductivity);
 GasThermoAdapter<Ns> adapter(p.species);Real mass[Ns]{};MaterialEnergyEvaluator evaluator{adapter.view(),mass,7,100,0};
 T=99;assert(invertMaterialEnergy<true>(evaluator,nan,100,3000,p.tolerances,T));assert(T==100);std::printf("direct %a\n",T);
 Counted counted;T=99;ugkwp::ThermoInversionControls<double> controls;
 assert(ugkwp::invertCaloricEnergy(counted,nan,100.,3000.,controls,T)==ugkwp::ThermoStatus::NonFiniteInput);assert(T==99&&counted.count==0);std::printf("generic %a %d\n",T,counted.count);
}
'''

METADATA = r'''
#include "gasTransport/MixtureThermo.H"
#include <cassert>
#include <cstring>
#include <cmath>
using namespace ugkwp;
int main(int argc,char**argv){assert(argc==2);
 SpeciesThermoData<double> species[2];double linear[6]={1000,.2,17,1100,.1,23};
 for(int k=0;k<2;++k){species[k].molarMass=.028+.004*k;species[k].coefficientOffset=3*k;species[k].referenceTemperature=300;species[k].referenceEntropy=1500;species[k].hasEntropyReference=true;}
 SpeciesThermoView<double,1> declared;declared.species=species;declared.coefficients=linear;declared.coefficientCount=6;
 int index=0;double T=700;double nasa[14]={3,0,0,0,0,10,7,4,0,0,0,0,20,8};
 if(!std::strcmp(argv[1],"count"))declared.coefficientCount=1;
 else if(!std::strcmp(argv[1],"index"))index=1; // allocated species[1], despite declared Ns=1
 else if(!std::strcmp(argv[1],"offset")){declared.coefficients=linear+3;species[0].coefficientOffset=-3;declared.coefficientCount=3;}
 else if(!std::strcmp(argv[1],"nasa")){species[0].model=SpeciesThermoModel::NASA7;species[0].midTemperature=1000;declared.coefficients=nasa;declared.coefficientCount=7;T=2000;}
 else return 2;
 const auto&d=species[index];const double*c=declared.coefficients+d.coefficientOffset;const double R=universalGasConstant<double>()/d.molarMass;
 gasThermoDetail::SpeciesValues<double> expected;
 if(d.model==SpeciesThermoModel::LinearCp){expected.cp=c[0]+c[1]*T;expected.h=c[2]+T*(c[0]+.5*c[1]*T);expected.entropy=d.referenceEntropy+c[0]*std::log(T/d.referenceTemperature)+c[1]*(T-d.referenceTemperature);}
 else{c+=7;expected.cp=R*(c[0]+T*(c[1]+T*(c[2]+T*(c[3]+T*c[4]))));expected.h=R*(c[5]+T*(c[0]+T*(c[1]/2+T*(c[2]/3+T*(c[3]/4+T*c[4]/5)))));expected.entropy=R*(c[0]*std::log(T)+c[6]+T*(c[1]+T*(c[2]/2+T*(c[3]/3+T*c[4]/4))));}
 expected.cv=expected.cp-R;expected.e=expected.h-R*T;
 gasThermoDetail::SpeciesValues<double> strict{-11,-12,-13,-14,-15},before=strict;
 assert((gasThermoDetail::evaluateSpecies<double,1,double,false>(index,T,declared,strict)==ThermoStatus::InvalidModel));assert(std::memcmp(&strict,&before,sizeof before)==0);
 gasThermoDetail::SpeciesValues<double> completed{-11,-12,-13,-14,-15};
 assert(gasThermoDetail::evaluateSpecies(index,T,declared,completed)==ThermoStatus::InvalidModel);
 assert(std::memcmp(&completed,&expected,sizeof expected)==0);
 if(index==0){double mass[1]={2},energy=-21,capacity=-22;assert(evaluateMixtureEnergy(mass,T,declared,energy,capacity)==ThermoStatus::InvalidModel);assert(energy==2*expected.e&&capacity==2*expected.cv);}
}
'''


def run_probe(folder, root, source, args=()):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "probe.cpp"
    path.write_text(source)
    exe = folder / "probe"
    result = subprocess.run(["g++", "-std=c++14", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I" + str(root / "common"), "-I" + str(root / "applications/CHMT"), str(path), "-o", str(exe)], capture_output=True, text=True)
    (folder / "build.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe), *args], capture_output=True, text=True)
    (folder / "run.log").write_text(result.stdout + result.stderr)
    return result


def test_actual_chmt_successful_generic_nan_fallback_matches_base(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    archive = subprocess.check_output(["git", "archive", BASE, "common", "applications/CHMT"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(base, filter="data")
    old = run_probe(tmp_path / "original", base, MATERIAL)
    new = run_probe(tmp_path / "current", ROOT, MATERIAL)
    assert old.returncode == new.returncode == 0, old.stderr + new.stderr
    assert old.stdout == new.stdout


@pytest.mark.parametrize("scenario", ["count", "index", "offset", "nasa"])
def test_actual_backing_completes_declared_metadata_refusal(tmp_path, scenario):
    result = run_probe(tmp_path, ROOT, METADATA, [scenario])
    assert result.returncode == 0, result.stderr
