"""Actual required caloric expressions for private terminal conduction."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess

ROOT = SOURCE_ROOT


def test_terminal_material_caloric_outputs(tmp_path):
    source = tmp_path / 'probe.cpp'
    source.write_text(r'''
#include "materials/MaterialCaloric.H"
#include <cassert>
#include <cmath>
#include <limits>
int main(){
 using namespace chmt;
 PhysicsConfig p;
 for(auto& t:p.condensed){t.rho=1000;t.cp0=-100;t.cp1=0;t.e0=7;t.conductivity=2;t.Tmin=100;t.Tmax=3000;}
 for(auto& t:p.species){t.R=300;t.cp0=1000;t.cp1=.5;t.e0=17;t.Tmin=100;t.Tmax=3000;}
 SolidQ q;q.condensed[0]=1;q.energy=condensedE(p.condensed[0],500);
 MaterialPrimitive out;bool numerical=false;
 assert(!recoverMaterial(q,.001,p,out,nullptr,-1,&numerical)&&numerical);
 assert(recoverMaterial<true>(q,.001,p,out));
 assert(out.temperature==500&&out.porosity==0&&out.pressure==0&&out.conductivity==2);
 MaterialCaloric c;std::string error;
 assert(!materialCaloric(q,p,c,error,&numerical)&&numerical);
 assert(materialCaloric<true>(q,p,c,error));
 assert(c.energy<true>(500)==q.energy&&c.secant<true>(100,500)==-100);
 GasThermoAdapter<Ns> adapter(p.species);Real mass[Ns]{};mass[0]=2;
 MaterialEnergyEvaluator evaluator{adapter.view(),mass,0,-4000,0};
 Real energy=123,capacity=456;
 assert(evaluator(500,energy,capacity)==ugkwp::ThermoStatus::NonPositiveHeatCapacity);
 assert(energy==123&&capacity==456);
 assert(evaluator.evaluate<true>(500,energy,capacity)==ugkwp::ThermoStatus::Success);
 assert(energy==-4000*500+2*speciesE(p.species[0],500));
 assert(capacity==-4000+2*speciesCv(p.species[0],500));
 const Real nan=std::numeric_limits<Real>::quiet_NaN();
 assert(evaluator.evaluate<true>(nan,energy,capacity)==ugkwp::ThermoStatus::Success);
 assert(std::isnan(energy)&&std::isnan(capacity));
 TerminalMaterialEnergyEvaluator tracked{evaluator};
 assert(tracked(451,energy,capacity)==ugkwp::ThermoStatus::Success&&tracked.available&&tracked.lastTemperature==451);
 Real T=99;assert(invertMaterialEnergy<true>(evaluator,nan,100,3000,p.tolerances,T));
 assert(T==100); // Common target refusal precedes evaluation; actual lower formula is then evaluated.
 p.species[0].model=static_cast<ugkwp::SpeciesThermoModel>(99);q.pore[0]=.1;
 assert(!materialCaloric<true>(q,p,c,error,&numerical)&&!numerical);
 assert(!recoverMaterial<true>(q,.001,p,out,nullptr,-1,&numerical)&&!numerical);
 // Current-width coefficient storage is truly absent, not numerically invalid.
 auto missing=adapter.view();missing.coefficients=nullptr;
 MaterialEnergyEvaluator absent{missing,mass,0,0,0};T=99;
 assert(!invertMaterialEnergy<true>(absent,1000,100,3000,p.tolerances,T)&&T==99);
}
''')
    exe = tmp_path / 'probe'
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
                    '-I' + str(ROOT / 'applications/CHMT'), '-I' + str(ROOT / 'common'),
                    str(source), '-o', str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=30)
