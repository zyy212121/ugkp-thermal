"""Actual property and input values survive first-error reporting."""
import pathlib
import subprocess

def test_actual_nonfinite_value(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[3]
    source = tmp_path / "probe.cpp"
    source.write_text(r'''#include "gasTransport/MixtureThermo.H"
#include "gasTransport/SpeciesDiffusion.H"
#include <cmath>
#include <limits>
int main() {
 using namespace ugkwp;
 ugkp::diagnostics::DeviceContext d{};
 SpeciesThermoData<double> meta[1];
 meta[0].molarMass=.028;meta[0].minTemperature=1;meta[0].maxTemperature=3000;
 double coef[3]={1000,0,0}; SpeciesThermoView<double,1> t;
 t.species=meta;t.coefficients=coef;t.coefficientCount=3;t.diagnostics=&d;
 double y[1]={1},g[1]={0},f[1]={7},e=7;
 bool ok=correctedSpeciesDiffusion(y,g,std::numeric_limits<double>::quiet_NaN(),(const double*)nullptr,0.,.7,300.,t,f,e);
 if(ok || !std::isnan(d.firstFatal.observed))return 1;
 d={};coef[1]=std::numeric_limits<double>::max();
 double cp=speciesCp(0,300.,t);
 if(!std::isnan(cp) || !std::isinf(d.firstFatal.observed))return 2;
 d={};y[0]=std::numeric_limits<double>::quiet_NaN();
 double R=mixtureGasConstant(y,t);
 if(!std::isnan(R) || !std::isnan(d.firstFatal.observed))return 3;
 return 0;
}
''')
    binary = tmp_path / "probe"
    subprocess.run(["g++", "-std=c++17", "-I" + str(root / "common"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
