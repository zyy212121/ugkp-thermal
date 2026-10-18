"""Diagnostic-only reductions must not reject finite physical components."""
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT

@pytest.mark.parametrize('probe',['fractions','packet'])
def test_finite_components_with_overflowing_audit(tmp_path,probe):
    bodies={
'fractions':r'''#include "gasTransport/MixtureThermo.H"
#include <cassert>
#include <limits>
int main(){ugkp::diagnostics::DeviceContext d{}; double p[2]={1e308,1e308},y[2]={-7,-7};
bool ok=ugkwp::mixtureMassFractionsFromDensities<double,2>(p,1.,1e-10,y,&d);
assert(ok && !d.fatalClaim && y[0]==1 && y[1]==1 && d.warnings[0].count);
d={};p[0]=std::numeric_limits<double>::quiet_NaN();
assert((!ugkwp::mixtureMassFractionsFromDensities<double,2>(p,1.,1e-10,y,&d)));assert(d.fatalClaim);
}''',
'packet':r'''#include "coupling/ExchangeLedger.H"
#include <cassert>
#include <limits>
int main(){chmt::ExchangePacket p{};p.kind=chmt::ExchangeKind::GasSolid;p.gasCell=0;p.solidCell=0;
p.conductive=1e308;p.advective=-1e308;ugkp::diagnostics::DeviceContext d{};chmt::Tolerances t;t.diagnostics=&d;chmt::DeviceStatus s{};
assert(chmt::validatePacketMath(p,t,&s));assert(!d.fatalClaim);assert(d.warnings[0].count);
d={};p.conductive=std::numeric_limits<double>::infinity();assert(!chmt::validatePacketMath(p,t,&s));assert(d.fatalClaim);
}''' }
    source=tmp_path/'probe.cpp';source.write_text(bodies[probe]);binary=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-I'+str(SOURCE_ROOT/'common'),'-I'+str(SOURCE_ROOT/'applications/CHMT'),str(source),'-o',str(binary)],check=True)
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def test_inventory_audit_sum_overflow(tmp_path):
    source=tmp_path/'inventory.cpp';source.write_text(r'''#include "coupling/ExchangeLedger.H"
#include <cassert>
int main(){ugkp::diagnostics::DeviceContext d{};
chmt::GasQ gas{};gas.mass=1e308;gas.species[0]=gas.species[1]=1e308;
assert(chmt::validGasInventory(gas,&d));assert(!d.fatalClaim && d.warnings[0].count);
d={};chmt::FilmQ film{};film.mass=1e308;film.species[0]=film.species[1]=1e308;
assert(chmt::validFilmInventory(film,false,&d));assert(!d.fatalClaim && d.warnings[0].count);
d={};chmt::SolidQ solid{};solid.pore[0]=solid.pore[1]=1e308;
assert(chmt::validSolidInventory(solid,&d));assert(!d.fatalClaim && d.warnings[0].count);
d={};gas.species[0]=-1;assert(!chmt::validGasInventory(gas,&d));assert(d.fatalClaim);
}''');binary=tmp_path/'inventory'
    subprocess.run(['g++','-std=c++17','-I'+str(SOURCE_ROOT/'common'),'-I'+str(SOURCE_ROOT/'applications/CHMT'),str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
