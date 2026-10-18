"""Exchange residuals are observational; physical faults use shared diagnostics."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_exchange_diagnostics_and_value_preservation(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    source = tmp_path / "exchange.cpp"
    source.write_text(r'''
#include "coupling/ExchangeLedger.H"
#include <cassert>
#include <limits>
using namespace chmt;
using namespace ugkp::diagnostics;
int main() {
    DeviceContext ctx{};Tolerances tol;tol.diagnostics=&ctx;
    ExchangePacket p;p.gasCell=p.solidCell=0;p.mass=.25;
    p.species[0]=.20;p.condensed[0]=.24;p.energy=1000;p.advective=900;
    DeviceStatus status{};
    assert(validatePacketMath(p,tol,&status));
    assert(!ctx.fatalClaim&&ctx.warnings[0].count==3&&status.code==0);
    GasQ gas;gas.mass=1;gas.species[0]=1;gas.energy=1e5;
    SolidQ solid;solid.condensed[0]=2;solid.energy=4e5;
    assert(applyPacketTransaction(p,&gas,&solid,nullptr,nullptr,tol,&status));
    assert(gas.mass==1.25&&gas.species[0]==1.20&&gas.energy==101000);
    assert(solid.condensed[0]==1.76&&solid.energy==399000);
    assert(p.species[0]==.20&&p.mass==.25&&p.energy==1000&&p.advective==900);
    assert(!ctx.fatalClaim&&p.consumerMask==(ConsumeGas|ConsumeSolid));
    FilmQ film;film.mass=1;film.species[0]=.9;film.enthalpy=2;
    assert(validFilmInventory(film,false,&ctx)&&film.species[0]==.9);
    film.mass=0;film.species[0]=0;film.enthalpy=1e-14;
    assert(validFilmInventory(film,false,&ctx)&&film.enthalpy==1e-14);
    reset(ctx,1,1);p.consumerMask=0;p.energy=std::numeric_limits<Real>::quiet_NaN();
    assert(!validatePacketMath(p,tol,&status));
    assert(ctx.fatalClaim&&ctx.firstFatal.code==Code::NonFinite);
    reset(ctx,2,1);p.energy=1000;p.gasCell=-1;
    assert(!validatePacketMath(p,tol,&status));
    assert(ctx.fatalClaim&&ctx.firstFatal.code==Code::InvalidIndex);
    reset(ctx,3,1);p.gasCell=0;
    assert(!applyPacketTransaction(p,nullptr,&solid,nullptr,nullptr,tol,&status));
    assert(ctx.fatalClaim&&ctx.firstFatal.code==Code::MissingOperator);
    reset(ctx,4,1);film.enthalpy=std::numeric_limits<Real>::quiet_NaN();
    assert(!validFilmInventory(film,false,&ctx));
    assert(ctx.fatalClaim&&ctx.firstFatal.code==Code::NonFinite);
}
''')
    binary = tmp_path / "exchange"
    subprocess.run([compiler, "-std=c++17", "-DUGKWP_GAS_SPECIES=2",
                    "-I" + str(ROOT / "applications/CHMT"),
                    "-I" + str(ROOT / "common"), str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
