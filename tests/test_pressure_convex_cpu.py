"""CPU executions of the host/device per-face convex pressure algebra.

These tests compile the production template for float and double. They exercise
local endpoints and the convex decomposition after arbitrary shared-face minima;
they do not claim to execute a CUDA kernel or reproduce its reduction ordering.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "common/GpuPressureConvexAlgebra.cuh"

CPP = r'''
#include "GpuPressureConvexAlgebra.cuh"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
using Real = TEST_REAL;
using pressure_convex::limitFace;
long double internal(long double rho, long double p, long double e) {
    return e-p*p/(2*rho);
}
auto call(Real rho, Real p, Real e, Real floor, Real du, Real dp, Real de) {
    return limitFace(rho,p,Real(0),Real(0),e,floor,du,dp,Real(0),Real(0),de);
}
void check(bool yes, const char* message) {
    if(!yes) { std::cerr << message << '\n'; std::exit(1); }
}
int main(int argc,char** argv) {
    const std::string mode=argv[1];
    const long double tol=128*std::numeric_limits<Real>::epsilon();
    if(mode=="identity") {
        auto r=call(2,3,10,1,0,0,0);
        check(r.valid && r.beta==Real(1),"zero delta must be the identity, even at zero dU");
        r=call(1,0,2,0,10,1,0);
        check(r.valid && r.beta==Real(1),"admissible endpoint must be retained exactly");
    } else if(mode=="roots") {
        auto r=call(1,0,1,0,100,2,0);
        check(r.valid && r.beta>Real(.70) && r.beta<=Real(std::sqrt(.5)),"quadratic root incorrect");
        check(internal(1,(long double)r.beta*2,1)>=0,"quadratic root crossed boundary");
        r=call(1,0,1,0,100,0,-4);
        check(r.valid && r.beta>Real(.249) && r.beta<=Real(.25),"linear root incorrect");
        r=call(1,1,1,0,100,2,-1000000);
        check(r.valid && r.beta>0 && internal(1,1+(long double)r.beta*2,1-(long double)r.beta*1000000)>=0,"cancellation-safe root required");
        r=call(1,1,Real(.5),0,100,-1,0);
        check(r.valid && r.beta==Real(1),"inward direction at zero slack must survive");
    } else if(mode=="velocity") {
        auto r=call(1,100,10000,0,Real(.5),2,0);
        check(r.valid && r.beta>Real(.249) && r.beta<=Real(.25),"dU must constrain increment, not absolute velocity");
        r=call(1,0,1,0,0,1,0);
        check(r.valid && r.beta==0,"zero dU must reject a nonzero momentum increment");
        r=call(1,0,1,0,0,0,1);
        check(r.valid && r.beta==1,"zero dU must retain energy-only increments");
    } else if(mode=="cold") {
        auto r=call(2,2,1,10,10,0,0);
        check(r.valid && r.beta==1,"cold valid state must not require energy injection");
        r=call(2,2,1,10,10,0,-1);
        check(r.valid && r.beta==0,"cold state cannot lose internal energy");
        r=call(2,2,1,10,10,-1,0);
        check(r.valid && r.beta==1,"cold state may gain internal energy through momentum reduction");
        r=call(1,0,Real(.25),10,10,1,0);
        check(r.valid && r.beta==0,"below-configured-floor state must preserve its initial internal energy");
    } else if(mode=="invalid") {
        const Real nan=std::numeric_limits<Real>::quiet_NaN();
        const Real inf=std::numeric_limits<Real>::infinity();
        for(Real rho:{Real(0),Real(-1),nan,inf}) {
            auto r=call(rho,0,1,0,1,0,0); check(!r.valid && r.beta==0,"invalid rho was sanitized");
        }
        for(int field=0;field<10;++field) for(Real bad:{nan,inf,-inf}) {
            Real v[10]={1,0,0,0,1,0,1,0,0,0}; v[field]=bad;
            auto r=limitFace(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],Real(0));
            check(!r.valid && r.beta==0,"nonfinite raw input was sanitized");
        }
        check(!call(1,2,1,0,1,0,0).valid,"negative initial internal energy was accepted");
        check(!call(1,0,1,-1,1,0,0).valid,"negative configured floor was accepted");
        check(!call(1,0,1,0,-1,0,0).valid,"negative dU was accepted");
        auto r=call(1,0,1,0,1,0,nan); check(!r.valid && r.beta==0,"nonfinite energy delta was accepted");
    } else if(mode=="subnormal_margin") {
        const Real rho=std::numeric_limits<Real>::denorm_min()*Real(123);
        const Real du=std::numeric_limits<Real>::min()*Real(.333);
        const Real dp=std::numeric_limits<Real>::min();
        auto r=call(rho,0,1,0,du,dp,0);
        const long double exactCap=(long double)rho*du/dp;
        check(r.valid && r.beta>0,"representable subnormal limit was discarded");
        check((long double)r.beta<=exactCap,"relative epsilon margin did not move subnormal beta toward safety");
    } else if(mode=="helpers") {
        auto initial=pressure_convex::initialState(Real(2),Real(2),Real(0),Real(0),Real(1),Real(10));
        check(initial.valid && initial.internal==0 && initial.floor==0,"initial helper must preserve a cold state");
        check(pressure_convex::admissibleIncrement(Real(1),Real(0),Real(0),Real(0),Real(1),Real(0),Real(1),Real(1),Real(0),Real(0),Real(0)),"feasible aggregate rejected");
        for(Real bad:{Real(-1),std::numeric_limits<Real>::quiet_NaN(),std::numeric_limits<Real>::infinity()}) {
            check(!pressure_convex::admissibleIncrement(Real(1),Real(0),Real(0),Real(0),Real(1),Real(0),bad,Real(0),Real(0),Real(0),Real(0)),"aggregate guard must reject invalid dU even for zero momentum delta");
        }
    } else if(mode=="three_dimensional") {
        auto r=limitFace(Real(1),Real(0),Real(0),Real(0),Real(1),Real(0),Real(1),Real(1),Real(2),Real(2),Real(0));
        check(r.valid && r.beta>Real(.332) && r.beta<Real(.334),"3D dU norm is incorrect");
        const long double length=3*(long double)r.beta;
        check(length<=1,"3D dU radius crossed");
        const Real big=std::numeric_limits<Real>::max()/Real(4);
        r=limitFace(big,big,big,big,Real(2)*big,Real(0),Real(10),-big,-big,-big,Real(0));
        check(r.valid && r.beta==Real(1),"scaled three-axis kinetic energy overflowed");
        // Near a tangent, negative b must use the rationalized root branch.
        r=call(1,1,Real(.5)+Real(128)*std::numeric_limits<Real>::epsilon(),0,100,1,-1);
        check(r.valid && r.beta>0,"small positive internal budget was lost");
        const long double p=1+(long double)r.beta,e=Real(.5)+Real(128)*std::numeric_limits<Real>::epsilon()-(long double)r.beta;
        check(internal(1,p,e)>=0,"near-tangent boundary crossed");
    } else if(mode=="cancellation") {
        // Both raw faces sum to zero. The old net-cell lambda is one, but a
        // neighbour reducing just the second face uncancels a forbidden kick.
        const Real raw[2]={2,-2}, omega=Real(.5);
        check(internal(1,raw[0]+raw[1],1)==1,"invalid cancellation fixture");
        check(internal(1,raw[0],1)<0,"old min-neighbour failure did not reproduce");
        auto a=call(1,0,1,0,100,raw[0]/omega,0);
        auto b=call(1,0,1,0,100,raw[1]/omega,0);
        Real cell=std::min(a.beta,b.beta);
        check(a.valid && b.valid && cell>Real(.35) && cell<Real(.36),"per-face endpoint limit incorrect");
        for(int i=0;i<=100;++i) for(int j=0;j<=100;++j) {
            Real shared0=std::min(cell,Real(i)/100), shared1=std::min(cell,Real(j)/100);
            long double dp=(long double)shared0*raw[0]+(long double)shared1*raw[1];
            check(internal(1,dp,1)>=-tol,"arbitrary smaller shared-face coefficients broke convexity");
        }
        // Quantify the deliberate local-price: exact cancellation is globally
        // admissible at beta=1; independent endpoints retain about 35.355%.
        const long double loss=1-(long double)cell;
        check(loss>.646 && loss<.647,"unexpected cancellation over-dissipation");
        std::cout << "cancellation retained=" << cell << " discarded=" << loss << '\n';
    } else if(mode=="random") {
        std::mt19937 gen(48151); std::uniform_real_distribution<double> u(0,1);
        for(int trial=0;trial<1500;++trial) {
            Real rho=Real(.1+10*u(gen)), p=Real(10*u(gen)-5);
            Real energy=Real((long double)p*p/(2*rho)+.1+10*u(gen));
            Real floor=Real(.05), du=Real(.01+5*u(gen));
            Real dp[6],de[6],cell=1;
            for(int f=0;f<6;++f) {
                dp[f]=Real(20*u(gen)-10); de[f]=Real(20*u(gen)-10);
                auto r=call(rho,p,energy,floor,du,Real(6)*dp[f],Real(6)*de[f]);
                check(r.valid,"finite valid random state rejected"); cell=std::min(cell,r.beta);
            }
            long double totalP=p,totalE=energy,change=0;
            for(int f=0;f<6;++f) {
                Real shared=std::min(cell,Real(u(gen))); change+=(long double)shared*dp[f];
                totalP+=(long double)shared*dp[f]; totalE+=(long double)shared*de[f];
            }
            check(internal(rho,totalP,totalE)>=floor-tol*(1+energy),"random shared minima broke internal energy");
            check(std::abs(change)<=((long double)rho*du)*(1+tol),"random shared minima broke dU");
        }
    } else if(mode=="huge") {
        const Real big=std::numeric_limits<Real>::max()/Real(8);
        auto r=call(big,big,big,0,2,big,0);
        check(r.valid && r.beta>Real(.4),"overflow-safe finite state was rejected");
        check(internal((long double)big,(long double)big*(1+r.beta),(long double)big)>=0,"huge-state root crossed boundary");
        r=call(1,0,1,0,1,big,0);
        check(r.valid && r.beta>0 && (long double)r.beta*big<=1,"huge delta requires representable tiny beta");
        Real tiny=std::numeric_limits<Real>::min()*Real(16);
        r=call(tiny,0,tiny,0,1,tiny,0);
        check(r.valid && r.beta==1,"tiny uniformly scaled state should remain admissible");
        r=call(1,0,big,0,big,0,big);
        check(r.valid && r.beta==1,"large finite energy-only increment was rejected");
    }
}
'''

@pytest.fixture(scope="module", params=["float", "double"])
def executable(request, tmp_path_factory):
    assert HEADER.exists(), "production host/device convex pressure helper is missing"
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required to exercise the production C++ header")
    directory = tmp_path_factory.mktemp("pressure_convex_" + request.param)
    source = directory / "check.cpp"
    source.write_text(CPP)
    binary = directory / "check"
    command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
               "-DTEST_REAL=" + request.param, "-I", str(ROOT / "common"), str(source), "-o", str(binary)]
    build = subprocess.run(command, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    return binary

@pytest.mark.parametrize("mode", ["identity", "roots", "velocity", "cold", "invalid", "helpers", "subnormal_margin", "three_dimensional", "cancellation", "random", "huge"])
def test_real_convex_pressure_algebra(executable, mode):
    run = subprocess.run([str(executable), mode], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
