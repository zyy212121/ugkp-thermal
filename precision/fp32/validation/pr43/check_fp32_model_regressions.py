#!/usr/bin/env python3
"""Prepare production-function regressions; execution requires explicit --run.

During implementation use --syntax-only or --compile-only. The checkpoint-2
batch may use --run. Radiation stubs replace scheduling/ownership only; the
borrow function, borrowed view, and temperature import are production code.
"""
from pathlib import Path
import argparse
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def function(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def radiation_source():
    borrow = function((ROOT / "applications/CHMT/gpu/Backend.cu").read_text(),
                      "bool borrowRadiationState(")
    importer = function((ROOT / "common/gpu/DiscreteOrdinateRadiation.cu").read_text(),
                        "template<class Real> __global__ void importBoundaryState(")
    return r"""
#include <cassert>
#include <string>
#include <iostream>
#include "gpu/RadiationResidentView.H"
#define __global__
struct Index { int x=0; } blockIdx, threadIdx, blockDim{1};
using ugkp::gpu::RadiationParticleDeviceView;
struct ParticleRadiationState { RadiationParticleDeviceView resident; int response=0; };
struct Particles {
    bool active() const { return true; }
    bool result(bool ok,std::string&) const { return ok; }
    void* owner() const { return nullptr; }
};
struct Backend { Particles particles; void* gasCaptureStream=nullptr; };
namespace ugkp { namespace thermal {
inline bool orderParticlePhaseAfterGas(void*,void*) { return true; }
}}
static float gasBoundaryFp32[1]={300.f};
static double gasBoundaryFp64[1]={300.};
static double contactArea[1]={0.25},contactScale[1]={1.},wallEmissivity[1]={0.8};
static bool ugkwpGpuResidentStrictBorrowRadiationParticleState(void*,RadiationParticleDeviceView* out) {
    out->enabled=true;out->boundaryFaces=1;
    out->boundaryTemperature=gasBoundaryFp64;
    out->boundaryTemperatureFp32=gasBoundaryFp32;
    out->boundaryEmissivity=wallEmissivity;
    out->contactArea=contactArea;out->contactScale=contactScale;
    return true;
}
static bool ugkwpGpuResidentStrictParticleRadiationResponse(void*,int*) { return true; }
static bool downloadRadiationState(Backend&,ParticleRadiationState&,std::string&) { return true; }
""" + borrow + "\n" + importer + r"""
int main() {
    Backend backend;ParticleRadiationState state;std::string error;
    assert(borrowRadiationState(backend,state,error));
    assert(state.resident.boundaryTemperature==nullptr);
    assert(state.resident.boundaryTemperatureFp32==nullptr);
    const unsigned char useTemperature[1]={1},contact[1]={1};
    const double area[1]={1.};
    // CoupledRadiation::advance has populated this from the final WallProgram knot.
    double temperature[1]={1200.},emissivity[1]={1.};
    importBoundaryState<double>(1,state.resident,useTemperature,contact,area,temperature,emissivity);
    assert(temperature[0]==1200.);
    assert(emissivity[0]==0.8*(1.-0.25));
    // The ordinary gas-backed import is still allowed to use its FP32 entry.
    RadiationParticleDeviceView gas;
    ugkwpGpuResidentStrictBorrowRadiationParticleState(nullptr,&gas);
    const unsigned char noContact[1]={0};
    importBoundaryState<double>(1,gas,useTemperature,noContact,area,temperature,emissivity);
    assert(temperature[0]==300.);
    std::cout<<"PASS CHMT material wall 1200 K, gas boundary 300 K\n";
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--syntax-only", action="store_true")
    mode.add_argument("--compile-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="fp32-model-regressions-") as tmp:
        temporary = Path(tmp)
        radiation = temporary / "chmt_radiation_wall.cpp"
        radiation.write_text(radiation_source())
        sources = [ROOT / "validation/pr43/fp32_model_regressions.cpp", radiation]
        for source in sources:
            command = ["g++", "-std=c++17", "-O0", "-Wall", "-Wextra"]
            for include in [ROOT, ROOT / "common", ROOT / "common/gasNumerics"]:
                command += ["-I", str(include)]
            executable = temporary / source.stem
            command += [str(source)]
            command += ["-fsyntax-only"] if args.syntax_only else ["-o", str(executable)]
            subprocess.run(command, check=True)
            print("COMPILED (not executed):", source.name, flush=True)
            if args.run:
                subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    main()
