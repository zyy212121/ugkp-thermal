"""CHT output precision and scalar lifecycle probes, without a GPU/OpenFOAM runtime."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = (SOURCE_ROOT / 'applications/CHT')
FRONTEND = ROOT / "diluteUgkwpFoam.C"
GUARD = "requireLosslessGasEnergyRestartPrecision"
CALL = GUARD + "(runTime.controlDict());"


def definition(source, signature):
    start = source.index(signature)
    body = source.index("{", start)
    depth = 1
    end = body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def compile_run(tmp_path, source):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ is needed for the host production-code probe")
    code = tmp_path / "probe.cpp"
    executable = tmp_path / "probe"
    code.write_text(source)
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
                    str(code), "-o", str(executable)], check=True, capture_output=True, text=True)
    return subprocess.run([str(executable)], check=True, capture_output=True, text=True)


def test_guard_rejects_lossy_ascii_and_accepts_lossless_formats(tmp_path):
    source = FRONTEND.read_text()
    assert "inline void " + GUARD in source, "CHT has no shared gas-energy restart precision guard"
    guard = definition(source, "inline void " + GUARD)
    compile_run(tmp_path, r'''
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
using scalar = double;
using label = int;
using word = std::string;
struct dictionary {
    word format; int precision;
    template<class T> T lookupOrDefault(const char* key, const T& fallback) const {
        if constexpr (std::is_same<T,word>::value) return format.empty() ? fallback : format;
        else return std::string(key)=="writePrecision" && precision>=0 ? precision : fallback;
    }
};
struct FatalStream { template<class T> FatalStream& operator<<(const T&) { return *this; } };
FatalStream fatal;
#define FatalErrorInFunction fatal
struct FatalToken {};
constexpr FatalToken FatalError{};
int exit(FatalToken) { throw std::runtime_error("fatal"); }
''' + guard + r'''
int main() {
    for (const auto format : {word(), word("ascii")}) {
        for (const int precision : {-1,0,6,16}) {
            bool rejected=false;
            try { requireLosslessGasEnergyRestartPrecision({format,precision}); }
            catch(const std::runtime_error&) { rejected=true; }
            if(!rejected) return 1;
        }
        for (const int precision : {17,20})
            requireLosslessGasEnergyRestartPrecision({format,precision});
    }
    for (const int precision : {-1,0,6,17})
        requireLosslessGasEnergyRestartPrecision({"binary",precision});
}
''')


def test_startup_and_both_write_routes_validate_current_output_settings():
    source = FRONTEND.read_text()
    main = source[source.index("int main("):]
    assert CALL in main, "resident startup does not validate restart precision"
    assert main.index(CALL) < main.index("resident.initialise"), "validation is too late"
    pure = main[main.index("if (gpuResidentPureGasOnly && !gpuResidentSolidThermalCoupling)"):]
    pure = pure[:pure.index("GpuParticleResidentSolver resident;")]
    pure_write = pure[pure.index("if (runTime.writeTime())"):]
    assert CALL in pure_write, "pure-gas writes bypass lossless precision enforcement"
    assert pure_write.index(CALL) < pure_write.index("resident.downloadToHostMirror")
    particle = main[main.index("GpuParticleResidentSolver resident;"):]
    particle_write = particle[particle.index("const bool thermalCouplingWrite ="):]
    assert CALL in particle_write, "particle writes bypass lossless precision enforcement"
    assert particle_write.index(CALL) < particle_write.index("persistAtWriteTime")
    assert particle_write.index(CALL) < particle_write.index("resident.downloadToHostMirror")
    assert "validateBeforeResidentInitialise" in particle, "existing thermal preflight was removed"


def test_actual_double_copy_helpers_preserve_restart_and_old_float_values(tmp_path):
    source = (ROOT / "gpu/GpuResidentStrict.cu").read_text()
    upload = definition(source, "template<class T>\nint copyToDevice")
    download = definition(source, "template<class T>\nint copyToHost")
    compile_run(tmp_path, r'''
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
int cudaMemcpy(void* dst,const void* src,size_t n,int) { std::memcpy(dst,src,n);return 0; }
void setLastError(const char*,int) {}
''' + upload + '\n' + download + r'''
bool same(double a,double b) { return std::memcmp(&a,&b,sizeof a)==0; }
int main() {
    const float oldFloat=static_cast<float>(1000000000.03);
    const double values[]={1000000000.03,1000000000.05,-1000000000.03,
                          static_cast<double>(oldFloat)};
    for (double original:values) {
        double device=0,host=0;
        if(copyToDevice(&device,&original,1,"upload") || copyToHost(&host,&device,1,"download")) return 1;
        if(!same(original,host)) return 2;
        std::stringstream ascii;
        ascii << std::setprecision(std::numeric_limits<double>::max_digits10) << host;
        double restored=0; ascii >> restored;
        if(!same(original,restored)) return 3;
        std::stringstream binary;
        binary.write(reinterpret_cast<const char*>(&host),sizeof host);
        binary.read(reinterpret_cast<char*>(&restored),sizeof restored);
        if(!same(original,restored)) return 4;
    }
    if(values[3]!=static_cast<double>(oldFloat) || values[3]==values[0]) return 5;
    std::stringstream lossy; lossy << std::setprecision(6) << values[0];
    double lost=0;lossy>>lost;
    if(same(values[0],lost)) return 6;
}
''')
