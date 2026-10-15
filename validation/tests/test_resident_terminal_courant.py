"""Execute the real resident Courant ABI bodies with serial host-only shims.

Both CHT and FSH bodies are extracted from production source. The immutable
baseline supplies the old body, not a hand-written approximation of it. Typed
host/device owners point at double arrays; kernel shims write those arrays and
the thrust shim actually reduces them. This checks wrapper orchestration,
numerical admission, and error propagation, not CUDA kernel arithmetic or GPU
execution. No real CUDA claim is made.

For a numerical-admission red check, set RESIDENT_COURANT_SOURCE_REVISION to
5f1a39ac96e2e7ff9a0640a4db344d9cccd34245 when running this file. The candidate
body then comes from that revision and continuation tests must fail.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT


from pathlib import Path
import itertools
import math
import os
import re
import subprocess

import pytest


ROOT = SOURCE_ROOT
ORIGINAL = "5f1a39ac96e2e7ff9a0640a4db344d9cccd34245"
ABI = "ugkwpGpuResidentStrictComputeGasCourant"
BACKENDS = (
    "applications/CHT/gpu/GpuResidentStrict.cu",
    "applications/FSH/private_backend/GpuResidentStrict.cu",
)


def _read_source(path, revision=None):
    if revision:
        return subprocess.check_output(
            ["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True
        )
    return (ROOT / path).read_text()


def _abi_body(source, name):
    start = source.index('extern "C" int ' + ABI)
    opening = source.index("{", start)
    depth = 0

    tokens = re.finditer(
        r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/|[{}]',
        source[opening:],
        re.S,
    )
    for token in tokens:
        if token.group() == "{":
            depth += 1
        elif token.group() == "}":
            depth -= 1
            if depth == 0:
                body = source[start : opening + token.end()]
                break
    else:
        raise AssertionError("Unterminated production Courant ABI body")
    body = body.replace(ABI, name, 1)

    body, count = re.subn(
        r"\b(\w+)<<<(.*?)>>>\s*\(",
        lambda m: f'launch("{m[1]}", {m[2]}, {m[1]}, ',
        body,
        flags=re.S,
    )
    assert count == 13, f"Unexpected Courant launch count: {count}"
    return body


SHIMS = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
using GpuReal = double;
#define GPU_R(x) x
static_assert(sizeof(GpuReal) == 8 && std::numeric_limits<GpuReal>::is_iec559,
              "This fixture requires actual IEEE-754 float64 storage");
constexpr double OfGreat = 1.0 / 2.22044604925031308085e-16;
constexpr int capacity = 4;
#define ALL_FIELDS(X) \
    X(rho) X(rhoUx) X(rhoE) X(rhoK) X(rhoOmega) X(inverseVolume) \
    X(Ux) X(Tgas) X(k) X(omega) X(boundaryU) X(boundaryT) \
    X(riemannU) X(riemannT) X(gradU) X(gradT) X(limiter) X(muT) \
    X(faceCo) X(sstGrad) X(gasFluxPositivityScale) \
    X(gasDiffusionNumber) X(sstSourceNumber)
struct Fields {
#define DECLARE_ARRAY(name) std::array<double, capacity> name;
    ALL_FIELDS(DECLARE_ARRAY)
#undef DECLARE_ARRAY
};
struct DeviceState {
    DeviceState* deviceState = nullptr;
    int nCells = 3, nFaces = 4, hostTurbulenceModel = 0;
    int fixedCellBlockThreads = 2, fixedFaceBlockThreads = 3;
    double gammaGas = 1.4, Rgas = 287.0;
#define DECLARE_POINTER(name) double* name = nullptr;
    ALL_FIELDS(DECLARE_POINTER)
#undef DECLARE_POINTER
};
struct Fixture {
    Fields fields;
    DeviceState host, device;
    Fixture(int cells, int faces, int turbulence, const std::string& result) {
        host.nCells = cells; host.nFaces = faces;
        host.hostTurbulenceModel = turbulence;
#define INITIALISE_FIELD(name) \
        fields.name.fill(-777.25); host.name = fields.name.data();
        ALL_FIELDS(INITIALISE_FIELD)
#undef INITIALISE_FIELD
        for (int i = 0; i < capacity; ++i) {
            host.rho[i] = 1.0 + i * 0.25;
            host.rhoUx[i] = host.rho[i] * (2.0 + i);
            host.rhoE[i] = host.rho[i] * (220000.0 + 1000.0 * i);
            host.rhoK[i] = 0.2 + 0.1 * i;
            host.rhoOmega[i] = 0.7 + 0.05 * i;
            host.inverseVolume[i] = 1.0 + 0.2 * i;
            if (result == "large") host.inverseVolume[i] *= OfGreat;
            if (result == "inf") host.inverseVolume[i] = INFINITY;
            if (result == "nan") host.inverseVolume[i] = NAN;
        }
        device = host;
        host.deviceState = &device;
        device.deviceState = &device;
    }
};
char lastError[2048] = "";
std::vector<std::string> trace;
DeviceState* expectedDevice = nullptr;
std::string failLaunch, plumbing;
int copies = 0, pendingCudaError = 0;
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
void setLastErrorText(const char* text) {
    std::snprintf(lastError, sizeof(lastError), "%s", text);
}
void setLastError(const char* text, cudaError_t error) {
    std::snprintf(lastError, sizeof(lastError), "%s: CUDA error %d", text, error);
}
DeviceState* asState(void* handle) { return static_cast<DeviceState*>(handle); }
int validateState(DeviceState* s, const char*) {
    if (s && s->deviceState) return 0;
    setLastErrorText("missing resident owner"); return 1;
}
int tuneFixedWorkBlockThreads(DeviceState*, double dt, double schedule) {
    char text[160]; std::snprintf(text, sizeof(text), "tune %a %a", dt, schedule);
    trace.emplace_back(text);
    if (plumbing == "tune") { setLastErrorText("tuning failure"); return 1; }
    return 0;
}
cudaError_t cudaGetLastError() {
    const int error = pendingCudaError; pendingCudaError = 0;
    trace.emplace_back("cuda " + std::to_string(error)); return error;
}
template<class Kernel, class... Args>
void launch(const char* name, int grid, int block, Kernel kernel,
            DeviceState* device, Args... args) {
    assert(device == expectedDevice); // The actual typed device owner is used.
    trace.emplace_back(std::string("launch ") + name + " " +
                       std::to_string(grid) + " " + std::to_string(block));
    if (failLaunch == name) { pendingCudaError = 719; return; }
    kernel(device, args...);
}
template<class T>
int copyToHost(T* destination, const T* source, int count, const char* label) {
    ++copies; trace.emplace_back(std::string("copy ") + label);
    std::copy_n(source, count, destination); return 0;
}
void recoverGasPrimitivesKernel(DeviceState* s) {
    for (int i = 0; i < s->nCells; ++i) {
        s->Ux[i] = s->rhoUx[i] / s->rho[i];
        s->Tgas[i] = (s->rhoE[i] / s->rho[i] - 0.5 * s->Ux[i] * s->Ux[i])
                      * (s->gammaGas - 1.0) / s->Rgas;
    }
}
void recoverSstPrimitivesKernel(DeviceState* s) {
    for (int i = 0; i < s->nCells; ++i) {
        s->k[i] = s->rhoK[i] / s->rho[i];
        s->omega[i] = s->rhoOmega[i] / s->rho[i];
    }
}
void updateLegacyGasBoundaryMirrorKernel(DeviceState* s, double time) {
    for (int f = 0; f < s->nFaces; ++f) {
        s->boundaryU[f] = s->Ux[f % s->nCells] + (f + 1) * 0.01 * time;
        s->boundaryT[f] = s->Tgas[f % s->nCells] + 0.1 * time;
    }
}
void updateRiemannBoundaryMirrorKernel(DeviceState* s) {
    for (int f = 0; f < s->nFaces; ++f) {
        const int other = (f + 1) % s->nCells;
        s->riemannU[f] = 0.5 * (s->boundaryU[f] + s->Ux[other]);
        s->riemannT[f] = 0.5 * (s->boundaryT[f] + s->Tgas[other]);
    }
}
void applySstWallFunctionStateKernel(DeviceState* s) {
    for (int i = 0; i < s->nCells; ++i) {
        s->k[i] += 0.01 * std::fabs(s->Ux[i]);
        s->omega[i] += 0.1 * s->k[i];
    }
}
void computeGasPrimitiveGradientsKernel(DeviceState* s, bool boundary) {
    assert(boundary);
    for (int i = 0; i < s->nCells; ++i) {
        const int other = (i + 1) % s->nCells;
        s->gradU[i] = s->Ux[other] - s->Ux[i];
        s->gradT[i] = s->Tgas[other] - s->Tgas[i];
        if (s->nFaces) {
            s->gradU[i] += s->riemannU[i % s->nFaces] - s->Ux[i];
            s->gradT[i] += s->riemannT[i % s->nFaces] - s->Tgas[i];
        }
    }
}
void computeGasGradientLimiterKernel(DeviceState* s) {
    for (int i = 0; i < s->nCells; ++i)
        s->limiter[i] = 1.0 / (1.0 + std::fabs(s->gradU[i]) +
                              0.01 * std::fabs(s->gradT[i]));
}
void computeGasEddyViscosityKernel(DeviceState* s) {
    for (int i = 0; i < s->nCells; ++i)
        s->muT[i] = s->hostTurbulenceModel == 3
            ? s->rho[i] * s->k[i] / s->omega[i] * s->limiter[i] : 0.0;
}
void computeGasCourantFieldKernel(DeviceState* s, double dt) {
    for (int f = 0; f < s->nFaces; ++f)
        s->faceCo[f] = dt * (std::fabs(s->riemannU[f]) +
                            std::sqrt(std::fabs(s->riemannT[f]))) *
                       s->inverseVolume[f % s->nCells];
}
void computeSstGradientsKernel(DeviceState* s) {
    for (int i = 0; i < s->nCells; ++i)
        s->sstGrad[i] = s->k[(i + 1) % s->nCells] - s->k[i] +
                       0.1 * s->omega[i] + s->gradU[i];
}
void computeGasConvectiveCourantByCellKernel(DeviceState* s, double dt) {
    for (int i = 0; i < s->nCells; ++i) {
        double value = dt * (std::fabs(s->Ux[i]) +
            std::sqrt(s->gammaGas * s->Rgas * s->Tgas[i])) * s->inverseVolume[i];
        for (int f = i; f < s->nFaces; f += s->nCells) value += s->faceCo[f];
        s->gasFluxPositivityScale[i] = value;
    }
}
void computeGasDiffusionNumberKernel(DeviceState* s, double dt, double target) {
    for (int i = 0; i < s->nCells; ++i)
        s->gasDiffusionNumber[i] = dt * target *
            (1.0 + std::fabs(s->gradU[i]) + s->muT[i]) * s->inverseVolume[i];
}
void computeSstStabilityNumberKernel(DeviceState* s, double dt, double target) {
    for (int i = 0; i < s->nCells; ++i)
        s->sstSourceNumber[i] = dt * target *
            (s->k[i] + s->omega[i] + std::fabs(s->sstGrad[i])) * s->inverseVolume[i];
}
namespace thrust {
template<class T> using device_ptr = T*;
constexpr int device = 0;
template<class T> T* max_element(int, T* begin, T* end) {
    assert(end - begin == expectedDevice->nCells);
    const char* field = begin == expectedDevice->gasFluxPositivityScale ? "convective"
        : begin == expectedDevice->gasDiffusionNumber ? "diffusion"
        : begin == expectedDevice->sstSourceNumber ? "sst" : "unexpected";
    assert(std::strcmp(field, "unexpected") != 0);
    T* found = std::max_element(begin, end);
    char text[128];
    std::snprintf(text, sizeof(text), "reduce %s %td %td %a", field,
                  end - begin, found - begin, *found);
    trace.emplace_back(text);
    return found;
}
}
'''


MAIN = r'''
int main(int argc, char** argv) {
    assert(argc == 12);
    const bool original = std::string(argv[1]) == "original";
    Fixture fixture(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), argv[8]);
    const double dt = std::strtod(argv[5], nullptr);
    const double target = std::strtod(argv[6], nullptr);
    const double schedule = std::strtod(argv[7], nullptr);
    failLaunch = argv[9]; plumbing = argv[10];
    expectedDevice = &fixture.device;
    void* handle = plumbing == "owner" ? nullptr : &fixture.host;
    double output = -987.5;
    double* destination = std::string(argv[11]) == "null" ? nullptr : &output;
    const int rc = original
        ? originalCourant(handle, dt, target, schedule, destination)
        : candidateCourant(handle, dt, target, schedule, destination);
    std::printf("RESULT %d %a %d\nERROR %s\n", rc, output, copies, lastError);
#define PRINT_FIELD(name) \
    std::printf("FIELD " #name); \
    for (double value : fixture.fields.name) std::printf(" %a", value); \
    std::puts("");
    ALL_FIELDS(PRINT_FIELD)
#undef PRINT_FIELD
    for (const auto& line : trace) std::printf("TRACE %s\n", line.c_str());
}
'''


@pytest.fixture(scope="module", params=BACKENDS, ids=["CHT", "FSH"])
def executable(request, tmp_path_factory):
    folder = tmp_path_factory.mktemp("resident-courant-" + request.param.split("/")[1])
    source = folder / "courant.cpp"
    source.write_text(
        SHIMS
        + _abi_body(_read_source(request.param, ORIGINAL), "originalCourant")
        + _abi_body(
            _read_source(request.param, os.environ.get("RESIDENT_COURANT_SOURCE_REVISION")),
            "candidateCourant",
        )
        + MAIN
    )
    binary = folder / "courant"
    result = subprocess.run(
        ["g++", "-std=c++17", "-O0", "-ffp-contract=off", str(source), "-o", str(binary)],
        capture_output=True, text=True, timeout=90,
    )
    (folder / "build.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


def _run(executable, revision="candidate", *, cells=3, faces=4, turbulence=3,
         dt="0.125", target="0.9", schedule="2.5", result="normal",
         failure="none", plumbing="none", output="value"):
    args = [revision, cells, faces, turbulence, dt, target, schedule, result,
            failure, plumbing, output]
    process = subprocess.run(
        [str(executable), *map(str, args)], capture_output=True, text=True, timeout=10
    )
    assert process.returncode == 0, process.stdout + process.stderr
    lines = process.stdout.splitlines()
    _, rc, value, copies = lines[0].split()
    return {
        "raw": process.stdout,
        "rc": int(rc), "value": float.fromhex(value), "copies": int(copies),
        "error": lines[1].removeprefix("ERROR "),
        "fields": {p[1]: [float.fromhex(v) for v in p[2:]]
                   for p in map(str.split, lines) if p[0] == "FIELD"},
        "trace": [line.removeprefix("TRACE ") for line in lines if line.startswith("TRACE ")],
    }


def _expected_launches(faces, turbulence):
    names = ["recoverGasPrimitivesKernel"]
    if turbulence == 3:
        names += ["recoverSstPrimitivesKernel"]
    if faces:
        names += ["updateLegacyGasBoundaryMirrorKernel", "updateRiemannBoundaryMirrorKernel"]
    if turbulence == 3:
        names += ["applySstWallFunctionStateKernel"]
    names += ["computeGasPrimitiveGradientsKernel", "computeGasGradientLimiterKernel",
              "computeGasEddyViscosityKernel"]
    if faces:
        names += ["computeGasCourantFieldKernel"]
    if turbulence == 3:
        names += ["computeSstGradientsKernel"]
    names += ["computeGasConvectiveCourantByCellKernel", "computeGasDiffusionNumberKernel"]
    if turbulence == 3:
        names += ["computeSstStabilityNumberKernel"]
    return names


def _assert_array_reduction(run, cells, turbulence):
    fields = ["gasFluxPositivityScale", "gasDiffusionNumber"]
    if turbulence == 3:
        fields += ["sstSourceNumber"]
    maxima = [max(run["fields"][field][:cells]) for field in fields]
    expected = maxima[0]
    for value in maxima[1:]:

        if math.isnan(expected):
            expected = value
        elif not math.isnan(value):
            expected = max(expected, value)
    assert (math.isnan(run["value"]) and math.isnan(expected)) or run["value"] == expected
    reductions = [line for line in run["trace"] if line.startswith("reduce ")]
    assert len(reductions) == len(fields)


@pytest.mark.parametrize("cells,faces,turbulence,inputs", list(itertools.product(
    [1, 3], [0, 4], [0, 3], [("0.005", "0.25", "0"), ("0.125", "0.9", "2.5")]
)))
def test_successful_whole_fields_launches_and_reductions_match_immutable_body(
    executable, cells, faces, turbulence, inputs
):
    dt, target, schedule = inputs
    options = dict(cells=cells, faces=faces, turbulence=turbulence,
                   dt=dt, target=target, schedule=schedule)
    original = _run(executable, "original", **options)
    current = _run(executable, **options)
    assert original["rc"] == current["rc"] == 0
    assert current["raw"] == original["raw"]
    assert current["copies"] == 0
    assert [line.split()[1] for line in current["trace"] if line.startswith("launch ")] == \
        _expected_launches(faces, turbulence)
    _assert_array_reduction(current, cells, turbulence)


NON_ADMITTING_SCALARS = [
    (field, value)
    for field, values in (
        ("dt", ["-0.125", "0", "nan", "inf", "-inf"]),
        ("target", ["-0.9", "0", "nan", "inf", "-inf"]),
        ("schedule", ["-2.5", "nan", "inf", "-inf"]),
    )
    for value in values
]


@pytest.mark.parametrize("field,value", NON_ADMITTING_SCALARS)
@pytest.mark.parametrize("turbulence", [0, 3])
def test_numerical_scalar_inputs_reach_existing_calculation(executable, field, value, turbulence):
    options = {field: value, "turbulence": turbulence}
    original = _run(executable, "original", **options)
    assert original["rc"] == 1
    assert not original["trace"], "The immutable body must reject before launch"
    current = _run(executable, **options)
    assert current["rc"] == 0, current["raw"]
    assert current["error"] == ""
    assert current["copies"] == 0
    assert [line.split()[1] for line in current["trace"] if line.startswith("launch ")] == \
        _expected_launches(4, turbulence)
    _assert_array_reduction(current, 3, turbulence)


@pytest.mark.parametrize("result", ["large", "inf", "nan"])
@pytest.mark.parametrize("turbulence", [0, 3])
def test_large_and_nonfinite_computed_fields_are_returned_without_diagnostic_copies(
    executable, result, turbulence
):
    original = _run(executable, "original", result=result, turbulence=turbulence)
    current = _run(executable, result=result, turbulence=turbulence)
    assert original["rc"] == 1
    assert original["copies"] == 5
    assert current["rc"] == 0, current["raw"]
    assert current["copies"] == 0
    assert current["error"] == ""

    assert [line for line in current["raw"].splitlines() if line.startswith("FIELD ")] == \
        [line for line in original["raw"].splitlines() if line.startswith("FIELD ")]
    assert current["trace"] == [line for line in original["trace"] if not line.startswith("copy ")]
    if result == "large":
        assert math.isfinite(current["value"]) and current["value"] >= 0.5 / math.ulp(1.0)
    elif result == "inf":
        assert current["value"] == math.inf
    else:
        assert math.isnan(current["value"])
    _assert_array_reduction(current, 3, turbulence)


def test_null_output_is_still_a_pointer_error_before_any_launch(executable):
    original = _run(executable, "original", output="null")
    current = _run(executable, output="null")
    assert original["rc"] == current["rc"] == 1
    assert original["value"] == current["value"] == -987.5
    assert current["fields"] == original["fields"]
    assert current["trace"] == original["trace"] == []
    assert current["copies"] == 0
    assert "output" in current["error"]


@pytest.mark.parametrize("plumbing", ["owner", "tune"])
def test_resident_owner_and_tuning_errors_remain_errors(executable, plumbing):
    original = _run(executable, "original", plumbing=plumbing)
    current = _run(executable, plumbing=plumbing)
    assert original["rc"] == current["rc"] == 1
    assert current["raw"] == original["raw"]
    assert not any(line.startswith("launch ") for line in current["trace"])


@pytest.mark.parametrize("failure", _expected_launches(4, 3))
def test_cuda_launch_errors_keep_the_existing_failure_and_stop_later_work(executable, failure):
    original = _run(executable, "original", failure=failure)
    current = _run(executable, failure=failure)
    assert original["rc"] == current["rc"] == 1
    assert current["raw"] == original["raw"]
    assert failure in current["error"] and "719" in current["error"]
    expected = _expected_launches(4, 3)
    assert [line.split()[1] for line in current["trace"] if line.startswith("launch ")] == \
        expected[:expected.index(failure) + 1]
    assert not any(line.startswith("reduce ") for line in current["trace"])
    assert current["copies"] == 0
