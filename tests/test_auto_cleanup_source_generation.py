"""Host-only generation checks; these do not compile or execute CUDA."""
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/auto_cleanup"


@pytest.mark.parametrize("app,bits", [
    ("gasUGKP", 64), ("FSH", 64), ("CHT", 64), ("CHT", 32),
])
def test_generate_cleanup_source_without_cuda(tmp_path, app, bits):
    result = subprocess.run(
        [sys.executable, str(FIXTURES / "behavior.py"), str(ROOT),
         str(tmp_path), app, str(bits), "--generate-only"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "build.log").exists()
    assert not (tmp_path / "auto_cleanup").exists()
    generated = (tmp_path / "auto_cleanup.cu").read_text()
    frozen = json.loads((FIXTURES / "cleanup_baseline.json").read_text())
    assert frozen["ordered_face"].replace(
        "pressureDeltaFromLimitedFaces", "baseline_pressureDeltaFromLimitedFaces",
    ) in generated
    assert "PASS actual automatic low-high-low-empty" in generated

    if app == "gasUGKP":
        assert "baseline_relaxColdWall" not in generated
        return

    old_cold = frozen[f"cold{bits}"].replace(
        "relaxColdWall1DParticlesToResidentGasKernel", "baseline_relaxColdWall",
    )
    if app == "FSH":
        old_cold = old_cold.replace("s.pContactAge[", "s.pTheta[")
    assert old_cold in generated
    # Mechanical age must use the app's real alias: CHT pContactAge, FSH pTheta.
    folder = "private_backend" if app == "FSH" else "gpu"
    production = (ROOT / "applications" / app / folder / "GpuResidentStrict.cu").read_text()
    age_field = "pTheta" if app == "FSH" else "pContactAge"
    assert f"#define GPU_CONTACT_AGE(s, i) s.{age_field}[i]" in production
    assert "GPU_CONTACT_AGE(contactState, 0) = GpuTime(0.5)*duration;" in generated
    assert "v->pDepositionArea[0] = finiteContact ? 0.0f : 1.7e-8f;" in generated
    assert 'ck(activeDt == coldDt, "frozen cold oracle requires full active dt");' in generated
    assert 'ck(intrinsicArea > GPU_R(0.0), "frozen cold oracle requires positive area");' in generated
    assert "PASS positive-area full-dt cold-wall outer kernel bitwise" in generated
    assert "test_cold_wall_contact_routes.py" in generated

    # Compile the exact generated host-side setup, using real field types and
    # the production area curve. This checks the oracle domain, not heat flow.
    scope = generated[generated.index(" const GpuTime coldDt ="):]
    scope = scope[:scope.index(" for(int step=0;")]
    fields = [age_field, "pContactDuration", "pContactMaximumArea",
              "pContactPeakFraction", "pDepositionArea", "pColdFrozenArea"]
    declarations = []
    for name in fields:
        field_type = re.search(rf"^\s*(\w+)\*\s+{name}\s*=", production, re.M)
        assert field_type is not None, name
        declarations.append(f"{field_type[1]} {name}[1]{{}};")
    probe = tmp_path / "oracle_domain.cpp"
    probe.write_text(
        '#include "GpuFiniteWallContact.H"\n#include <cstdlib>\n'
        '#include <initializer_list>\n'
        f"#define GPU_CONTACT_AGE(s, i) s.{age_field}[i]\n"
        "struct DeviceState {" + "\n".join(declarations) + "};\n"
        "void ck(bool condition, const char*) { if (!condition) std::abort(); }\n"
        "int main() { DeviceState a{}, b{}; DeviceState *s=&a, *r=&b;\n"
        "for (auto v : {s, r}) { v->pContactDuration[0]=8e-6; "
        "v->pContactMaximumArea[0]=2.2e-8; v->pContactPeakFraction[0]=.42; }\n"
        "for (int state : {int(Foam::gpuThermal::particleWallTransientDeposit), "
        "int(Foam::gpuThermal::particleWallDeposited)}) {\n" + scope + "}\n}\n"
    )
    executable = tmp_path / "oracle_domain"
    compile_result = subprocess.run(
        ["g++", "-std=c++17", f"-DUGKWP_GPU_REAL_BITS={bits}",
         "-I", str(ROOT / "common"), "-I", str(ROOT / "common/wall"),
         str(probe), "-o", str(executable)],
        capture_output=True, text=True, timeout=30,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    run_result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
