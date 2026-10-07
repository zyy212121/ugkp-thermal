"""Execute the shared OpenFOAM selector parser, including all old selectors."""
import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[3]
PROBE = r'''
#include "gasTransport/GasNumericsIO.H"
#include "IStringStream.H"
#include <iostream>
#include <iterator>
int main() {
    std::string input((std::istreambuf_iterator<char>(std::cin)), {});
    Foam::IStringStream stream(input);
    Foam::dictionary schemes(stream);
    Foam::IStringStream controlsStream("robustFallback true;");
    Foam::dictionary controls(controlsStream);
    const auto result = Foam::readGasNumericsConfiguration(schemes, &controls);
    std::cout << result.gasFluxScheme << " " << result.gasReconstruction << " "
              << result.gasLimiter << " " << result.gasTimeIntegrator << " "
              << result.gasRobustFallback << std::endl;
}
'''
@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    if not os.environ.get("WM_PROJECT_DIR"):
        pytest.skip("OpenFOAM environment is required for native selector parser")
    path = tmp_path_factory.mktemp("gas_numerics_io")
    (path/"Make").mkdir()
    (path/"probe.C").write_text(PROBE)
    (path/"Make/files").write_text(f"probe.C\nEXE = {path}/probe\n")
    (path/"Make/options").write_text(f"EXE_INC = -I{ROOT}/common -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/OpenFOAM/lnInclude\nEXE_LIBS = -lOpenFOAM\n")
    result = subprocess.run(["wmake"], cwd=path, capture_output=True, text=True)
    return path/"probe", result

def dictionary(flux="HLLC", reconstruction="MUSCL", limiter="barthJespersen", time="SSPRK2"):
    momentum = "upwind" if reconstruction == "limitedLinear 1" else reconstruction
    return f'''
fluxScheme {flux}; gasLimiter {limiter};
ddtSchemes {{ default {time}; }}
gradSchemes {{ default Gauss linear; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default corrected; }}
laplacianSchemes {{ default Gauss linear corrected; }}
divSchemes
{{
 default none;
 div(phi,U) Gauss {momentum};
 div(phi,e) Gauss {reconstruction};
 div(phi,K) Gauss {reconstruction};
 div(phi,(p|rho)) Gauss {reconstruction};
 div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}}
'''

def test_all_original_numerical_selections_use_shared_parser(probe):
    binary, build = probe
    assert build.returncode == 0, build.stdout + build.stderr
    for flux_id, flux in enumerate(("Tadmor","Kurganov","HLLE","HLLC","Roe","HLLEM","HLLC-ADC","SLAU2","SLAU2.2"),1):
        for rec_id, rec in enumerate(("upwind","MUSCL","limitedLinear 1")):
            for limiter_id, limiter in enumerate(("none","barthJespersen","venkatakrishnan")):
                for time_id, time in enumerate(("Euler","SSPRK2","SSPRK3"),1):
                    run = subprocess.run([str(binary)], input=dictionary(flux,rec,limiter,time), capture_output=True, text=True)
                    assert run.returncode == 0, run.stdout+run.stderr
                    assert run.stdout.strip() == f"{flux_id} {rec_id} {limiter_id} {time_id} 1"

@pytest.mark.parametrize("old,new,message", [
    ("fluxScheme HLLC;", "", "must define top-level fluxScheme"),
    ("fluxScheme HLLC;", "fluxScheme bogus;", "Unsupported fvSchemes fluxScheme"),
    ("gasLimiter barthJespersen;", "gasLimiter bogus;", "Unsupported gas limiter"),
    ("default SSPRK2;", "default CrankNicolson;", "Unsupported ddtSchemes/default"),
    ("default Gauss linear;", "default Gauss cubic;", "Gauss linear"),
])
def test_invalid_selectors_keep_original_diagnostics(probe, old, new, message):
    binary, build = probe
    assert build.returncode == 0, build.stdout+build.stderr
    run = subprocess.run([str(binary)], input=dictionary().replace(old,new), capture_output=True, text=True)
    assert run.returncode != 0
    assert message in run.stdout+run.stderr
