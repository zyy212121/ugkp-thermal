"""Independent finite-volume analytic reference and native case preparation."""
from pathlib import Path
import importlib.util
import math
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / "test/gasUGKP/mixtureTransport"

def module(name):
    path = CASE / (name + ".py")
    assert path.exists(), f"native mixtureTransport {name} must exist"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded

def test_reference_uses_exact_finite_volume_average():
    ref = module("reference")
    # Over [0, L/4], integral sin(2*pi*x/L) / width = 2/pi.
    actual = ref.species_cell_average(0.0, 0.25, 0.0, length=1.0,
        velocity=0.0, diffusivity=0.0, mean=0.5, amplitude=0.2)
    assert math.isclose(actual, 0.5 + 0.2*2/math.pi, abs_tol=1e-15)
    assert not math.isclose(actual, 0.5 + 0.2*math.sin(math.pi/4), abs_tol=1e-4)

def test_reference_advection_shift_and_diffusion_decay():
    ref = module("reference")
    kw = dict(length=2.0, velocity=0.7, diffusivity=0.04, mean=0.5, amplitude=0.2)
    t = 0.8
    actual = ref.species_cell_average(0.2, 0.3, t, **kw)
    translated = ref.species_cell_average(0.2-0.7*t, 0.3-0.7*t, 0.0, **kw)
    assert math.isclose(actual-0.5, (translated-0.5)*math.exp(-0.04*math.pi**2*t), abs_tol=1e-15)
    values = [ref.species_cell_average(i/32, (i+1)/32, t, **dict(kw, length=1.0)) for i in range(32)]
    assert math.isclose(sum(values)/32, 0.5, abs_tol=1e-15)

def test_case_generation_has_native_constant_model_and_species_fields(tmp_path):
    module("make_case")
    p = subprocess.run([sys.executable, str(CASE / "make_case.py"), "--case", str(tmp_path), "--cells", "16"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "gasMode mixtureFrozen;" in (tmp_path / "constant/gasModelProperties").read_text()
    assert "species (A B);" in (tmp_path / "constant/gasModelProperties").read_text()
    assert "gpuResidentPureGasOnly true;" in (tmp_path / "constant/schedulingProperties").read_text()
    assert "type cyclic;" in (tmp_path / "system/blockMeshDict").read_text()
    ref = module("reference")
    a = ref.read_scalar_field(tmp_path / "0/Y_A", 16)
    b = ref.read_scalar_field(tmp_path / "0/Y_B", 16)
    assert len(a) == len(b) == 16
    assert all(abs(x+y-1.0) < 2e-15 for x, y in zip(a,b))
    assert math.isclose(sum(a)/16, 0.5, abs_tol=1e-15)

def test_comparison_checks_both_species_and_conservative_bulk(tmp_path):
    maker = module("make_case")
    ref = module("reference")
    maker.create_case(tmp_path, cells=16)
    metrics = ref.compare_case(tmp_path, time=0.0)
    assert metrics["max_species_sum_error"] < 2e-15
    assert metrics["species_linf"] < 2e-15
    assert metrics["max_relative_bulk_error"] < 2e-14
    text = (tmp_path / "0/Y_B").read_text().replace("\n0.5", "\n0.6", 1)
    # Replace an actual scalar to ensure the checker is not reference-only.
    values = ref.read_scalar_field(tmp_path / "0/Y_B", 16)
    text = text.replace(format(values[0], ".17g"), "0.9", 1)
    (tmp_path / "0/Y_B").write_text(text)
    changed = ref.compare_case(tmp_path, time=0.0)
    assert changed["max_species_sum_error"] > 0.1

def test_native_mesh_has_valid_translational_periodic_pair(tmp_path):
    import shutil
    import pytest
    if not shutil.which("blockMesh"):
        pytest.skip("OpenFOAM blockMesh is required")
    maker = module("make_case")
    maker.create_case(tmp_path, cells=16)
    run = subprocess.run(["blockMesh", "-case", str(tmp_path)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    assert "nCells: 16" in run.stdout

def test_native_checker_requires_matching_backend_identity_and_completion(tmp_path):
    import pytest
    ref=module("reference")
    assert hasattr(ref,"verify_native_run"), "native results require actual configured binary identity"
    log=tmp_path/"log.gasUGKP"
    log.write_text("ordinary legacy run\nEnd\n")
    with pytest.raises(ValueError): ref.verify_native_run(log)
    identity="Shared gas model: api=1 Ns=2 mode=1 speciesOrderHash=12445161613445958710 thermoHash=17390867132586628798 mechanismHash=0"
    log.write_text(identity+"\nEnd\n")
    assert ref.verify_native_run(log)["Ns"]==2
    log.write_text(identity.replace("Ns=2","Ns=10")+"\nEnd\n")
    with pytest.raises(ValueError): ref.verify_native_run(log)
    log.write_text(identity+"\n")
    with pytest.raises(ValueError): ref.verify_native_run(log)
