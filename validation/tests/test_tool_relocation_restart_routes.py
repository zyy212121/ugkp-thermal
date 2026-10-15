"""Fresh-process witnesses for relocated CHT/FSH restart-tool routes."""
from validation.tools.paths import SOURCE_ROOT

import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest


sys.path.insert(0, str(SOURCE_ROOT / "tool/restart"))
import thermal_particle_restart_current as current

FSH6 = "UGKP_FSH_PARTICLES_SCHEMA6_BIN"
MARKERS = [FSH6, "UGKP_THERMAL_PARTICLES_SCHEMA7_T32_BIN",
           "UGKP_THERMAL_PARTICLES_SCHEMA7_T64_BIN", current.FSH_MARKER,
           current.CHT_MARKER]


def historical_specs():
    path = SOURCE_ROOT / "tool/FSH/fsh_particle_restart_schema6.py"
    spec = importlib.util.spec_from_file_location("relocated_legacy_fsh_specs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._field_specs(6)


def write_input(path, marker):
    specs = historical_specs() if marker == FSH6 else current.field_specs(marker)
    data = {name: np.ones((2,) if width == 1 else (2, width), dtype=dtype)
            for name, dtype, width in specs}
    data.update(cell=np.array([1, 2], dtype="<i4"),
                status=np.ones(2, dtype="<i4"),
                rng=np.array([101, 202], dtype="<u8"),
                orig_id=np.array([2**53 + 1, 2**53 + 2], dtype="<u8"),
                pm=np.array([2., 3.]), pStuck=np.zeros(2, dtype="u1"),
                pStuckFaceId=np.full(2, -1, dtype="<i4"),
                pDepositionArea=np.zeros(2), pContactPeakFraction=np.zeros(2))
    with path.open("wb") as stream:
        stream.write(f"{marker} 2 2\n".encode())
        stream.write(struct.pack("<I", 2))
        for name, dtype, _ in specs:
            np.asarray(data[name], dtype=dtype).tofile(stream)
    return {name: np.asarray(data[name], dtype=dtype) for name, dtype, _ in specs}


def child_env():
    # Do not let warmed pytest imports or an inherited FSH PYTHONPATH mask the route.
    return {**os.environ, "PYTHONPATH": str(SOURCE_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1"}


@pytest.mark.parametrize("marker", MARKERS)
@pytest.mark.parametrize("operation", ["metadata", "read"])
def test_cht_reader_resolves_legacy_and_current_inputs_in_fresh_process(tmp_path, marker, operation):
    source = tmp_path / "source.dat"
    original = write_input(source, marker)
    output = tmp_path / "read.npz"
    program = r'''
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import ugkp_particle_restart as reader
if sys.argv[4] == "metadata":
    print(json.dumps(reader.input_metadata(sys.argv[2])))
else:
    np.savez(sys.argv[3], **reader.collect_restart(sys.argv[2]))
'''
    result = subprocess.run([sys.executable, "-c", program,
                             str(SOURCE_ROOT / "tool/CHT"), str(source),
                             str(output), operation], cwd=tmp_path, env=child_env(),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    if operation == "metadata":
        metadata = json.loads(result.stdout)
        assert metadata["source_format"] == marker
        assert metadata["defaulted_saved_velocity"] == (marker == FSH6)
    else:
        with np.load(output) as restored:
            for name, dtype, _ in current.field_specs(current.CHT_MARKER):
                expected = (original[name].astype(dtype) if name in original
                            else np.full(restored[name].shape, -1 if name == "pStuckFaceId" else 0, dtype=dtype))
                np.testing.assert_array_equal(restored[name], expected)


@pytest.mark.parametrize("marker", MARKERS)
def test_cht_cli_routes_legacy_and_current_owner_in_fresh_process(tmp_path, marker):
    source, output = tmp_path / "source.dat", tmp_path / "output.dat"
    original = write_input(source, marker)
    result = subprocess.run([sys.executable, str(SOURCE_ROOT / "tool/CHT/refine_particle_cells.py"),
                             str(source), str(output), "--cell", "1", "--factor", "1"],
                            cwd=tmp_path, env=child_env(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    owner = current.FSH_MARKER if marker.startswith("UGKP_FSH_") else current.CHT_MARKER
    assert manifest["format"] == owner
    assert manifest["source_format"] == marker
    restored = {name: np.concatenate([chunk[name] for _, chunk in current.iter_chunks(output)])
                for name, _, _ in current.field_specs(owner)}
    for name, dtype, _ in current.field_specs(owner):
        expected = (original[name].astype(dtype) if name in original
                    else np.full(restored[name].shape, -1 if name == "pStuckFaceId" else 0, dtype=dtype))
        np.testing.assert_array_equal(restored[name], expected)
