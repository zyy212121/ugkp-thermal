from pathlib import Path
import os, subprocess, time, json

root = Path(__file__).resolve().parents[3]
out = root / "work/nearwall-fix-20261020"
out.mkdir(parents=True, exist_ok=True)
tests = [
    "gas_wall/test_wall_geometry.py",
    "gas_wall/test_wall_matching_averages.py",
    "gas_wall/test_wall_matching_state.py",
    "gas_wall/test_wall_matching_extrapolation.py",
    "gas_wall/test_wall_matching_rank.py",
    "gas_wall/test_jacobian_point_cache.py",
    "gas_wall/test_seed_point_reuse.py",
    "gas_wall/test_wall_jacobian_scale.py",
    "gas_wall/test_sst_low_k_initialization.py",
    "gas_wall/test_wall_interval_reuse.py",
    "gas_wall/test_wall_publication_work.py",
    "gas_wall/test_wall_volume_coupling.py",
    "gas_transport/test_boundary_layer_volume_projection.py",
    "gas_transport/test_boundary_layer_full_strain.py",
    "gas_transport/test_boundary_layer_tangential_input.py",
    "gas_transport/test_boundary_layer_stage_gradients.py",
    "gas_transport/test_boundary_layer_source_stability.py",
    "gas_transport/test_boundary_layer_blowing_transport.py",
    "gas_transport/test_boundary_layer_blowing_face_frames.py",
    "gas_transport/test_internal_transport_interpolation.py",
    "gas_transport/test_mixture_contact_flux.py",
    "gas_transport/test_mixture_slau2_flux.py",
]
command = ["python3", "-m", "pytest", "-q", *("validation/tests/" + p for p in tests),
           "--junitxml=" + str(out / "targeted.xml")]
env = dict(os.environ, NVCC="/usr/local/cuda-13.1/bin/nvcc")
start = time.monotonic()
with (out / "targeted.log").open("w") as log:
    result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
meta = {"command": command, "exit_code": result.returncode, "seconds": time.monotonic() - start}
(out / "targeted-run.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta))
print((out / "targeted.log").read_text()[-5000:])
raise SystemExit(result.returncode)
