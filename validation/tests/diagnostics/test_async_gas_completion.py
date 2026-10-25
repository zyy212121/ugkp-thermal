"""Every shared launch following an asynchronous fatal has a uniform entry gate."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_all_shared_advance_and_courant_kernels_gate_fatal_before_work():
    launches = set()
    for name in ("GpuGasAdvance.cuh", "GpuGasCourant.cuh"):
        launches.update(re.findall(r"(\w+)(?:<[^;{}]*?>)?\s*<<<", (ROOT / "common" / name).read_text()))
    launches.update(("evaluateGasBoundaryLayerKernel", "evaluateGasBoundaryLayerTeamKernel",
                     "evaluateGasBoundaryLayerTile8Kernel"))
    sources = [p.read_text() for p in (ROOT / "common").rglob("*.cuh")]
    for kernel in sorted(launches):
        pattern = re.compile(r"__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?void\s+" + kernel + r"\s*\([^)]*\)\s*\{")
        bodies = [text[match.end():] for text in sources for match in pattern.finditer(text)]
        assert len(bodies) == 1, kernel
        # A null-state check is uniform; no numerical access or per-lane return
        # may precede the block-wide barrier in blockMayProceed.
        prefix = bodies[0].split("blockMayProceed", 1)[0]
        assert len(prefix) < 180 and "threadIdx" not in prefix and "blockIdx" not in prefix, kernel
        assert "return" not in re.sub(r"if\s*\(!state\)\s*return\s*;", "", prefix), kernel


def test_mixture_frontend_publishes_clock_after_completed_public_advance():
    frontend = (ROOT / "applications/gasUGKP/diluteUgkwpFoam.C").read_text()
    assert "resident.advanceOneStep(runTime.deltaTValue(), runTime.value());\n                    runTime++;" in frontend
    backend = (ROOT / "applications/gasUGKP/private_backend/GpuResidentStrict.cu").read_text()
    public = backend.split('extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly', 1)[1]
    interval = public.split("advanceGasRequestedInterval<SharedGasTrialPolicy>", 1)[1].split("#ifndef", 1)[0]
    assert interval.index("SharedGasTrialPolicy::complete(s)") < interval.index("return result||completion")
