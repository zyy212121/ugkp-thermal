#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "GpuResidentStrict.constant-response-time.patch").read_text(encoding="utf-8")
ADDED = "\n".join(
    line[1:]
    for line in PATCH.splitlines()
    if line.startswith("+") and not line.startswith("+++")
)
FRONTEND = (ROOT / "gpu/GpuDragModel.H").read_text(encoding="utf-8")
DEVICE_MODEL = (ROOT / "private_backend/GpuDragModels.cuh").read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    for stale_snapshot in (
        "gpu/GpuBackendApi.H",
        "gpu/GpuBackendClient.C",
        "gpu/GpuBackendProtocol.H",
        "gpu/GpuResidentStrict.H",
        "private_backend/GpuBackendServer.C",
        "private_backend/GpuResidentStrict.cu",
    ):
        require(not (ROOT / stale_snapshot).exists(), f"stale common-source snapshot remains: {stale_snapshot}")
    require("GpuConstantResponseTimeDragModel" in FRONTEND, "frontend model missing")
    require('modelName_("constantResponseTime")' in FRONTEND, "model name missing")
    require("parameters_[0] = particleResponseTime" in FRONTEND, "drag time missing")
    require("parameters_[1] = particleThermalResponseTime" in FRONTEND, "thermal time missing")
    require("parameters_[2] = gasHeatCapacity" in FRONTEND, "gas capacity missing")
    require("ConstantResponseTimeDeviceDrag" in DEVICE_MODEL, "device drag adapter missing")
    require(ADDED.count("ConstantResponseTimeDeviceDrag") == 2, "both drag launches required")
    require(ADDED.count("s.hostDragModel == 2") == 2, "both heat paths required")
    require("const double gasCapacity = epsG*rhoG*gasHeatCapacity;" in ADDED, "gas capacity basis missing")
    require("gasCv/clampMin(gasHeatCapacity, OfSmall)" in ADDED, "gas energy conversion missing")
    require("(void)modelId;" in ADDED and "return true;" in ADDED, "drag activity override missing")
    require("dragParameter0 <= 0.0" in ADDED, "adapter validation missing")
    print("constant-response-time adapter contract: PASS")


if __name__ == "__main__":
    main()
