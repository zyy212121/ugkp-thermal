#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (ROOT / "GpuResidentStrict.constant-response-time.patch").read_text(
    encoding="utf-8"
)
AXIAL_PATCH = (ROOT / "GpuResidentStrict.axial-fluctuation.patch").read_text(
    encoding="utf-8"
)
FRONTEND = (ROOT / "gpu/GpuDragModel.H").read_text(encoding="utf-8")
DEVICE_MODEL = (ROOT / "private_backend/GpuDragModels.cuh").read_text(
    encoding="utf-8"
)


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
        require(
            not (ROOT / stale_snapshot).exists(),
            f"stale common-source snapshot remains: {stale_snapshot}",
        )

    require("GpuConstantResponseTimeDragModel" in FRONTEND, "frontend model missing")
    require('modelName_("constantResponseTime")' in FRONTEND, "model name missing")
    require("parameters_[0] = particleResponseTime" in FRONTEND, "drag time missing")
    require(
        "parameters_[1] = particleThermalResponseTime" in FRONTEND,
        "thermal time missing",
    )
    require("parameters_[2] = gasHeatCapacity" in FRONTEND, "gas capacity missing")
    require(
        "ConstantResponseTimeDeviceDrag" in DEVICE_MODEL,
        "device drag adapter missing",
    )
    require(PATCH.count("ConstantResponseTimeDeviceDrag") == 2, "both launches required")
    require(PATCH.count("s.hostDragModel == 2") == 2, "both heat paths required")
    require("dragParameter0 <= 0.0" in PATCH, "adapter validation missing")
    require("particleFluctuationDimensions_ < 3" in AXIAL_PATCH, "axial z clamp missing")
    require("particleFluctuationDimensions_ < 2" in AXIAL_PATCH, "axial y clamp missing")
    print("constant-response-time adapter contract: PASS")


if __name__ == "__main__":
    main()
