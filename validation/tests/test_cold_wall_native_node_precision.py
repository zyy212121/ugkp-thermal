"""Native coldWall1D precision contract; not a GPU physical validation."""
from validation.tools.paths import SOURCE_ROOT


def test_native_cold_node_storage_and_publication():
    for prefix, scalar in (("", "double"), ("precision/fp32/", "float")):
        root = SOURCE_ROOT / prefix
        resident = (root / "common/thermal/GpuResidentStrict.cu").read_text()
        state = (root / "common/thermal/ParticleState.H").read_text()
        publish = (root / "common/wall/GpuColdWallThermalCarry.cuh").read_text()
        assert f"{scalar}* pColdNodeSpecificEnthalpy" in resident
        assert f"{scalar}* compactPColdNodeSpecificEnthalpy" in resident
        assert f"coldNodeStorage*sizeof({scalar})" in resident
        assert f"{scalar} coldNodeSpecificEnthalpy[8]" in state
        assert "float cold2DNodeSpecificEnthalpy[64]" in state
        assert f"({scalar}* nodes,int count,{scalar} specificHeat)" in publish
        assert "nextCarry" not in publish
        for leaf in ("GpuColdWall1DPublish.inl", "GpuColdWall1DAdvance.inl"):
            source = (root / "common/wall" / leaf).read_text()
            assert "meanCorrection" not in source
            assert "oldCarry" not in source
        # Host boundary buffers use the existing double wire API; resident
        # copyToDevice/copyToHost performs native conversion for FP32.
        assert "const double* pColdNodeSpecificEnthalpy," in resident
    api = (SOURCE_ROOT / "common/thermal/GpuBackendApi.H").read_text()
    assert "const double* pColdNodeSpecificEnthalpy" in api
    mesh = (SOURCE_ROOT / "applications/CHMT/io/MeshIO.H").read_text()
    assert "h.coldNodeSpecificEnthalpy[j]=double(nodes[j])" in mesh
    copy = (SOURCE_ROOT / "common/GpuCellLocalThermalFields.cuh").read_text()
    assert "std::is_same<Scalar, float>::value" in copy
