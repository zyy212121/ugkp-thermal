from pathlib import Path


def test_resident_bridge_always_receives_positive_particle_thermal_properties():
    source = (Path(__file__).parents[2] / "createFields.H").read_text()
    rho_block = source[source.index("dimensionedScalar particleThermalRho"):source.index("dimensionedScalar particleCp")]
    cp_block = source[source.index("dimensionedScalar particleCp"):source.index("dimensionedScalar TpMin")]
    assert 'readScalar(ugkwpProps.lookup("particleRho"))' in rho_block
    assert 'readScalar(ugkwpProps.lookup("particleCp"))' in cp_block
    assert "solveParticleTemperature" not in rho_block
    assert "solveParticleTemperature" not in cp_block
    assert ": scalar(0)" not in rho_block
    assert ": scalar(0)" not in cp_block


def test_drag_model_ids_match_the_cuda_backend_contract():
    source = (Path(__file__).parents[2] / "gpu" / "GpuDragModel.H").read_text()
    schiller = source[source.index("class GpuSchillerNaumannDragModel"):source.index("class GpuGidaspowErgunWenYuDragModel")]
    gidaspow = source[source.index("class GpuGidaspowErgunWenYuDragModel"):source.index("inline autoPtr<GpuDragModel>")]
    assert "return 1;" in schiller
    assert "return 2;" in gidaspow


def test_particle_restart_creates_the_write_time_directory():
    source = (Path(__file__).parents[2] / "gpu" / "GpuResidentStrict.H").read_text()
    block = source[source.index("label writeParticleRestartMirror"):source.index("label writeParticleRestartMirror", source.index("label writeParticleRestartMirror") + 1) if source.count("label writeParticleRestartMirror") > 1 else source.index("void writeEpsGPrevRestartMirror")]
    assert "mkDir(runTime.timePath())" in block
