"""Current maintained startup scheduling contracts, without legacy migration."""
from pathlib import Path
import re
from validation.tools.paths import SOURCE_ROOT
ROOT=SOURCE_ROOT
RESIDENTS=["common/thermal/GpuResidentStrict.cu","applications/gasUGKP/private_backend/GpuResidentStrict.cu"]
FAMILIES=["Cell","Face","Particle","Reduction","Sparse","Radiation","Thermal","Material","ColdWall"]

def test_shared_configuration_reads_each_family_once():
    text=(ROOT/"common/GpuSchedulingConfiguration.H").read_text()
    for family in FAMILIES:
        assert text.count('"gpu'+family+'BlockThreads"')==1
    for source in ["applications/CHT/readGpuGasConfiguration.H","applications/gasUGKP/readGpuGasConfiguration.H","applications/CHMT/configuration/CouplingInput.H"]:
        text=(ROOT/source).read_text()
        assert '"schedulingProperties"' in text and "readGpuSchedulingConfiguration" in text

def test_startup_geometry_does_not_race_physical_kernels():
    publication=(ROOT/"common/GpuWorkPlanPublish.cuh").read_text()
    assert "publishStartupWorkPlan" in publication and "syncDeviceState" not in publication
    assert "if (s->fixedWorkBlockTuned) return 0;" in publication
    assert "s->fixedWorkBlockTuned=1;" in publication
    for token in ["cudaEventCreate","cudaEventElapsedTime","nth_element","launchToolB1"]:
        assert token not in publication
    for source in RESIDENTS:
        text=(ROOT/source).read_text()
        assert "publishStartupWorkPlan" in text
        assert "workPlan().cellBlockThreads" in text and "workPlan().faceBlockThreads" in text
        assert "tuneFixedWorkBlockThreads" not in text

def test_l2_retains_existing_scheduler():
    text=(ROOT/"common/GpuSchedulingConfiguration.H").read_text()
    for token in ["GpuCsrLevel::L0","GpuCsrLevel::L2","GpuHeavyReductionMode::automatic","gpuCsrHeavyReductionAutoInterval"]:
        assert token in text
    for source in RESIDENTS:
        assert "runAutomaticCsrSchedule" in (ROOT/source).read_text()

def test_removed_dynamic_inlet_is_not_a_runtime_switch():
    for source in RESIDENTS+["applications/CHT/diluteUgkwpFoam.C","applications/gasUGKP/diluteUgkwpFoam.C","common/GpuSchedulingConfiguration.H"]:
        text=(ROOT/source).read_text()
        for token in ["gpuResidentDynamicInlet","gpuResidentPressureTable","gpuResidentVolumeFractionTable","ConfigureScheduledInlet"]:
            assert token not in text

def test_explicit_cht_fixture_lists_all_block_families():
    text=(ROOT/"validation/shared_boundary/fixtures/cht/constant/schedulingProperties").read_text()
    for family in FAMILIES:
        assert re.search(r"gpu"+family+r"BlockThreads\s+128\s*;",text)
