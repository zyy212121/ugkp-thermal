"""Run production CUDA bodies with host stubs; GPU integration gates remain separate."""
from pathlib import Path
import re
import pytest
from test_l2_scheduling_behavior import function, compile_run, SOURCE

FIXTURES = Path(__file__).parent / "fixtures/l2_goal"

@pytest.mark.parametrize("name", ["fused_moments", "filtered_gather", "prefilled_gather", "probability_cache", "pressure_cache"])
def test_production_behavior(tmp_path, name):
    source = SOURCE.read_text() + "\n" + (SOURCE.parents[3] / "common/GpuCellLocalGather.cuh").read_text() + "\n" + (SOURCE.parents[3] / "common/CsrPersistentQueue.cuh").read_text()
    fixture = (FIXTURES / (name + ".cpp.in")).read_text()
    if name in ("filtered_gather", "prefilled_gather"):
        fixture = fixture.replace("template<int BlockThreads>\n{{gatherCellLocalRange}}",
            "template<int BlockThreads, bool IndexOnly>\n{{gatherCellLocalRangeImpl}}\n"
            "template<int BlockThreads>\n{{gatherCellLocalRange}}")
    fixture = re.sub(r"\{\{(\w+)\}\}", lambda match: function(source, match[1]), fixture)
    compile_run(tmp_path, fixture)
