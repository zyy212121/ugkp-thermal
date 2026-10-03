"""Run production CUDA bodies with host stubs; GPU integration gates remain separate."""
from pathlib import Path
import re
import pytest
from test_l2_scheduling_behavior import function, compile_run, SOURCE

FIXTURES = Path(__file__).parent / "fixtures/l2_goal"

@pytest.mark.parametrize("name", ["fused_moments", "filtered_gather", "prefilled_gather", "probability_cache", "pressure_cache"])
def test_production_behavior(tmp_path, name):
    source = SOURCE.read_text() + "\n" + (SOURCE.parents[3] / "common/GpuCellLocalGather.cuh").read_text() + "\n" + (SOURCE.parents[3] / "common/CsrPersistentQueue.cuh").read_text()
    source += "\n" + (SOURCE.parents[3] / "common/GpuCellLocalPrimary.cuh").read_text()
    fixture = (FIXTURES / (name + ".cpp.in")).read_text()
    if name in ("filtered_gather", "prefilled_gather"):
        fixture = fixture.replace("template<int BlockThreads>\n{{gatherCellLocalRange}}",
            "template<int BlockThreads, bool IndexOnly>\n{{gatherCellLocalRangeImpl}}\n"
            "template<int BlockThreads>\n{{gatherCellLocalRange}}")
    if name == "fused_moments":
        macros = "\n#define __forceinline__ inline\n#define GPU_OPERATOR_REAL double\n#define GPU_OPERATOR_R(x) (x)\n#define GPU_OPERATOR_THERMAL 0\n"
        atomic = '#include "' + str(SOURCE.parents[3] / "common/operators/accumulateParticleMomentsAtomicKernel.cuh") + '"\n'
        moments = "\n#define GPU_MOMENT_REAL double\n#define GPU_MOMENT_R(x) (x)\n#define GPU_MOMENT_THERMAL 0\n#define GPU_MOMENT_GATHER 1\n"
        moments += '#include "' + str(SOURCE.parents[3] / "common/GpuParticleMoments.cuh") + '"\n'
        if re.search(r'^__global__\s+void accumulateParticleMomentsAtomicKernel\s*\(', SOURCE.read_text(), re.M):
            atomic = '{{accumulateParticleMomentsAtomicKernel}}'
        fixture = fixture.replace("{{accumulateParticleMomentsAtomicKernel}}", macros + atomic + moments)
        fixture = fixture.replace("template<bool HeavyReductionEnabled, bool GatherSurvivors = false>\n{{accumulateParticleMomentsSegmentedKernel}}", "")
        fixture = fixture.replace("{{accumulateCsrHeavyMomentTask}}", "")

    if name == "pressure_cache":
        common = SOURCE.parents[3] / "common"
        closure = '\nusing PressureReal=double; using PressureTime=double;\n'
        closure += '#include "' + str(common / 'GpuPressureUnsortedAlgebra.cuh') + '"\n'
        closure += '#include "' + str(common / 'GpuPressureParticleUpdate.cuh') + '"\n'
        closure += '#include "' + str(common / 'GpuPressureCellTraversal.cuh') + '"\n'
        fixture = fixture.replace('{{applyCollisionalPressureProjectionOneParticle}}',
            closure + '{{applyCollisionalPressureProjectionOneParticle}}')
        fixture = fixture.replace('template<bool SplitDirectory, bool CompactParticles=false>\n{{applyCollisionalPressureProjectionKernel}}',
            '#include "' + str(common / 'GpuPressureAnalyticAdapter.cuh') + '"\n'
            'template<bool SplitDirectory, bool CompactParticles=false>\n{{applyCollisionalPressureProjectionKernel}}')
    fixture = re.sub(r"\{\{(\w+)\}\}", lambda match: function(source, match[1]), fixture)
    compile_run(tmp_path, fixture)
