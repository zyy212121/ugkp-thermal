import subprocess
from validation.tools.paths import SOURCE_ROOT


def test_native_owners_share_acoustic_query_and_schedule():
    owners = ('applications/gasUGKP/private_backend/GpuResidentStrict.cu',
              'common/thermal/GpuResidentStrict.cu')
    for base in (SOURCE_ROOT, SOURCE_ROOT / 'precision/fp32'):
        for path in owners:
            text = (base / path).read_text()
            assert text.count('#include "gasTransport/ResidentCourant.inl"') == 1
            resident = text.split('struct DeviceState\n{', 1)[1].split('\n};', 1)[0]
            assert resident.rstrip().endswith('#include "gpu/AdvanceOwnerFields.inl"')
            assert 'extern "C" int ugkwpGpuResidentStrictComputeGasCourant' not in text
    for api in ('applications/gasUGKP/gpu/GpuBackendApi.H', 'common/thermal/GpuBackendApi.H'):
        signature = (SOURCE_ROOT / api).read_text().split('ugkwpGpuResidentStrictComputeGasCourant', 1)[1].split(';', 1)[0]
        assert 'scheduleTime' in signature and 'targetMaxCo' not in signature
    for path in ('common/GpuGasCourant.cuh', 'common/GpuGasCourantReduction.cuh',
                 'common/gasTransport/ResidentCourant.inl'):
        assert not (SOURCE_ROOT / 'precision/fp32' / path).exists()
    kernel = (SOURCE_ROOT / 'common/GpuGasCourant.cuh').read_text()
    assert 'computeGasDiffusionNumberKernel' not in kernel and 'computeSstStabilityNumberKernel' not in kernel
    assert 'gasFluxPositivityScale' in kernel and 'computeGasConvectiveCourantByCellKernel' in kernel
    steady = SOURCE_ROOT / 'applications/gasUGKPSteady/private_backend/GpuSteadyBackend.cu'
    assert 'gasUGKP/private_backend/GpuResidentStrict.cu' in steady.read_text()


def test_transient_and_chmt_share_growth_and_step_caps(tmp_path):
    transient = (SOURCE_ROOT / 'common/gpu/AdvanceState.cuh').read_text()
    assert 'selectCourantTimeStep(state->dt,proposed,request->growthLimit,request->maximumDt,true)' in transient
    source = tmp_path / 'caps.cpp'
    source.write_text(r"""
#include "applications/CHMT/gpu/GasTimeStepControl.H"
#include <cassert>
int main() {
    assert(ugkwp::selectCourantTimeStep(1.,.25,1.2,10.,true)==.25);
    assert(ugkwp::selectCourantTimeStep(1.,50.,1.2,10.,true)==1.2);
    assert(ugkwp::selectCourantTimeStep(1.,50.,1.2,1.1,true)==1.1);
    assert(ugkwp::selectCourantTimeStep(1.,.25,1.2,10.,false)==1.);
    const auto fixed=chmt::selectGasMicrostep(false,1.,.75,.25,1.2,10.,true);
    assert(fixed.accepted==.75&&fixed.next==1.);
    for(double remaining:{.125,.75,2.})for(double stable:{.25,1.,1.3,50.})
    for(double maximum:{.2,1.1,10.})for(bool refreshed:{true,false}) {
        const double next=ugkwp::selectCourantTimeStep(1.,stable,1.2,maximum,refreshed);
        const auto step=chmt::selectGasMicrostep(true,1.,remaining,stable,1.2,maximum,refreshed);
        assert(step.next==next&&step.accepted==std::min(remaining,next));
    }
}
""")
    binary = tmp_path / 'caps'
    subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(SOURCE_ROOT), str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
