"""Execute the production membership helper with instrumented scratch reads."""
import subprocess
from validation.tools.paths import SOURCE_ROOT


def membership_source():
    text = (SOURCE_ROOT / "common/GpuPairedParticleHeat.cuh").read_text()
    return text[text.index("__device__ __forceinline__ bool pairedParticleHeatMember"):
                text.index("struct PairedParticleHeatCandidate")]


def test_contact_disabled_does_not_read_scratch(tmp_path):
    source = r"""
#include <cassert>
#define __device__
#define __forceinline__ inline
namespace Foam { namespace gpuThermal { constexpr int particleWallMobile=0; } }
struct TrackedByte {
 unsigned char value=0; mutable int reads=0;
 unsigned char operator[](int) const { ++reads; return value; }
};
struct DeviceState {
 int particleStuckModelConfigured=0;
 TrackedByte pStuck,particleContactHeatEpoch;
};
""" + membership_source() + r"""
int main() {
 for(int configured : {0,1}) for(int state : {0,1,2,3}) for(int epoch : {0,255}) {
  DeviceState s; s.particleStuckModelConfigured=configured;
  s.pStuck.value=state;s.particleContactHeatEpoch.value=epoch;
  const bool member=pairedParticleHeatMember(s,0);
  assert(member == bool(configured && (state || epoch)));
  assert(s.pStuck.reads == configured);
  assert(s.particleContactHeatEpoch.reads == (configured && state==0));
 }
}
"""
    cpp=tmp_path/"membership.cpp"
    cpp.write_text("#include <initializer_list>\n"+source)
    exe=tmp_path/"membership"
    subprocess.run(["g++","-std=c++17","-O2",str(cpp),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
