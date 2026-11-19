import struct
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MAGIC = 0x55474B50444F4D31


@pytest.mark.parametrize("bits", [32, 64])
def test_production_cache_io(tmp_path, bits):
    folder = ROOT / "common/gpu"
    codec = (folder / "RadiationWarmStart.cuh").read_text()
    reader = codec[codec.index("template<class Saved>"):]
    (tmp_path / "RadiationWarmStart.H").write_bytes((folder / "RadiationWarmStart.H").read_bytes())
    real = "float" if bits == 32 else "double"
    (tmp_path / "RadiationPrecision.H").write_text(
        "namespace ugkp { namespace gpu { using RadiationReal = " + real + "; }}")
    cpp = tmp_path / "cache.cpp"
    cpp.write_text('''#include "RadiationWarmStart.H"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
namespace ugkp { namespace gpu {
''' + reader + '''
int main(int argc, char** argv) {
    using namespace ugkp::gpu;
    ReadWarmStart result;
    std::string error;
    if (!readWarmStartFile(argv[1], 2, 3, 41, result, error)) return 1;
    if (!result.initialized) return result.values.empty() ? 0 : 2;
    if (result.values != std::vector<double>{0,1,2,3,4,5}) return 2;
    return result.metadataChanged ? 3 : 0;
}
''')
    binary, cache = tmp_path / "test", tmp_path / "cache.bin"
    subprocess.run(["g++", "-std=c++17", str(cpp), "-o", str(binary)], check=True)

    def read(header, payload=b""):
        cache.write_bytes(struct.pack("=7Q", *header) + payload)
        return subprocess.run([binary, cache]).returncode

    assert subprocess.run([binary, cache]).returncode == 1
    width = bits // 8
    for encoded in (4, 8):
        payload = struct.pack("=6" + ("f" if encoded == 4 else "d"), *range(6))
        header = [MAGIC, 2, encoded, 2, 3, 41, 1]
        assert read(header, payload) == (0 if width == encoded else 3)
        assert read([MAGIC, 99, encoded, 2, 3, 77, 1], payload) == 3
        assert read(header, payload[:-1]) == 1
        assert read(header, payload + b"x") == 1
        assert read([MAGIC, 2, encoded, 3, 3, 41, 1], payload) == 1
    assert read([MAGIC, 2, width, 2, 3, 41, 0]) == 0
    assert read([MAGIC, 2, 99, 2, 3, 41, 0]) == 1


def test_application_hooks():
    cht = (ROOT / "applications/CHT/thermal/GpuSolidThermalCoupler.C").read_text()
    chmt = (ROOT / "applications/CHMT/io/CoupledRadiation.C").read_text()
    writer = (ROOT / "applications/CHMT/io/AcceptedOutputWriter.H").read_text()
    for source in (cht, chmt):
        assert "->readWarmStart" in source
        assert "->freezeWarmStart" in source
        assert "->acceptParentSolve()" in source
    assert "radiation->freezeWarmStart" in writer
    assert 'staging+"/domWarmStart.bin"' in writer
