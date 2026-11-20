#!/usr/bin/env python3
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

                                                                           

import shutil
import struct
import subprocess
import unittest

import pytest


ROOT = (SOURCE_ROOT / 'applications/gasUGKP')
FRONTEND = ROOT / "gpu" / "GpuResidentStrict.H"


class ParticleBinaryRestartContract(unittest.TestCase):
    def test_binary_reader_converts_actual_blocks_and_reports_short_reads(self) -> None:
        source = FRONTEND.read_text()


        required = (
            "readBinaryParticleBlock",
            "std::vector<Wire>",
            "is.gcount() != bytes",
            "binary restart chunk count",
        )
        for token in required:
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_v5_reuses_the_established_upload_contract(self) -> None:
        source = FRONTEND.read_text()
        self.assertIn("ugkwpGpuResidentStrictUploadParticleRestartMirror", source)
        self.assertNotIn("ugkwpGpuResidentStrictUploadBinaryParticleRestart", source)

    def test_ugkp_binary_restart_persists_particle_mass(self) -> None:
        source = FRONTEND.read_text()
        self.assertIn('os << "UGKP_PARTICLES_SCHEMA1_BIN "', source)
        self.assertIn(
            'writeBinaryParticleBlock(os, pm.begin() + offset, count, "pm", restartFile);',
            source,
        )
        self.assertIn(
            'readBinaryParticleBlock(is, pm.begin() + offset, count, "pm", restartFile);',
            source,
        )

    def test_restart_upload_never_forces_existing_mass_to_injection_mass(self) -> None:
        cuda = (ROOT / "private_backend/GpuResidentStrict.cu").read_text()
        self.assertNotIn("forceParticleMassToParcelMassKernel", cuda)
        self.assertNotIn("s.pm[i] = s.injectionParcelMass", cuda)

    def test_legacy_mass_and_future_injection_mass_are_distinct(self) -> None:
        wrapper = FRONTEND.read_text()
        fields = (ROOT / "createFields.H").read_text()
        self.assertIn("legacyRestartParcelMass_", wrapper)
        self.assertIn('"injectionParcelMass"', fields)
        self.assertIn('"legacyRestartParcelMass"', fields)


@pytest.fixture(scope="module")
def native_restart_probe(tmp_path_factory):
    if not shutil.which("wmake"):
        pytest.skip("OpenFOAM environment required")
    root = tmp_path_factory.mktemp("gas-restart-layout")
    (root / "Make").mkdir()
    (root / "Make/files").write_text(f"probe.C\nEXE = {root}/probe\n")
    (root / "Make/options").write_text(
        f"EXE_INC = -DUGKWP_USE_CUDA -DUGKWP_GAS_SPECIES=2 -I{ROOT} -I{ROOT}/gpu -I{SOURCE_ROOT}/common "
        "-I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\n"
        "EXE_LIBS = -lfiniteVolume -lmeshTools -lOpenFOAM\n"
    )
    (root / "probe.C").write_text(r'''
#include "gpu/GpuResidentStrict.H"
#include <iostream>

extern "C" void ugkwpGpuResidentStrictRelease(void*){}
extern "C" const char* ugkwpGpuResidentStrictLastError(){return "invalid decoded restart values";}
extern "C" int ugkwpGpuResidentStrictUploadParticleRestartMirror(void*,int count,
    const double* x,const double*,const double*,const double*,const double*,const double*,
    const double*,const double*,const double*,const double* mass,
    const int*,const int* status,const unsigned long long*,const unsigned long long* origId){
    if(count!=1||x[0]!=.125||status[0]!=1||origId[0]!=12)return 1;
    std::cout<<"decoded pm="<<mass[0]<<'\n';
    return 0;
}
class RestartProbe : public Foam::GpuResidentStrictBase {
public:
    void load(const Foam::Time& time){
        handle_=this;
        legacyRestartParcelMass_=2;
        readParticleRestartMirrorIfPresent(time);
        handle_=nullptr;
    }
};
int main(int argc,char** argv){
    using namespace Foam;
    #include "setRootCase.H"
    #include "createTime.H"
    RestartProbe{}.load(runTime);
}
''')
    result = subprocess.run(["wmake"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return root / "probe"


@pytest.mark.parametrize("layout", ["weighted-alias", "legacy-alias", "weighted-trailing",
                                   "weighted-padding", "weighted-truncated"])
def test_native_restart_reads_fields_before_applying_diagnostics(tmp_path, native_restart_probe, layout):
    (tmp_path / "system").mkdir()
    (tmp_path / "constant").mkdir()
    (tmp_path / "0").mkdir()
    (tmp_path / "system/controlDict").write_text(
        "FoamFile {version 2.0; format ascii; class dictionary; object controlDict;}\n"
        "application gasUGKP; startFrom startTime; startTime 0; stopAt endTime; endTime 0; "
        "deltaT 1e-8; writeControl timeStep; writeInterval 1;\n"
    )
    legacy = layout == "legacy-alias"
    version = 909 if legacy else 5 if layout == "weighted-alias" else 1
    data = f"UGKP_PARTICLES_SCHEMA{version}_BIN 1 0\n".encode()
    if layout == "weighted-padding":
        data += struct.pack("<I", 0)
    values = (.125, 2, 3, 4, 5, 6, 300, 8, .001)
    data += (struct.pack("<I9d2i2Q", 1, *values, 0, 1, 11, 12) if legacy
             else struct.pack("<I10d2i2Q", 1, *values, 3, 0, 1, 11, 12))
    if layout == "weighted-trailing":
        data += b"producer metadata"
    if layout == "weighted-truncated":
        data = data[:-1]
    (tmp_path / "0/gpuResidentStrictParticles.dat").write_bytes(data)
    result = subprocess.run([str(native_restart_probe), "-case", str(tmp_path)],
                            capture_output=True, text=True)
    if layout == "weighted-truncated":
        assert result.returncode != 0
        assert "required input data unavailable or invalid" in result.stderr
        assert "truncated particle field pOrigId" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"decoded pm={2 if legacy else 3}" in result.stdout


if __name__ == "__main__":
    unittest.main()
