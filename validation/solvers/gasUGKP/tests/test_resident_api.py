"""Typed resident ABI and build controls after removal of gasUGKP IPC."""
from validation.tools.paths import SOURCE_ROOT
import subprocess
import pytest
ROOT=SOURCE_ROOT
@pytest.mark.parametrize('species',[2,3])
def test_typed_api_configuration_and_species_identity(tmp_path,species):
    source=tmp_path/'probe.cpp';binary=tmp_path/'probe'
    source.write_text(r"""
#include "applications/gasUGKP/gpu/GpuBackendApi.H"
#include "common/gasTransport/GasBuildConfig.H"
#include <cassert>
#include <type_traits>
int main(){
 using namespace ugkwpGpu;
 using Configure=int(*)(void*,const GasModelConfigureArgsV1*,const char*,const char*);
 using Download=int(*)(void*,const GasSpeciesIdentityV1*,double*);
 static_assert(std::is_same<decltype(&ugkwpGpuResidentStrictConfigureGasModelV1),Configure>::value);
 static_assert(std::is_same<decltype(&ugkwpGpuResidentStrictDownloadSpeciesV1),Download>::value);
 static_assert(std::is_trivially_copyable<GasModelConfigureArgsV1>::value);
 GasSpeciesIdentityV1 a{1,ugkwp::compiledGasSpecies,123,456,789},b=a;
 assert(sameGasSpeciesIdentity(a,b));++b.speciesCount;assert(!sameGasSpeciesIdentity(a,b));
 b=a;++b.speciesOrderHash;assert(!sameGasSpeciesIdentity(a,b));
 b=a;++b.thermoHash;assert(!sameGasSpeciesIdentity(a,b));
 b=a;++b.mechanismHash;assert(!sameGasSpeciesIdentity(a,b));
 BoundaryLayerConfigV1 wall;assert(wall.version==1&&wall.quadratureCount==0&&wall.reservedAudit==0);
 assert(wall.nodes==ugkwp::gaswall::defaultWallNodes&&wall.stretch==ugkwp::gaswall::defaultWallStretch);
 assert(boundaryLayerDiagnosticScalars==25);
}
""")
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=undefined','-fno-sanitize-recover=undefined',f'-DUGKWP_GAS_SPECIES={species}','-I',str(ROOT),str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)

def test_build_preserves_absent_and_explicit_empty_species_options():
    script=(ROOT/'tool/gasUGKP/build_cuda_solver.sh').read_text()
    line=next(line for line in script.splitlines() if line.startswith('gas_species='))
    for setup,expected in [('unset UGKWP_GAS_SPECIES','2'),('UGKWP_GAS_SPECIES=""',''),('UGKWP_GAS_SPECIES=3','3')]:
        result=subprocess.run(['bash','-c',setup+'; '+line+'; printf "%s" "$gas_species"'],capture_output=True,text=True,check=True)
        assert result.stdout==expected
    assert '-DUGKWP_GAS_SPECIES="$gas_species"' in script
    assert '-DUGKWP_GAS_SPECIES=$gas_species' in script
