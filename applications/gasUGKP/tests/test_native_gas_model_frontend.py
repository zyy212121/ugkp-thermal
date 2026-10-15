"""Actual OpenFOAM frontend initialization, without a GPU backend substitute."""
import importlib.util
import os
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[3]
CASE=ROOT/"test/gasUGKP/mixtureTransport"
PROBE=r'''
#include "fvCFD.H"
#include "gpu/SharedGasModelInput.H"
using namespace Foam;
int main(int argc,char**argv){
 #include "setRootCase.H"
 #include "createTime.H"
 #include "createMesh.H"
 SharedGasModelInput model(runTime,mesh);
 volVectorField U(IOobject("U",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
 volScalarField p(IOobject("p",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
 volScalarField T(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
 volScalarField rho(IOobject("rho",runTime.timeName(),mesh,IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE),p/dimensionedScalar("RT",dimVelocity*dimVelocity,1.));
 volVectorField rhoU(IOobject("rhoU",runTime.timeName(),mesh,IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE),rho*U);
 volScalarField rhoE(IOobject("rhoE",runTime.timeName(),mesh,IOobject::READ_IF_PRESENT,IOobject::AUTO_WRITE),p);
 const char* selectedWall=std::getenv("GAS_MODEL_TEST_WALL");
 const bool moving=std::getenv("GAS_MODEL_TEST_MOVING")!=nullptr;
 model.validateNumerics(1,1,0,2,selectedWall?3:0,false,selectedWall?std::atoi(selectedWall):0);
 // Native gas frontend has no moving-mesh argument. Exercise the shared ALE
 // capability directly, with Euler so wall-family scope causes the rejection.
 if(moving){
  ugkwp::GasCapabilityRequest request;request.mode=model.model().mode;
  request.fluxScheme=1;request.reconstruction=1;request.timeIntegrator=1;
  request.turbulenceModel=3;request.sstWallTreatment=std::atoi(selectedWall);request.movingGeometry=true;
  const auto result=ugkwp::validateGasCapabilities(request);
  if(!result)FatalErrorInFunction<<result.message<<exit(FatalError);
 }
 model.initialiseFields(runTime,mesh,rho,rhoU,rhoE,U,p,T);
 if(std::getenv("GAS_MODEL_WRITE_RESTART")){runTime.setTime(1.,1);runTime.writeNow();}
 Info<<"MODEL_INITIALIZED "<<model.model().speciesNames.size()<<" "<<rho[0]<<" "<<rhoE[0]<<" mode="<<static_cast<int>(model.model().mode)<<nl;
 return 0;
}
'''
def build_probe(root,species):
    if not os.environ.get("WM_PROJECT_DIR"):
        pytest.skip("OpenFOAM environment required")
    (root/"Make").mkdir();(root/"probe.C").write_text(PROBE)
    (root/"Make/files").write_text(f"probe.C\nEXE = {root}/probe\n")
    (root/"Make/options").write_text(f"EXE_INC = -DUGKWP_GAS_SPECIES={species} -I{ROOT}/applications/gasUGKP -I{ROOT}/common -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = -lfiniteVolume -lmeshTools -lOpenFOAM\n")
    build=subprocess.run(["wmake"],cwd=root,capture_output=True,text=True)
    return root/"probe",build

@pytest.fixture(scope="module")
def native_probe(tmp_path_factory):
    return build_probe(tmp_path_factory.mktemp("gas_frontend"),2)

@pytest.fixture(scope="module")
def native_reacting_probe(tmp_path_factory):
    return build_probe(tmp_path_factory.mktemp("gas_reacting_frontend"),10)

def make_case(path):
    spec=importlib.util.spec_from_file_location("native_mixture_case",CASE/"make_case.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.create_case(path,cells=8)
    mesh=subprocess.run(["blockMesh","-case",str(path)],capture_output=True,text=True)
    assert mesh.returncode==0,mesh.stdout+mesh.stderr

def test_native_model_reads_species_and_formation_energy(native_probe,tmp_path):
    binary,build=native_probe
    assert build.returncode==0,build.stdout+build.stderr
    make_case(tmp_path)
    result=subprocess.run([str(binary),"-case",str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert "MODEL_INITIALIZED 2" in result.stdout

def test_native_model_uses_pt_y_over_redundant_density(native_probe,tmp_path):
    binary,build=native_probe
    assert build.returncode==0,build.stdout+build.stderr
    make_case(tmp_path)
    import re
    path=tmp_path/"0/rho"
    path.write_text(re.sub(r"internalField uniform [^;]+;","internalField uniform 99;",path.read_text()))
    result=subprocess.run([str(binary),"-case",str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert "MODEL_INITIALIZED 2 1.137" in result.stdout

def test_native_frontend_reaches_model_configuration_before_initial_upload():
    wrapper=(ROOT/"applications/gasUGKP/gpu/GpuResidentStrict.H").read_text()
    solver=(ROOT/"applications/gasUGKP/diluteUgkwpFoam.C").read_text()
    configuration=(ROOT/"applications/gasUGKP/readGpuGasConfiguration.H").read_text()
    fields=(ROOT/"applications/gasUGKP/createFields.H").read_text()
    assert "SharedGasModelInput sharedGasModel(runTime, mesh);" in configuration
    assert "sharedGasModel.initialiseFields" in fields
    body=wrapper[wrapper.index("void initialiseGasBase"):wrapper.index("public:\n    void configureSst")]
    assert body.index("configureResident(handle_)") < body.index("uploadGasOnlyFields") < body.index("uploadInitialSpecies(handle_)")
    assert "resident.downloadSharedGasSpecies(sharedGasModel, rho)" in solver


def test_native_restart_round_trip_uses_full_precision_and_same_identity(native_probe,tmp_path):
    binary,build=native_probe
    assert build.returncode==0,build.stdout+build.stderr
    make_case(tmp_path)
    control=tmp_path/"system/controlDict"
    control.write_text(control.read_text().replace("writePrecision 17;","writePrecision 6;"))
    first=subprocess.run([str(binary),"-case",str(tmp_path)],env=dict(os.environ,GAS_MODEL_WRITE_RESTART="1"),capture_output=True,text=True)
    assert first.returncode==0,first.stdout+first.stderr
    assert (tmp_path/"1/gasModelIdentity").is_file()
    control.write_text(control.read_text().replace("startTime 0;","startTime 1;"))
    restart=subprocess.run([str(binary),"-case",str(tmp_path)],capture_output=True,text=True)
    assert restart.returncode==0,restart.stdout+restart.stderr
    identity=tmp_path/"1/gasModelIdentity"
    identity.write_text(identity.read_text().replace("thermoHash", "originalThermoHash",1)+ '\nthermoHash "123";\n')
    changed=subprocess.run([str(binary),"-case",str(tmp_path)],capture_output=True,text=True)
    assert changed.returncode==0,changed.stdout+changed.stderr
    identity.unlink()
    missing=subprocess.run([str(binary),"-case",str(tmp_path)],capture_output=True,text=True)
    assert missing.returncode==0,missing.stdout+missing.stderr


def test_native_reacting_frontend_reads_pinned_mechanism_and_formation_energy(native_reacting_probe,tmp_path):
    binary,build=native_reacting_probe
    assert build.returncode==0,build.stdout+build.stderr
    spec=importlib.util.spec_from_file_location("native_reactor_frontend",ROOT/"test/gasUGKP/chemistryReactor/make_case.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.create_case(tmp_path)
    mesh=subprocess.run(["blockMesh","-case",str(tmp_path)],capture_output=True,text=True)
    assert mesh.returncode==0,mesh.stdout+mesh.stderr
    result=subprocess.run([str(binary),"-case",str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert "MODEL_INITIALIZED 10" in result.stdout


def test_native_frontend_rejects_species_shape_mismatch(native_reacting_probe,tmp_path):
    binary,build=native_reacting_probe
    assert build.returncode==0,build.stdout+build.stderr
    make_case(tmp_path)
    result=subprocess.run([str(binary),"-case",str(tmp_path)],capture_output=True,text=True)
    assert result.returncode>0
    assert "missing species storage for compiled gas model" in result.stdout+result.stderr


def test_native_legacy_needs_no_mixture_fields_or_identity(native_probe,tmp_path):
    binary,build=native_probe
    assert build.returncode==0,build.stdout+build.stderr
    make_case(tmp_path)
    for name in ("constant/gasModelProperties","0/Y_A","0/Y_B"):
        (tmp_path/name).unlink()
    result=subprocess.run([str(binary),"-case",str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert "MODEL_INITIALIZED 0" in result.stdout
    assert not (tmp_path/"0/gasModelIdentity").exists()


@pytest.mark.parametrize("mode,wall,moving,accepted", [
    ("frozen","0",False,True),
    ("frozen","1",False,True),
    ("frozen","1",True,True),
    ("chemistry","0",False,True),
    ("chemistry","1",False,True),
])
def test_native_mixture_sst_gate_uses_configured_wall_treatment(request,tmp_path,mode,wall,moving,accepted):
    binary,build=request.getfixturevalue("native_reacting_probe" if mode=="chemistry" else "native_probe")
    assert build.returncode==0,build.stdout+build.stderr
    if mode=="chemistry":
        spec=importlib.util.spec_from_file_location("native_wall_reactor_case",ROOT/"test/gasUGKP/chemistryReactor/make_case.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.create_case(tmp_path)
        mesh=subprocess.run(["blockMesh","-case",str(tmp_path)],capture_output=True,text=True)
        assert mesh.returncode==0,mesh.stdout+mesh.stderr
    else:
        make_case(tmp_path)
    env=dict(os.environ,GAS_MODEL_TEST_WALL=wall)
    if moving:env["GAS_MODEL_TEST_MOVING"]="1"
    else:env.pop("GAS_MODEL_TEST_MOVING",None)
    result=subprocess.run([str(binary),"-case",str(tmp_path)],env=env,capture_output=True,text=True)
    assert (result.returncode==0)==accepted,result.stdout+result.stderr
    if not accepted:
        assert "wallFunction supports only fixed impermeable mixtureFrozen" in result.stdout+result.stderr


def test_native_same_ns10_binary_accepts_reacting_frozen_and_legacy_modes(native_reacting_probe,tmp_path):
    import re
    binary,build=native_reacting_probe
    assert build.returncode==0,build.stdout+build.stderr
    spec=importlib.util.spec_from_file_location("same_binary_reactor",ROOT/"test/gasUGKP/chemistryReactor/make_case.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.create_case(tmp_path)
    mesh=subprocess.run(["blockMesh","-case",str(tmp_path)],capture_output=True,text=True)
    assert mesh.returncode==0,mesh.stdout+mesh.stderr
    model=tmp_path/"constant/gasModelProperties"
    chemistry=model.read_text()
    frozen=re.sub(r"chemistryControls\s*\{[^}]*\}","",chemistry)
    frozen=re.sub(r"(?m)^\s*(mechanism|phase)\s+[^;]+;", "", frozen)
    frozen=frozen.replace("mixtureChemistry","mixtureFrozen")
    for content,mode in [(chemistry,2),(frozen,1),(None,0)]:
        if content is None:
            model.unlink()
        else:
            model.write_text(content)
        result=subprocess.run([str(binary),"-case",str(tmp_path)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        assert f"mode={mode}" in result.stdout


@pytest.mark.parametrize("model_content", [None,"gasMode single;"])
def test_native_explicit_legacy_selection_ignores_old_identity(native_probe,tmp_path,model_content):
    binary,build=native_probe
    assert build.returncode==0,build.stdout+build.stderr
    make_case(tmp_path)
    first=subprocess.run([str(binary),"-case",str(tmp_path)],env=dict(os.environ,GAS_MODEL_WRITE_RESTART="1"),capture_output=True,text=True)
    assert first.returncode==0,first.stdout+first.stderr
    control=tmp_path/"system/controlDict"
    control.write_text(control.read_text().replace("startTime 0;","startTime 1;"))
    model=tmp_path/"constant/gasModelProperties"
    if model_content is None:
        model.unlink()
    else:
        header=model.read_text().split("\ngasMode",1)[0]
        model.write_text(header+model_content+"\n")
    result=subprocess.run([str(binary),"-case",str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert "MODEL_INITIALIZED 0" in result.stdout


def test_frontend_and_actual_backend_share_reordered_indices_and_rate_outputs(tmp_path):
    """Real frontend parser + real adapter allocation/copy, using host CUDA memory."""
    from test_shared_gas_adapter import ADAPTER_PROBE
    from test_gas_model_io import mixture
    from test_gas_mechanism_io import mechanism
    from test_gas_species_mapping import reordered
    if not os.environ.get('WM_PROJECT_DIR'):pytest.skip('OpenFOAM environment required')
    builddir=tmp_path/'build';builddir.mkdir();(builddir/'Make').mkdir()
    source=ADAPTER_PROBE.split('int main(){',1)[0]+r'''
#include "fvCFD.H"
#include "gpu/SharedGasModelInput.H"
#include "common/chemistry/GasRates.H"
#include <iomanip>
extern "C" const char* ugkwpGpuResidentStrictLastError(){return last.c_str();}
using namespace Foam;
int main(int argc,char**argv){
 #include "setRootCase.H"
 #include "createTime.H"
 #include "createMesh.H"
 SharedGasModelInput input(runTime,mesh);
 DeviceState state,device;state.deviceState=&device;
 input.configureResident(&state);
 auto& g=state.gasSpecies;double c[]={2,7},production[]={-999,-999},progress[]={-999};
 ugkwp::ChemistryStatus status;
 assert(ugkwp::evaluateGasRates(300.,c,g.thermo,g.mechanism,production,progress,status));
 std::cout<<std::setprecision(17)<<"MAPPED "<<g.mechanism.reactants[0].species<<" "<<g.mechanism.products[0].species
 <<" "<<g.mechanism.efficiencies[0].species<<" "<<g.mechanism.stoichiometricBasis[0]<<" "<<g.mechanism.stoichiometricBasis[1]
 <<" "<<production[0]<<" "<<production[1]<<" "<<progress[0]
 <<" "<<ugkwp::speciesCp(0,300.,g.thermo)<<" "<<ugkwp::speciesCp(1,300.,g.thermo)<<"\n";
 releaseSharedGasSpecies(g);assert(!allocations);
}
'''
    (builddir/'probe.C').write_text(source)
    (builddir/'Make/files').write_text(f'probe.C\nEXE = {builddir}/probe\n')
    (builddir/'Make/options').write_text(f'EXE_INC = -std=c++17 -DUGKWP_GAS_SPECIES=2 -I{ROOT} -I{ROOT}/applications/gasUGKP -I{ROOT}/common -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = -lfiniteVolume -lmeshTools -lOpenFOAM\n')
    built=subprocess.run(['wmake'],cwd=builddir,capture_output=True,text=True)
    assert built.returncode==0,built.stdout+built.stderr
    text=mechanism().replace('type elementary','type thirdBody').replace('efficiencies ()','efficiencies (0 2)')
    model=mixture().replace('gasMode mixtureFrozen;','gasMode mixtureChemistry; mechanism test.mechanism; phase test;').replace('coefficients (1040 0 0); }\n}', 'coefficients (1200 0 0); }\n}')
    outputs=[]
    for index,chemistry in enumerate((text,reordered(text))):
        case=tmp_path/f'case{index}';make_case(case)
        (case/'constant/gasModelProperties').write_text(model)
        (case/'constant/test.mechanism').write_text(chemistry)
        result=subprocess.run([str(builddir/'probe'),'-case',str(case)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        mapped=next(line for line in result.stdout.splitlines() if line.startswith('MAPPED '))
        outputs.append(mapped)
    assert outputs==['MAPPED 0 1 0 -1 1 -22 22 22 1040 1200']*2
