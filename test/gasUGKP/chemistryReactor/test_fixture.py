"""Host-only fixture/checker evidence; no native CUDA solver is invoked."""
import importlib.util
import json
from pathlib import Path
import subprocess
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name):
    path = HERE/(name+".py")
    assert path.is_file(), "native reactor fixture implementation must exist"
    spec = importlib.util.spec_from_file_location("reactor_fixture_"+name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generator_uses_full_pinned_reactive_model_and_periodic_fields(tmp_path):
    generate = load("make_case")
    contract = generate.create_case(tmp_path, "fine")
    assert contract["reference_case"] == "nitrogen_1200K_1atm"
    assert contract["required_build"]["UGKWP_GAS_SPECIES"] == 10
    assert contract["volume"] == 1e-6
    assert contract["chemistry_controls"]["relativeTolerance"] == 1e-7
    assert len(contract["species"]) == 10
    for name in contract["required_fields"]:
        text = (tmp_path/"0"/name).read_text()
        assert "type cyclic" in text
    assert "Tadmor" in (tmp_path/"system/fvSchemes").read_text()
    assert "HLLC" not in (tmp_path/"system/fvSchemes").read_text()
    assert "mixtureChemistry" in (tmp_path/"constant/gasModelProperties").read_text()
    assert "reaction28" in (tmp_path/"constant/h2o2.mechanism").read_text()
    assert not (tmp_path/"0/Y_A").exists()
    assert not (tmp_path/"0/Y_B").exists()


def test_default_runner_does_not_execute_native_solver(tmp_path):
    load("make_case")
    result = subprocess.run(["python", str(HERE/"run.py"), "--output", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    status = json.loads((tmp_path/"run_status.json").read_text())
    assert status["native_execution"] == "NOT_RUN"
    assert status["validation"] == "NOT_RUN"
    assert (tmp_path/"coarse/system/controlDict").exists()
    assert (tmp_path/"fine/system/controlDict").exists()


def fabricated_checker_inputs(case, variant="fine", temperature_offset=0):
    """Manufactured parser fixtures only, deliberately restricted to pytest tmpdirs."""
    generate, checker = load("make_case"), load("check_case")
    contract = generate.create_case(case, variant)
    reference = generate.reference_case()
    for state in reference["states"]:
        time = state["time"]
        if not any(abs(time-t) < 1e-14 for t in contract["check_times"]):
            continue
        directory = case/format(time, ".12g")
        directory.mkdir(exist_ok=True)
        t = state["temperature"] + temperature_offset
        density = state["density"]
        y = state["massFractions"]
        thermo = checker.thermodynamics(t, y)
        values = {"rho": ("1 -3 0 0 0 0 0",density), "T":("0 0 0 1 0 0 0",t),
                  "p": ("1 -1 -2 0 0 0 0",density*thermo["gas_constant"]*t),
                  "rhoE": ("1 -1 -2 0 0 0 0",density*thermo["specific_internal_energy"])}
        for name, (dims,value) in values.items():
            (directory/name).write_text(generate._base.field(name,dims,value))
        for name, dims in (("U","0 1 -1 0 0 0 0"),("rhoU","1 -2 -1 0 0 0 0")):
            (directory/name).write_text(generate._base.field(name,dims,"(0 0 0)",True))
        for name,value in zip(contract["species"],y):
            (directory/("Y_"+name)).write_text(generate._base.field("Y_"+name,"0 0 0 0 0 0 0",value))
    identity = contract["required_build"]
    (case/"log.gasUGKP").write_text("MANUFACTURED CHECKER TEST INPUT, NOT A NATIVE RUN\nShared gas model: api=1 Ns=10 mode=2 speciesOrderHash="+identity["speciesOrderHash"]+" thermoHash="+identity["thermoHash"]+" mechanismHash="+identity["mechanismHash"]+"\nEnd\n")
    return checker, contract


def test_checker_accepts_independent_reference_fixture(tmp_path):
    checker, _ = fabricated_checker_inputs(tmp_path)
    result = checker.compare_case(tmp_path)
    assert result["passed"]
    assert result["normalized_history_error"] < 1e-13


@pytest.mark.parametrize("corruption", ["missing_species", "wrong_identity", "sensible_energy", "species_mass", "momentum", "temperature", "missing_history"])
def test_checker_rejects_bad_native_evidence(tmp_path, corruption):
    checker, contract = fabricated_checker_inputs(tmp_path)
    last = tmp_path/format(contract["end_time"], ".12g")
    if corruption == "missing_species":
        (last/"Y_H2O").unlink()
    elif corruption == "wrong_identity":
        path=tmp_path/"log.gasUGKP"
        path.write_text(path.read_text().replace("Ns=10", "Ns=2"))
    elif corruption == "sensible_energy":
        (last/"rhoE").write_text(load("make_case")._base.field("rhoE","1 -1 -2 0 0 0 0",1e6))
    elif corruption == "species_mass":
        (last/"Y_H2").write_text(load("make_case")._base.field("Y_H2","0 0 0 0 0 0 0",0.2))
    elif corruption == "momentum":
        (last/"rhoU").write_text(load("make_case")._base.field("rhoU","1 -2 -1 0 0 0 0","(0.01 0 0)",True))
    elif corruption == "temperature":
        (last/"T").write_text(load("make_case")._base.field("T","0 0 0 1 0 0 0",1200))
    else:
        (tmp_path/"1e-06"/"T").unlink()
    result=checker.compare_case(tmp_path)
    assert not result["passed"], result
    assert result["failures"]


def test_reference_uncertainty_floor_allows_already_converged_pair(tmp_path):
    checker,_=fabricated_checker_inputs(tmp_path/"coarse","coarse")
    fabricated_checker_inputs(tmp_path/"fine","fine")
    result=checker.compare_pair(tmp_path/"coarse",tmp_path/"fine")
    assert result["passed"]
    assert result["refinement_passed"]


def test_refinement_rejects_worse_fine_history_even_when_single_case_passes(tmp_path):
    checker,_=fabricated_checker_inputs(tmp_path/"coarse","coarse")
    fabricated_checker_inputs(tmp_path/"fine","fine",temperature_offset=1e-6)
    result=checker.compare_pair(tmp_path/"coarse",tmp_path/"fine")
    assert result["coarse"]["passed"],result
    assert result["fine"]["passed"],result
    assert not result["refinement_passed"]
    assert not result["passed"]


def test_generated_native_inputs_pass_real_common_host_loaders(tmp_path):
    generator=load("make_case")
    case=tmp_path/"case"
    generator.create_case(case,"fine")
    source=tmp_path/"parse.cpp"
    source.write_text(r'''
#include "common/gasTransport/GasMechanismIO.H"
#include <cassert>
#include <string>
int main(int argc,char**argv) {
    assert(argc==2);
    std::string directory=argv[1];
    auto model=ugkwp::readGasModelProperties(directory+"/constant/gasModelProperties");
    assert(model.mode==ugkwp::GasMode::MixtureChemistry);
    assert(model.speciesNames.size()==10);
    assert(model.chemistryControls.relativeTolerance==1e-7);
    auto mechanism=ugkwp::readGasMechanismProperties<10>(ugkwp::resolveGasMechanismPath(model,directory+"/constant"),model.thermoView<10>(),model.phase);
    assert(mechanism.reactions.size()==29 && mechanism.independentRank==6);
    assert(mechanism.mechanismHash==UINT64_C(13247918587359511140));
}
''')
    executable=tmp_path/"parse"
    subprocess.run(["g++","-std=c++14","-Wall","-Wextra","-Werror","-pedantic","-I",str(ROOT),str(source),"-o",str(executable)],check=True)
    subprocess.run([str(executable),str(case)],check=True)


def test_checker_does_not_accept_shortened_contract_history(tmp_path):
    checker,_=fabricated_checker_inputs(tmp_path)
    path=tmp_path/"case_contract.json"
    contract=json.loads(path.read_text())
    contract["check_times"]=[0]
    path.write_text(json.dumps(contract))
    assert not checker.compare_case(tmp_path)["passed"]


def test_checker_rejects_changed_actual_chemistry_controls(tmp_path):
    checker,_=fabricated_checker_inputs(tmp_path)
    path=tmp_path/"constant/gasModelProperties"
    path.write_text(path.read_text().replace("relativeTolerance 9.9999999999999995e-08;", "relativeTolerance 0.1;"))
    assert not checker.compare_case(tmp_path)["passed"]


def test_refinement_requires_coarse_and_fine_roles(tmp_path):
    checker,_=fabricated_checker_inputs(tmp_path/"coarse","coarse")
    fabricated_checker_inputs(tmp_path/"fine","fine")
    assert not checker.compare_pair(tmp_path/"fine",tmp_path/"coarse")["passed"]
