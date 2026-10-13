"""Host behavior of the one shared constant/gasModelProperties parser."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[3]
PROBE = r'''
#include "common/gasTransport/GasModelIO.H"
#include <iostream>
#include <iterator>
int main(int argc, char**) {
    std::string text((std::istreambuf_iterator<char>(std::cin)), {});
    try {
        auto model = ugkwp::parseGasModelProperties(text, argc < 2);
        std::cout << static_cast<int>(model.mode) << " " << model.speciesNames.size()
                  << " " << model.speciesOrderHash << " " << model.thermoHash;
        if (!model.speciesNames.empty()) {
            auto view = model.thermoView<2>();
            std::cout << " " << view.coefficientCount << " " << model.diffusionCoefficients[0]
                      << " " << view.species[0].hasEntropyReference;
        }
        if (model.mode == ugkwp::GasMode::MixtureChemistry)
            std::cout << " " << model.chemistryControls.relativeTolerance << " "
                      << model.chemistryControls.maximumSteps << " " << model.chemistryControls.thermo.maximumIterations;
        std::cout << "\n";
    } catch (const std::exception& error) { std::cerr << error.what(); return 2; }
}
'''

@pytest.fixture(scope="session")
def build(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gas_model_io")
    source = directory / "probe.cpp"
    source.write_text(PROBE)
    binary = directory / "probe"
    result = subprocess.run(["g++", "-std=c++14", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), str(source), "-o", str(binary)], capture_output=True, text=True)
    return result, binary

def parse(build, text, absent=False):
    result, binary = build
    assert result.returncode == 0, "Shared host gas model parser must compile:\n" + result.stderr
    return subprocess.run([str(binary)] + (["absent"] if absent else []), input=text, capture_output=True, text=True)

def mixture():
    return '''FoamFile { version 2.0; format ascii; class dictionary; object gasModelProperties; }
gasMode mixtureFrozen;
species (A B);
speciesThermo
{
 A { model linearCp; molarMass 0.028; minTemperature 100; maxTemperature 4000; coefficients (1040 0 0); }
 B { model linearCp; molarMass 0.028; minTemperature 100; maxTemperature 4000; coefficients (1040 0 0); }
}
diffusion { model constant; coefficients (0.002 0.002); turbulentSchmidt 0.7; }
'''

def test_absent_dictionary_keeps_no_species_legacy(build):
    p = parse(build, "", absent=True)
    assert p.returncode == 0, p.stderr
    assert p.stdout.split()[:2] == ["0", "0"]

def test_explicit_single_keeps_no_species_legacy(build):
    p = parse(build, "gasMode single;")
    assert p.returncode == 0, p.stderr
    assert p.stdout.split()[:2] == ["0", "0"]

def test_linear_frozen_loads_exact_order_and_optional_entropy(build):
    p = parse(build, mixture())
    assert p.returncode == 0, p.stderr
    fields = p.stdout.split()
    assert fields[:2] == ["1", "2"]
    assert int(fields[2]) != 0 and int(fields[3]) != 0
    assert fields[4:] == ["6", "0.002", "0"]

@pytest.mark.parametrize("old,new,diagnostic", [
    ("mixtureFrozen", "typo", "gasMode"),
    ("coefficients (0.002 0.002)", "coefficients (0.002)", "diffusion"),
    ("molarMass 0.028", "molarMass nan", "finite"),
    ("model linearCp", "model mystery", "thermo model"),
])
def test_actual_missing_or_unimplemented_models_fail_at_read(build, old, new, diagnostic):
    p = parse(build, mixture().replace(old, new))
    assert p.returncode == 2
    assert diagnostic in p.stderr, p.stderr

@pytest.mark.parametrize("extra,mode", [("enableChemistry true;", "1"), ("gasMode single;", "0")])
def test_unused_metadata_and_last_dictionary_value_do_not_refuse(build,extra,mode):
    p=parse(build,mixture()+extra)
    assert p.returncode==0,p.stderr
    assert p.stdout.split()[0]==mode


def test_present_empty_dictionary_does_not_silently_default(build):
    p = parse(build, "")
    assert p.returncode == 2 and "gasMode" in p.stderr

def test_chemistry_needs_explicit_mechanism_and_phase(build):
    p = parse(build, mixture().replace("mixtureFrozen", "mixtureChemistry"))
    assert p.returncode == 2 and "mechanism" in p.stderr

def test_hashes_preserve_semantics_and_species_order(build):
    one = parse(build, mixture())
    commented = parse(build, "// comment\n" + mixture().replace("gasMode", "/* mode */ gasMode"))
    reordered = parse(build, mixture().replace("species (A B)", "species (B A)"))
    changed = parse(build, mixture().replace("1040 0 0", "1100 0 0"))
    assert all(x.returncode == 0 for x in (one, commented, reordered, changed))
    assert one.stdout == commented.stdout
    assert one.stdout.split()[2] != reordered.stdout.split()[2]
    assert one.stdout.split()[3] != changed.stdout.split()[3]

def test_disjoint_metadata_intervals_do_not_reject_readable_polynomials(build):
    text = mixture().replace(' B { model linearCp; molarMass 0.028; minTemperature 100; maxTemperature 4000;',
        ' B { model linearCp; molarMass 0.028; minTemperature 5000; maxTemperature 6000;')
    p = parse(build, text)
    assert p.returncode == 0,p.stderr

def test_nasa_thermo_reference_pressure_is_explicit(build):
    text = mixture().replace('model linearCp;', 'model NASA7; midTemperature 1000;')
    text = text.replace('coefficients (1040 0 0);', 'coefficients (3.5 0 0 0 0 0 0 3.5 0 0 0 0 0 0);')
    missing = parse(build, text)
    assert missing.returncode == 2 and "referencePressure" in missing.stderr
    valid = parse(build, text.replace('model NASA7;', 'model NASA7; referencePressure 101325;'))
    assert valid.returncode == 0, valid.stderr
    assert valid.stdout.split()[4] == "28"

def chemistry():
    return mixture().replace("mixtureFrozen", "mixtureChemistry") + 'mechanism "model.mechanism"; phase gas;\n'

def test_chemistry_controls_are_loaded_from_shared_types(build):
    p = parse(build, chemistry() + 'chemistryControls { relativeTolerance 2e-7; maximumSteps 300; thermo { maximumIterations 123; } }')
    assert p.returncode == 0, p.stderr
    assert p.stdout.split()[-3:] == ["2e-07", "300", "123"]

@pytest.mark.parametrize("controls,diagnostic", [
    ("relativeTolerance -1;", "chemistryControls"),
    ("maximumSteps 0;", "chemistryControls"),
    ("maximumSteps 2.5;", "integer"),
    ("minimumStep 1; maximumStep 0.1;", "chemistryControls"),
    ("unknownStep 1;", "unknown"),
])
def test_read_chemistry_controls_without_range_or_unused_key_admission(build,controls,diagnostic):
    p = parse(build, chemistry() + 'chemistryControls {' + controls + '}')
    if diagnostic == "integer":
        assert p.returncode == 2 and diagnostic in p.stderr,p.stderr
    else:
        assert p.returncode == 0,p.stderr

def test_frozen_mode_ignores_unused_chemistry_controls(build):
    p = parse(build, mixture() + 'chemistryControls { maximumSteps 1; }')
    assert p.returncode == 0,p.stderr

@pytest.mark.parametrize('old,new', [
    ('gasMode mixtureFrozen;', 'schemaVersion 99; unusedMetadata { future yes; } gasMode mixtureFrozen;'),
    ('coefficients (1040 0 0)', 'coefficients (1040 0 0 123 456)'),
    ('minTemperature 100; maxTemperature 4000;', 'minTemperature -100; maxTemperature -200;'),
    ('coefficients (0.002 0.002)', 'coefficients (-0.002 0.002 123)'),
    ('turbulentSchmidt 0.7', 'turbulentSchmidt -0.7'),
    ('molarMass 0.028', 'molarMass 1e-310'),
])
def test_t07_read_actual_thermo_coefficients_without_metadata_admission(build, old, new):
    p = parse(build, mixture().replace(old, new))
    assert p.returncode == 0, p.stderr
    assert p.stdout.split()[:2] == ['1', '2']

@pytest.mark.parametrize('text,diagnostic', [
    (mixture().replace('coefficients (1040 0 0)', 'coefficients (1040 0)'), 'coefficient'),
    (mixture().replace('model linearCp', 'model unknownThermo'), 'thermo model'),
    (mixture().replace('coefficients (0.002 0.002)', 'coefficients (0.002)'), 'diffusion'),
])
def test_t07_actual_required_model_input_still_fails(build, text, diagnostic):
    p = parse(build, text)
    assert p.returncode == 2 and diagnostic in p.stderr


@pytest.mark.parametrize("old,new", [('species (A B)', 'species (A A)'), ('coefficients (0.002 0.002)', 'coefficients (-0.002 0.002)'), ('coefficients (1040 0 0)', 'coefficients (1 0 0)'), ('turbulentSchmidt 0.7', 'turbulentSchmidt 0'), ('minTemperature 100', 'minTemperature 5000')])
def test_thermo_parser_retains_finite_nonstandard_inputs(build,old,new):
    p=parse(build,mixture().replace(old,new))
    assert p.returncode==0,p.stderr

def test_integer_chemistry_budgets_keep_actual_signed_value(build):
    p=parse(build,chemistry()+"chemistryControls {maximumSteps -3;}")
    assert p.returncode==0,p.stderr
    assert p.stdout.split()[-2]=="-3"
