"""Read canonical numerical tables and retain all actual reaction channels."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import struct
import subprocess
import pytest

ROOT = SOURCE_ROOT

def hash_bytes(h, values):
    for byte in values:
        h = ((h ^ byte)*1099511628211) & ((1<<64)-1)
    return h

def species_hash(names):
    h = 14695981039346656037
    for name in names:
        h = hash_bytes(h, name.encode()+b"\0")
    return h

def mechanism(names=("A","B"), product_coeff=1., reverse=False):
    order_hash = species_hash(names)
    h = order_hash
    def scalar(x):
        nonlocal h
        h = hash_bytes(h, struct.pack("<d", float(x) if x else 0.))
    def byte(x):
        nonlocal h
        h = hash_bytes(h, [x])
    for x in (101325,1,-1,1): scalar(x)
    for x in (0,int(reverse),0): byte(x)
    for x in (0,1,0,0,0,0,0,1,0,0,0,0): scalar(x)
    byte(0)
    for x in (1,0,1,1,1,product_coeff,0): scalar(x)
    return f'''schemaVersion 1;
units siMolar;
phase test;
species ({' '.join(names)});
speciesOrderHash {order_hash};
mechanismHash {h};
referencePressure 101325;
independentRank 1;
stoichiometricBasis (-1 1);
reactions
{{
 reaction0
 {{
  type elementary; reversible {'true' if reverse else 'false'}; duplicate false; sourceIndex 0;
  reactants (0 1); products (1 {product_coeff});
  highRate (1 0 0); lowRate (0 0 0);
  defaultEfficiency 1; efficiencies ();
  troe (0 0 0 0); troeHasT2 false;
 }}
}}
'''

PROBE = r'''
#include "common/gasTransport/GasMechanismIO.H"
#include <iostream>
#include <iterator>
int main(int argc,char** argv) {
    if(argc==3){ugkwp::GasModelConfiguration model;model.mode=ugkwp::GasMode::MixtureChemistry;model.mechanism=argv[1];std::cout<<ugkwp::resolveGasMechanismPath(model,argv[2]);return 0;}
    const std::string text((std::istreambuf_iterator<char>(std::cin)), {});
    ugkwp::SpeciesThermoData<double> species[2];
    const double coefficients[6]={1040,0,0,1040,0,0};
    const double atoms[2]={1,1};
    for(int s=0;s<2;++s){species[s].molarMass=.028;species[s].minTemperature=100;species[s].maxTemperature=4000;species[s].referencePressure=101325;species[s].coefficientOffset=3*s;}
    ugkwp::SpeciesThermoView<double,2> thermo;
    thermo.species=species;thermo.coefficients=coefficients;thermo.coefficientCount=6;
    thermo.elementCount=1;thermo.elementComposition=atoms;
    thermo.speciesOrderHash=ugkwp::gasModelIoDetail::speciesIdentity({"A","B"});
    try {
        const auto loaded=ugkwp::parseGasMechanismProperties<2>(text,thermo,"test");
        const auto view=loaded.mechanismView<2>();
        std::cout << view.reactionCount << " " << view.independentRank << " " << view.speciesOrderHash << "\n";
    } catch(const std::exception& error){std::cerr<<error.what();return 2;}
}
'''
@pytest.fixture(scope="module")
def build(tmp_path_factory):
    directory=tmp_path_factory.mktemp("mechanism_io")
    (directory/"probe.cpp").write_text(PROBE)
    result=subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror","-I",str(ROOT),str(directory/"probe.cpp"),"-o",str(directory/"probe")],capture_output=True,text=True)
    return result,directory/"probe"

def parse(build,text):
    compiled,binary=build
    assert compiled.returncode==0,compiled.stderr
    return subprocess.run([str(binary)],input=text,capture_output=True,text=True)

def test_canonical_complete_mechanism_loads_and_preserves_identity(build):
    p=parse(build,mechanism())
    assert p.returncode==0,p.stderr
    assert p.stdout.split()==["1","1",str(species_hash(("A","B")))]

@pytest.mark.parametrize("old,new,diagnostic",[
    ("units siMolar", "units cgs", "units"),
    ("phase test", "phase other", "phase"),
    ("type elementary", "type PLog", "reaction type"),
    ("highRate (1 0 0)", "highRate (2 0 0)", "mechanismHash"),
    ("efficiencies ()", "efficiencies (0)", "pairs"),
    ("sourceIndex 0", "sourceIndex 8", "sourceIndex"),
])
def test_metadata_changes_do_not_hide_real_unsupported_rate_or_missing_pairs(build,old,new,diagnostic):
    text=mechanism().replace(old,new)
    if diagnostic == "pairs":
        text=text.replace("type elementary", "type thirdBody")
    p=parse(build,text)
    if diagnostic in ("reaction type","pairs"):
        assert p.returncode==2 and diagnostic in p.stderr,p.stderr
    else:
        assert p.returncode==0,p.stderr

def test_mass_element_imbalance_is_not_a_parser_proof(build):
    p=parse(build,mechanism(product_coeff=2))
    assert p.returncode==0,p.stderr

def test_parser_does_not_fabricate_missing_entropy_data(build):
    p=parse(build,mechanism(reverse=True))
    assert p.returncode==0,p.stderr

def test_raw_views_use_actual_positional_indices_without_hash_proof(build):
    p=parse(build,mechanism(names=("B","A")))
    assert p.returncode==0,p.stderr


def test_mechanism_relative_paths_use_case_constant_directory(build):
    compiled, binary = build
    assert compiled.returncode == 0, compiled.stderr
    relative = subprocess.run([str(binary), "kinetics/model.mechanism", "/case/constant/"], capture_output=True, text=True)
    absolute = subprocess.run([str(binary), "/shared/model.mechanism", "/case/constant"], capture_output=True, text=True)
    assert relative.stdout == "/case/constant/kinetics/model.mechanism"
    assert absolute.stdout == "/shared/model.mechanism"

@pytest.mark.parametrize('old,new', [
    ('schemaVersion 1;', 'schemaVersion 42; futureKey ignored;'),
    ('phase test;', 'phase unrelated; sourceSha256 descriptive;'),
    ('units siMolar;', 'units descriptive;'),
    ('highRate (1 0 0)', 'highRate (2 0 0 999)'),
    ('sourceIndex 0', 'sourceIndex 8'),
    ('stoichiometricBasis (-1 1)', 'stoichiometricBasis (-1 1 777)'),
])
def test_t07_mechanism_reads_required_tables_without_proof_metadata(build, old, new):
    p = parse(build, mechanism().replace(old, new))
    assert p.returncode == 0, p.stderr
    assert p.stdout.split()[:2] == ['1', '1']

@pytest.mark.parametrize('old,new,diagnostic', [
    ('highRate (1 0 0)', 'highRate (1 0)', 'highRate'),
    ('reactants (0 1)', 'reactants (2 1)', 'species'),
    ('type elementary', 'type PLog', 'reaction type'),
])
def test_t07_mechanism_missing_selected_inputs_still_fail(build, old, new, diagnostic):
    p = parse(build, mechanism().replace(old, new))
    assert p.returncode == 2 and diagnostic in p.stderr
