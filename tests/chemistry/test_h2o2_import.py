"""Pinned full h2o2 import: units, channel identity, reproducibility and ABI."""
from pathlib import Path
import importlib.util
import json
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "common/chemistry/mechanisms"


def importer():
    path = DATA / "import_h2o2.py"
    assert path.is_file(), "the reproducible pinned mechanism importer must exist"
    spec = importlib.util.spec_from_file_location("h2o2_import", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pinned_complete_import():
    m = importer().load_mechanism(DATA / "h2o2-cantera-3.1.0.yaml")
    assert m["species"] == ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR", "N2"]
    assert m["elements"] == ["O", "H", "Ar", "N"]
    assert len(m["reactions"]) == 29
    assert m["independentRank"] == 6
    assert all(r["reversible"] for r in m["reactions"])
    assert [i for i, r in enumerate(m["reactions"]) if r["duplicate"]] == [23, 24, 25, 26, 27, 28]
    assert [r["sourceIndex"] for r in m["reactions"]] == list(range(29))
    assert m["reactions"][0]["highRate"] == pytest.approx([1.2e5, -1, 0])
    assert m["reactions"][2]["highRate"] == pytest.approx([.0387, 2.7, 6260*4.184])
    assert m["reactions"][21]["lowRate"] == pytest.approx([2.3e6, -.9, -1700*4.184])
    assert m["reactions"][21]["type"] == "Troe"
    assert m["reactions"][21]["troe"] == [0.7346, 94.0, 1756.0, 5182.0, True]
    # Specific colliders are retained on both sides, so order is not reduced.
    assert m["reactions"][6]["reactants"] == [[1, 1.0], [3, 2.0]]
    assert m["reactions"][6]["products"] == [[3, 1.0], [6, 1.0]]
    assert m["reactions"][6]["highRate"][0] == pytest.approx(2.08e7)
    assert all(len(s["coefficients"]) == 14 for s in m["thermo"])
    assert m["thermo"][5]["molarMass"] == pytest.approx(0.018015)
    # Every basis column conserves elemental molar counts exactly.
    for e in range(4):
        for j in range(6):
            assert sum(m["thermo"][s]["atoms"][e]*m["stoichiometricBasis"][s*6+j] for s in range(10)) == 0


def test_import_rejects_changed_reference(tmp_path):
    mod = importer()
    path = tmp_path / "changed.yaml"
    path.write_bytes((DATA / "h2o2-cantera-3.1.0.yaml").read_bytes()+b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        mod.load_mechanism(path)


def test_generated_files_are_reproducible():
    importer()
    subprocess.run(["python", str(DATA / "import_h2o2.py"), "--check"], check=True, cwd=ROOT)
    manifest = json.loads((DATA / "h2o2.manifest.json").read_text())
    assert manifest["reference"]["version"] == "3.1.0"
    assert manifest["reference"]["phase"] == "ohmech"
    assert manifest["reference"]["sha256"] == "0efc6c52862741a29e0c29b65d979c7d8cb409db5282bca83b9c5437b3d8c8d4"
    assert manifest["species_count"] == 10
    assert manifest["reaction_count"] == 29
    import hashlib
    assert "independent_reference" in manifest
    reference = manifest["independent_reference"]
    assert reference["gibbs_reference_pressure_pa"] == 101325
    assert reference["sample_count"] == 63
    assert hashlib.sha256((DATA/reference["file"]).read_bytes()).hexdigest() == reference["sha256"]


@pytest.mark.parametrize("scalar", ["float", "double"])
def test_generated_header_owns_valid_views(tmp_path, scalar):
    importer()
    source = tmp_path / "check.cpp"
    source.write_text(r'''
#include "common/chemistry/mechanisms/GeneratedH2O2.H"
#include "common/gasTransport/GasModelIO.H"
#include <cassert>
#include <type_traits>
int main(int argc,char**argv) {
    using R=SCALAR;
    ugkwp::GeneratedH2O2<R> owner;
    auto tv=owner.thermoView(); auto mv=owner.mechanismView();
    assert(tv.speciesCount==10 && tv.coefficientCount==140 && tv.elementCount==4);
    assert(mv.reactionCount==29 && mv.independentRank==6);
    assert(mv.speciesOrderHash==tv.speciesOrderHash);
    assert(ugkwp::validateSpeciesThermoView(tv)==ugkwp::ThermoStatus::Success);
    auto copy=owner;
    assert(copy.thermoView().species!=tv.species);
    assert(copy.mechanismView().reactions!=mv.reactions);
    assert(mv.reactions[21].lowRate.activationEnergy<0);
    auto parsed=ugkwp::readGasModelProperties(argv[1]);
    assert(parsed.speciesOrderHash==tv.speciesOrderHash);
    assert(parsed.thermoHash==tv.thermoHash);
}
'''.replace("SCALAR", scalar))
    binary = tmp_path / "check"
    subprocess.run(["g++", "-std=c++14", "-Wall", "-Wextra", "-pedantic", "-I", str(ROOT), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary), str(DATA / "h2o2.gasModelProperties")], check=True)


def test_every_imported_channel_matches_independent_cantera_reference():
    """A missing collider, wrong A conversion, sign, duplicate or NASA range fails."""
    import math
    mod = importer()
    m = mod.load_mechanism(DATA / "h2o2-cantera-3.1.0.yaml")
    reference = json.loads((DATA / "h2o2.cantera-reference.json").read_text())
    assert reference["cantera_version"] == "3.1.0"
    assert reference["source_sha256"] == mod.SOURCE_SHA256
    assert len(reference["samples"]) == 63
    R = 8.31446261815324
    for sample in reference["samples"]:
        t, c = sample["temperature"], sample["concentrations"]
        gibbs = []
        for s in m["thermo"]:
            a = s["coefficients"][:7] if t <= s["midTemperature"] else s["coefficients"][7:]
            h = R*t*(a[0]+a[1]*t/2+a[2]*t*t/3+a[3]*t**3/4+a[4]*t**4/5+a[5]/t)
            entropy = R*(a[0]*math.log(t)+a[1]*t+a[2]*t*t/2+a[3]*t**3/3+a[4]*t**4/4+a[6])
            gibbs.append(h-t*entropy)
        assert gibbs == pytest.approx(sample["gibbsMolar"], rel=3e-13, abs=5e-10)
        production = [0.0]*10
        for i, reaction in enumerate(m["reactions"]):
            def k(rate):
                return rate[0]*t**rate[1]*math.exp(-rate[2]/(R*t))
            effective = k(reaction["highRate"])
            if reaction["type"] != "elementary":
                efficiencies = [reaction["defaultEfficiency"]]*10
                for index, efficiency in reaction["efficiencies"]:
                    efficiencies[index] = efficiency
                collider = sum(efficiencies[s]*c[s] for s in range(10))
                if reaction["type"] == "thirdBody":
                    effective *= collider
                else:
                    pr = k(reaction["lowRate"])*collider/effective
                    factor = 1.0
                    if reaction["type"] == "Troe":
                        alpha, t3, t1, t2, has_t2 = reaction["troe"]
                        fcent = (1-alpha)*math.exp(-t/t3)+alpha*math.exp(-t/t1)
                        if has_t2:
                            fcent += math.exp(-t2/t)
                        logfc = math.log10(fcent)
                        shift = math.log10(pr)-.4-.67*logfc
                        broadening = shift/(.75-1.27*logfc-.14*shift)
                        factor = 10**(logfc/(1+broadening*broadening))
                    effective *= pr/(1+pr)*factor
            forward = effective*math.prod(c[s]**n for s, n in reaction["reactants"])
            delta_g = sum(gibbs[s]*n for s, n in reaction["products"])-sum(gibbs[s]*n for s, n in reaction["reactants"])
            delta_n = sum(n for _, n in reaction["products"])-sum(n for _, n in reaction["reactants"])
            equilibrium = math.exp(-delta_g/(R*t))*(101325/(R*t))**delta_n
            reverse = effective/equilibrium*math.prod(c[s]**n for s, n in reaction["products"])
            assert forward == pytest.approx(sample["forwardRatesOfProgress"][i], rel=3e-12, abs=1e-100), (sample["name"], t, i, "forward")
            assert reverse == pytest.approx(sample["reverseRatesOfProgress"][i], rel=3e-12, abs=1e-100), (sample["name"], t, i, "reverse")
            for s, n in reaction["reactants"]:
                production[s] -= n*(forward-reverse)
            for s, n in reaction["products"]:
                production[s] += n*(forward-reverse)
        assert production == pytest.approx(sample["netProductionRates"], rel=1e-10, abs=2e-8)


@pytest.fixture(scope="module")
def rate_runner(tmp_path_factory):
    directory = tmp_path_factory.mktemp("full_h2o2_rates")
    source = directory / "rates.cpp"
    source.write_text(r'''
#include "common/chemistry/mechanisms/GeneratedH2O2.H"
#include "common/chemistry/GasRates.H"
#include "common/chemistry/GasJacobian.H"
#include <iostream>
#include <iomanip>
int main() {
    ugkwp::GeneratedH2O2<double> owner;
    auto t=owner.thermoView(); auto m=owner.mechanismView();
    auto status=ugkwp::validateGasMechanismView(t,m);
    if(!status){std::cerr<<"model invalid "<<int(status.code)<<'\n';return 1;}
    double T,c[10],w[10],q[29],j[100],d[10];
    std::cout<<std::setprecision(17);
    while(std::cin>>T) {
        for(auto& x:c)std::cin>>x;
        if(!ugkwp::evaluateGasRates(T,c,t,m,w,q,status)){
            std::cerr<<"rates failure "<<int(status.code)<<" reaction "<<status.reaction<<'\n';return 2;
        }
        if(!ugkwp::evaluateGasRateDerivatives(T,c,t,m,j,d,status)){
            std::cerr<<"derivatives failure "<<int(status.code)<<" reaction "<<status.reaction<<'\n';return 3;
        }
        for(auto x:w)std::cout<<x<<' ';for(auto x:q)std::cout<<x<<' ';
        for(auto x:j)std::cout<<x<<' ';for(auto x:d)std::cout<<x<<' ';std::cout<<'\n';
    }
}
''')
    binary = directory / "rates"
    subprocess.run(["g++", "-O2", "-std=c++14", "-I", str(ROOT), str(source), "-o", str(binary)], check=True)
    def run(states):
        data = "\n".join(" ".join(format(v, ".17g") for v in [t]+list(c)) for t, c in states)+"\n"
        result = subprocess.run([str(binary)], input=data, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        rows = [list(map(float, line.split())) for line in result.stdout.splitlines()]
        assert len(rows) == len(states)
        assert all(len(row) == 149 for row in rows)
        return rows
    return run


def test_cpp_rates_all_29_channels_against_63_cantera_states(rate_runner):
    reference = json.loads((DATA / "h2o2.cantera-reference.json").read_text())["samples"]
    rows = rate_runner([(s["temperature"], s["concentrations"]) for s in reference])
    for sample, row in zip(reference, rows):
        expected = sample["netProductionRates"]
        scale = max(abs(v) for v in expected)
        assert row[:10] == pytest.approx(expected, rel=2e-10, abs=max(1e-100, scale*2e-13)), sample
        for i, (qf, qr) in enumerate(zip(sample["forwardRatesOfProgress"], sample["reverseRatesOfProgress"])):
            assert row[10+i] == pytest.approx(qf-qr, rel=2e-10, abs=max(1e-100, (qf+qr)*2e-13)), (sample["name"], sample["temperature"], sample["pressure"], i)
        import math
        assert all(math.isfinite(v) for v in row[39:]), sample


@pytest.mark.parametrize("mixture", ["all_species", "zero_radicals", "peroxide"])
def test_cpp_derivatives_finite_at_zero_radicals_and_match_one_sided_limits(rate_runner, mixture):
    samples = json.loads((DATA / "h2o2.cantera-reference.json").read_text())["samples"]
    sample = next(s for s in samples if s["name"] == mixture and s["temperature"] == 1000 and s["pressure"] == 101325)
    c = sample["concentrations"]
    base = rate_runner([(1000, c)])[0]
    jacobian, thermal = base[39:139], base[139:]
    h = sum(c)*2e-6
    states = []
    for j in range(10):
        cp, cpp = c.copy(), c.copy()
        cp[j] += h
        cpp[j] += 2*h
        states += [(1000, cp), (1000, cpp)]
    # At exactly the NASA midpoint, use a one-sided low-range T derivative.
    dt = 1e-3
    states += [(1000-dt, c), (1000-2*dt, c)]
    perturbed = rate_runner(states)
    for j in range(10):
        finite = [(4*perturbed[2*j][s]-perturbed[2*j+1][s]-3*base[s])/(2*h) for s in range(10)]
        exact = [jacobian[s*10+j] for s in range(10)]
        tolerance = max(1e-6, max(map(abs, exact))*2e-7)
        assert exact == pytest.approx(finite, rel=2e-7, abs=tolerance), (mixture, j)
    finite_t = [(3*base[s]-4*perturbed[-2][s]+perturbed[-1][s])/(2*dt) for s in range(10)]
    tolerance_t = max(1e-6, max(map(abs, thermal))*2e-7)
    assert thermal == pytest.approx(finite_t, rel=2e-7, abs=tolerance_t)


def test_high_accuracy_reactor_reference_is_converged_and_conservative():
    path = DATA / "h2o2.cantera-reactor-reference.json"
    assert path.is_file(), "pinned independent full-mechanism reactor histories must exist"
    reference = json.loads(path.read_text())
    assert reference["cantera_version"] == "3.1.0"
    assert len(reference["cases"]) == 8
    for case in reference["cases"]:
        assert case["comparison"]["max_abs_temperature_difference"] < 1e-4
        assert case["comparison"]["max_abs_mass_fraction_difference"] < 1e-8
        assert len(case["states"]) == 14
        initial = case["states"][0]
        assert initial["time"] == 0
        assert case["states"][-1]["time"] == case["actual_horizon"]
        for state in case["states"]:
            assert 300 <= state["temperature"] <= 3500
            assert len(state["massFractions"]) == 10
            assert min(state["massFractions"]) >= -1e-20
            assert state["mass"] == pytest.approx(initial["mass"], rel=2e-13)
            assert state["density"] == pytest.approx(initial["density"], rel=2e-13)
            assert state["elementMoles"] == pytest.approx(initial["elementMoles"], rel=2e-9, abs=1e-20)
            jump = case["conservation_diagnostics"]["initial_composition_nasa7_midpoint_energy_jump_J"] if state["temperature"] > 1000 else 0
            assert state["internalEnergy"] == pytest.approx(initial["internalEnergy"]+jump, rel=2e-8, abs=1e-11)
