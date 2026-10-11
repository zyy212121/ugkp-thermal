#!/usr/bin/env python3
"""Regenerate the complete pinned Cantera 3.1.0 ideal-gas ohmech mechanism.

Development-only dependency: PyYAML 6.x. Optional --reference-json requires
Cantera 3.1.0. Neither Python nor Cantera is a solver runtime dependency.
All Arrhenius pre-exponentials are converted from cm/mol/s to m/mol/s by
A_SI = A_source * (1e-6)**(total_forward_order - 1). Activation energies
are converted from cal/mol to J/mol using 4.184 exactly. Explicit colliders
are retained on both equation sides, not cancelled from kinetic orders.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import yaml

HERE = Path(__file__).resolve().parent
VERSION = "3.1.0"
SOURCE_NAME = "h2o2-cantera-3.1.0.yaml"
SOURCE_URL = "https://raw.githubusercontent.com/Cantera/cantera/v3.1.0/data/h2o2.yaml"
SOURCE_SHA256 = "0efc6c52862741a29e0c29b65d979c7d8cb409db5282bca83b9c5437b3d8c8d4"
REFERENCE_JSON_SHA256 = "508f50970314753b8cd9f1510594e30ec99bfe25bb0253e7bbf73d2bc796bd7e"
LICENSE_SHA256 = "469fa95f76e0c652b4f204c2e5d3ffd184dbbb85b2a7fbc75355bee3a16e48e2"
SPECIES = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR", "N2"]
# Cantera 3.1.0 Elements.cpp standard atomic weights, g/mol.
ATOMIC_WEIGHTS = {"O": 15.999, "H": 1.008, "Ar": 39.95, "N": 14.007}
TYPE_IDS = {"elementary": 0, "thirdBody": 1, "Lindemann": 2, "Troe": 3}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fnv(data, value=14695981039346656037):
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64)-1)
    return value


def hash_scalar(value, number):
    return fnv(struct.pack("<d", float(number) if number else 0.0), value)


def hash_word(value, word):
    return fnv(word.encode("utf-8")+b"\0", value)


def identities(m):
    h = 14695981039346656037
    for name in m["species"]:
        h = hash_word(h, name)
    m["speciesOrderHash"] = h
    for sp in m["thermo"]:
        h = fnv(bytes([1]), h)  # SpeciesThermoModel::NASA7
        for key in ("molarMass", "minTemperature", "midTemperature", "maxTemperature", "referencePressure"):
            h = hash_scalar(h, sp[key])
        h = hash_scalar(h, 0)  # unused LinearCp entropy-reference fields
        h = hash_scalar(h, 0)
        h = fnv(bytes([0]), h)
    for sp in m["thermo"]:
        for value in sp["coefficients"]:
            h = hash_scalar(h, value)
    for element in m["elements"]:
        h = hash_word(h, element)
    for sp in m["thermo"]:
        for count in sp["atoms"]:
            h = hash_scalar(h, count)
    m["thermoHash"] = h
    # Semantic FNV identity shared with GasMechanismIO.H (binary64 LE scalars).
    h = hash_scalar(m["speciesOrderHash"], m["referencePressure"])
    h = hash_scalar(h, m["independentRank"])
    for v in m["stoichiometricBasis"]:
        h = hash_scalar(h, v)
    for r in m["reactions"]:
        h = fnv(bytes([TYPE_IDS[r["type"]], int(r["reversible"]), int(r["duplicate"])]), h)
        h = hash_scalar(h, r["sourceIndex"])
        for key in ("highRate", "lowRate"):
            for v in r[key]:
                h = hash_scalar(h, v)
        h = hash_scalar(h, r["defaultEfficiency"])
        for v in r["troe"][:4]:
            h = hash_scalar(h, v)
        h = fnv(bytes([int(r["troe"][4])]), h)
        for key in ("reactants", "products", "efficiencies"):
            h = hash_scalar(h, len(r[key]))
            for pair in r[key]:
                for v in pair:
                    h = hash_scalar(h, v)
    m["mechanismHash"] = h


def parse_side(text, species):
    text = text.replace("(+M)", "")
    terms = {}
    for piece in text.strip().split(" + "):
        piece = piece.strip()
        if piece == "M":
            continue
        match = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s+)?([A-Za-z][A-Za-z0-9]*)", piece)
        if not match or match[2] not in species:
            raise ValueError("unsupported stoichiometric term: " + piece)
        index = species.index(match[2])
        terms[index] = terms.get(index, 0.0) + float(match[1] or 1)
    return [[index, terms[index]] for index in sorted(terms)]


def independent_columns(columns):
    """Exact rational elimination; preserve original independent molar columns."""
    reduced, selected = [], []
    for i, column in enumerate(columns):
        candidate = [Fraction(v) for v in column]
        for pivot, previous in reduced:
            factor = candidate[pivot]
            candidate = [a-factor*b for a, b in zip(candidate, previous)]
        pivot = next((j for j, value in enumerate(candidate) if value), None)
        if pivot is not None:
            scale = candidate[pivot]
            candidate = [v/scale for v in candidate]
            reduced.append((pivot, candidate))
            selected.append(i)
    return selected


def load_mechanism(path):
    raw = Path(path).read_bytes()
    if sha256(raw) != SOURCE_SHA256:
        raise ValueError("reference SHA-256 mismatch; only the pinned Cantera 3.1.0 file is accepted")
    source = yaml.safe_load(raw)
    if source["units"] != {"length": "cm", "time": "s", "quantity": "mol", "activation-energy": "cal/mol"}:
        raise ValueError("unsupported source units")
    phase = next(p for p in source["phases"] if p["name"] == "ohmech")
    assert phase["thermo"] == "ideal-gas" and phase["species"] == SPECIES
    m = {"phase": "ohmech", "species": phase["species"], "elements": phase["elements"],
         "referencePressure": 101325.0, "thermo": [], "reactions": []}
    source_species = {sp["name"]: sp for sp in source["species"]}
    for name in m["species"]:
        sp = source_species[name]
        th = sp["thermo"]
        if th["model"] != "NASA7" or len(th["data"]) != 2 or any(len(a) != 7 for a in th["data"]):
            raise ValueError("only complete two-interval NASA7 thermo is supported")
        low, mid, high = map(float, th["temperature-ranges"])
        atoms = [float(sp["composition"].get(e, 0)) for e in m["elements"]]
        m["thermo"].append({"name": name, "molarMass": sum(sp["composition"].get(e, 0)*ATOMIC_WEIGHTS[e] for e in m["elements"])/1000.0,
                            "minTemperature": low, "midTemperature": mid, "maxTemperature": high,
                            "referencePressure": 101325.0, "coefficients": [float(v) for row in th["data"] for v in row], "atoms": atoms})
    def arrhenius(rate, order):
        return [float(rate["A"])*(1e-6)**(order-1), float(rate["b"]), float(rate["Ea"])*4.184]
    for i, item in enumerate(source["reactions"]):
        unknown = set(item)-{"equation", "type", "rate-constant", "low-P-rate-constant", "high-P-rate-constant", "Troe", "efficiencies", "duplicate", "default-efficiency"}
        if unknown:
            raise ValueError("unsupported rate fields: " + str(sorted(unknown)))
        equation = item["equation"]
        arrow = "<=>" if "<=>" in equation else "=>"
        left, right = equation.split(arrow)
        reactants, products = parse_side(left, SPECIES), parse_side(right, SPECIES)
        order = sum(c for _, c in reactants)
        tag = item.get("type", "elementary")
        if tag not in ("elementary", "three-body", "falloff"):
            raise ValueError("unsupported reaction type: " + tag)
        kind = {"elementary": "elementary", "three-body": "thirdBody", "falloff": "Troe" if "Troe" in item else "Lindemann"}[tag]
        high = arrhenius(item["high-P-rate-constant"], order) if tag == "falloff" else arrhenius(item["rate-constant"], order + (tag == "three-body"))
        low = arrhenius(item["low-P-rate-constant"], order+1) if tag == "falloff" else [0.0]*3
        tr = item.get("Troe", {})
        troe = [float(tr.get(key, 0)) for key in ("A", "T3", "T1", "T2")] + ["T2" in tr]
        eff = [[SPECIES.index(s), float(v)] for s, v in item.get("efficiencies", {}).items()]
        eff.sort()
        m["reactions"].append({"equation": equation, "type": kind, "reversible": arrow == "<=>", "duplicate": item.get("duplicate", False),
                               "sourceIndex": i, "reactants": reactants, "products": products, "highRate": high, "lowRate": low,
                               "defaultEfficiency": float(item.get("default-efficiency", 1)), "efficiencies": eff, "troe": troe})
    if len(m["reactions"]) != 29:
        raise ValueError("pinned mechanism must contain all 29 reaction channels")
    columns = []
    for r in m["reactions"]:
        column = [0.0]*len(SPECIES)
        for index, coeff in r["products"]:
            column[index] += coeff
        for index, coeff in r["reactants"]:
            column[index] -= coeff
        for e in range(len(m["elements"])):
            if sum(column[s]*m["thermo"][s]["atoms"][e] for s in range(len(SPECIES))) != 0:
                raise ValueError("unbalanced reaction " + str(r["sourceIndex"]))
        columns.append(column)
    indices = independent_columns(columns)
    m["independentRank"] = len(indices)
    m["basisReactionIndices"] = indices
    m["stoichiometricBasis"] = [columns[j][s] for s in range(len(SPECIES)) for j in indices]
    if len(indices) != 6:
        raise ValueError("expected exact rank-six molar stoichiometric space")
    identities(m)
    return m


def num(value):
    return format(float(value), ".17g") if value else "0"


def cpp(value):
    return "Real("+num(value)+")"


def boolean(value):
    return "true" if value else "false"


def header(m):
    arrays = {"reactants": [], "products": [], "efficiencies": []}
    entries = []
    for r in m["reactions"]:
        offsets = []
        for name in arrays:
            offsets += [str(len(arrays[name])), str(len(r[name]))]
            arrays[name].extend(r[name])
        values = ["GasReactionType::"+{"elementary": "Elementary", "thirdBody": "ThirdBody", "Lindemann": "Lindemann", "Troe": "Troe"}[r["type"]], boolean(r["reversible"]), boolean(r["duplicate"]), str(r["sourceIndex"])] + offsets
        values += ["{"+", ".join(map(cpp, r[k]))+"}" for k in ("highRate", "lowRate")]
        values += [cpp(r["defaultEfficiency"]), "{"+", ".join(map(cpp, r["troe"][:4]))+", "+boolean(r["troe"][4])+"}"]
        entries.append("        {"+", ".join(values)+"}, // "+str(r["sourceIndex"]+1)+": "+r["equation"])
    lines = ["// Generated by import_h2o2.py; do not edit generated tables.", "// Full Cantera v3.1.0 h2o2.yaml, ideal-gas phase ohmech.", "// Reference SHA-256: "+SOURCE_SHA256, "// Upstream copyright/license: CANTERA-LICENSE.txt.", "#ifndef UGKWP_GENERATED_H2O2_H", "#define UGKWP_GENERATED_H2O2_H", '#include "../../gasTransport/GasModel.H"', "", "namespace ugkwp", "{", "template<class Real>", "struct GeneratedH2O2", "{", "    static constexpr int speciesCount = 10;", "    static constexpr int reactionCount = 29;", "    static constexpr int independentRank = 6;", "    static constexpr int elementCount = 4;", "    const char* const speciesNames[10] = {"+", ".join(json.dumps(s) for s in SPECIES)+"};", "    const SpeciesThermoData<Real> species[10] = {" ]
    for i, s in enumerate(m["thermo"]):
        values = ["SpeciesThermoModel::NASA7", str(i*14)] + [cpp(s[k]) for k in ("molarMass", "minTemperature", "midTemperature", "maxTemperature", "referencePressure")] + ["Real(0)", "Real(0)", "false"]
        lines += ["        {"+", ".join(values)+"}, // "+s["name"]]
    lines += ["    };", "    const Real coefficients[140] = {"]
    for s in m["thermo"]:
        lines += ["        "+", ".join(map(cpp, s["coefficients"]))+", // "+s["name"]]
    lines += ["    };", "    const Real elementComposition[40] = {"]
    for s in m["thermo"]:
        lines += ["        "+", ".join(map(cpp, s["atoms"]))+","]
    lines += ["    };", "    const GasReactionData<Real> reactions[29] = {"] + entries + ["    };"]
    for name, values in arrays.items():
        kind = "GasColliderEfficiency" if name == "efficiencies" else "GasStoichTerm"
        lines += ["    const "+kind+"<Real> "+name+"["+str(len(values))+"] = {"]
        lines += ["        {"+str(s)+", "+cpp(v)+"}," for s, v in values]
        lines += ["    };"]
    lines += ["    // Dimensionless molar stoichiometric columns, species-major [10*6].", "    const Real stoichiometricBasis[60] = {"]
    for i in range(10):
        lines += ["        "+", ".join(map(cpp, m["stoichiometricBasis"][6*i:6*i+6]))+","]
    lines += ["    };", "", "    UGKWP_GAS_HD SpeciesThermoView<Real, 10> thermoView() const noexcept", "    {", "        SpeciesThermoView<Real, 10> view;", "        view.species = species; view.coefficients = coefficients;", "        view.coefficientCount = 140; view.elementCount = 4;", "        view.elementComposition = elementComposition;", "        view.speciesOrderHash = UINT64_C("+str(m["speciesOrderHash"])+");", "        view.thermoHash = UINT64_C("+str(m["thermoHash"])+");", "        return view;", "    }", "    UGKWP_GAS_HD GasMechanismView<Real, 10> mechanismView() const noexcept", "    {", "        GasMechanismView<Real, 10> view;", "        view.reactions = reactions; view.reactionCount = 29;"]
    for name, values in arrays.items():
        count = {"reactants": "reactantTermCount", "products": "productTermCount", "efficiencies": "efficiencyCount"}[name]
        lines += ["        view."+name+" = "+name+"; view."+count+" = "+str(len(values))+";"]
    lines += ["        view.stoichiometricBasis = stoichiometricBasis; view.independentRank = 6;", "        view.referencePressure = Real(101325);", "        view.speciesOrderHash = UINT64_C("+str(m["speciesOrderHash"])+");", "        view.mechanismHash = UINT64_C("+str(m["mechanismHash"])+");", "        return view;", "    }", "};", "} // namespace ugkwp", "#endif", ""]
    return "\n".join(lines)


def foam_list(values):
    return "("+" ".join(num(v) for v in values)+")"


def foam_mechanism(m):
    lines = ["// Generated from full Cantera 3.1.0 h2o2.yaml; SI m, mol, s, J/mol.", "schemaVersion 1;", "units siMolar;", 'sourceSha256 "'+SOURCE_SHA256+'";', "phase ohmech;", "species ("+" ".join(SPECIES)+");", "referencePressure 101325;"]
    for k in ("speciesOrderHash", "mechanismHash", "independentRank"):
        lines += [k+" "+str(m[k])+";"]
    lines += ["stoichiometricBasis "+foam_list(m["stoichiometricBasis"])+";", "reactions", "{"]
    for r in m["reactions"]:
        lines += ["    reaction"+str(r["sourceIndex"]), "    {", "        // "+r["equation"]]
        for k in ("type", "reversible", "duplicate", "sourceIndex"):
            v = boolean(r[k]) if isinstance(r[k], bool) else str(r[k])
            lines += ["        "+k+" "+v+";"]
        for k in ("reactants", "products", "efficiencies"):
            lines += ["        "+k+" "+foam_list([v for pair in r[k] for v in pair])+";"]
        for k in ("highRate", "lowRate"):
            lines += ["        "+k+" "+foam_list(r[k])+";"]
        lines += ["        defaultEfficiency "+num(r["defaultEfficiency"])+";", "        troe "+foam_list(r["troe"][:4])+";", "        troeHasT2 "+boolean(r["troe"][4])+";", "    }"]
    return "\n".join(lines+["}", ""])


def foam_thermo(m):
    lines = ["// Generated pinned mechanism example. Place alongside h2o2.mechanism.", "schemaVersion 1;", "gasMode mixtureChemistry;", "species ("+" ".join(SPECIES)+");", "elements ("+" ".join(m["elements"])+");", 'mechanism "h2o2.mechanism";', "phase ohmech;", "diffusion { model none; turbulentSchmidt 0.7; }", "speciesThermo", "{"]
    for s in m["thermo"]:
        lines += ["    "+s["name"], "    {", "        model NASA7;"]
        for k in ("molarMass", "minTemperature", "midTemperature", "maxTemperature", "referencePressure"):
            lines += ["        "+k+" "+num(s[k])+";"]
        for k in ("coefficients", "atoms"):
            lines += ["        "+k+" "+foam_list(s[k])+";"]
        lines += ["    }"]
    return "\n".join(lines+["}", ""])


def generate(m):
    files = {"GeneratedH2O2.H": header(m), "h2o2.mechanism": foam_mechanism(m), "h2o2.gasModelProperties": foam_thermo(m)}
    manifest = {"reference": {"project": "Cantera", "version": VERSION, "phase": "ohmech", "source": SOURCE_NAME, "url": SOURCE_URL, "sha256": SOURCE_SHA256},
                "license": {"file": "CANTERA-LICENSE.txt", "url": "https://raw.githubusercontent.com/Cantera/cantera/v3.1.0/License.txt", "sha256": LICENSE_SHA256},
                "independent_reference": {"file": "h2o2.cantera-reference.json", "sha256": REFERENCE_JSON_SHA256, "sample_count": 63,
                                          "gibbs_reference_pressure_pa": 101325, "dependencies": ["Cantera==3.1.0", "numpy==2.5.3", "ruamel.yaml==0.19.1"],
                                          "regenerate": "python common/chemistry/mechanisms/import_h2o2.py --reference-json common/chemistry/mechanisms/h2o2.cantera-reference.json"},
                "species": SPECIES, "species_count": 10, "reaction_count": 29, "elements": m["elements"], "independent_rank": m["independentRank"], "basis_reaction_indices_zero_based": m["basisReactionIndices"],
                "species_order_hash": str(m["speciesOrderHash"]), "thermo_hash": str(m["thermoHash"]), "mechanism_hash": str(m["mechanismHash"]),
                "units": {"concentration": "mol/m^3", "molar_mass": "kg/mol", "activation_energy": "J/mol", "pressure": "Pa", "time": "s", "basis": "dimensionless molar stoichiometry"},
                "conversion": {"pre_exponential": "A_source * (1e-6)^(total_forward_order-1); third-body order includes M; low-pressure order includes M", "activation_energy": "Ea_cal_per_mol * 4.184", "nasa7_order": "low seven followed by high seven", "explicit_colliders": "retained on both reaction sides"},
                "regenerate": "python common/chemistry/mechanisms/import_h2o2.py", "check": "python common/chemistry/mechanisms/import_h2o2.py --check", "development_dependencies": ["PyYAML==6.0.3"],
                "generated_sha256": {name: sha256(text.encode()) for name, text in files.items()}, "importer_sha256": sha256(Path(__file__).read_bytes())}
    files["h2o2.manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True)+"\n"
    return files


def reference_json(m):
    import cantera as ct
    if ct.__version__ != VERSION:
        raise ValueError("reference generation requires exactly Cantera " + VERSION)
    gas = ct.Solution(str(HERE/SOURCE_NAME), "ohmech")
    samples = []
    mixtures = {"all_species": {s: i+1 for i, s in enumerate(SPECIES)}, "zero_radicals": {"H2": 2, "O2": 1, "N2": 3.76}, "peroxide": {"H2O2": .2, "H2O": .3, "O2": .1, "AR": .4}}
    for name, composition in mixtures.items():
        for temperature in (300.0, 800.0, 999.999999, 1000.0, 1000.000001, 1500.0, 3000.0):
            for pressure in (1013.25, 101325.0, 10132500.0):
                gas.TPX = temperature, pressure, composition
                samples.append({"name": name, "temperature": temperature, "pressure": pressure, "massFractions": gas.Y.tolist(), "density": gas.density,
                                "concentrations": (gas.concentrations*1000).tolist(), "forwardRatesOfProgress": (gas.forward_rates_of_progress*1000).tolist(),
                                "reverseRatesOfProgress": (gas.reverse_rates_of_progress*1000).tolist(), "netProductionRates": (gas.net_production_rates*1000).tolist(),
                                "cpMolar": (gas.standard_cp_R*ct.gas_constant/1000).tolist(), "enthalpyMolar": (gas.standard_enthalpies_RT*ct.gas_constant*temperature/1000).tolist(),
                                "gibbsMolar": ((gas.standard_gibbs_RT-math.log(pressure/gas.reference_pressure))*ct.gas_constant*temperature/1000).tolist()})
    return json.dumps({"cantera_version": VERSION, "source_sha256": SOURCE_SHA256, "species": SPECIES, "units": "SI m, mol, s, J, Pa", "samples": samples}, sort_keys=True, indent=2)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if checked-in generated artifacts differ")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--reference-json", type=Path, help="also generate optional independent Cantera reference samples")
    args = parser.parse_args()
    if sha256((HERE/"CANTERA-LICENSE.txt").read_bytes()) != LICENSE_SHA256:
        raise ValueError("upstream license SHA-256 mismatch")
    m = load_mechanism(HERE/SOURCE_NAME)
    files = generate(m)
    stale = []
    for name, content in files.items():
        target = args.output_dir/name
        if args.check:
            if not target.exists() or target.read_bytes() != content.encode():
                stale.append(name)
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    if stale:
        raise SystemExit("stale generated artifacts: " + ", ".join(stale))
    if args.reference_json:
        content = reference_json(m)
        if sha256(content.encode()) != REFERENCE_JSON_SHA256:
            raise ValueError("Cantera reference sample SHA-256 changed; check pinned development dependencies")
        if args.check:
            if not args.reference_json.exists() or args.reference_json.read_bytes() != content.encode():
                raise SystemExit("stale independent reference samples: " + str(args.reference_json))
        else:
            args.reference_json.write_text(content)
    if args.check and sha256((HERE/"h2o2.cantera-reference.json").read_bytes()) != REFERENCE_JSON_SHA256:
        raise SystemExit("independent reference samples SHA-256 mismatch")
    print("verified" if args.check else "generated", "10 species, 29 reactions, rank 6; source SHA-256", SOURCE_SHA256)


if __name__ == "__main__":
    main()
