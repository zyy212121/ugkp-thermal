#!/usr/bin/env python3
"""Generate independent Cantera 3.1.0 closed, constant-V adiabatic histories.

Only a development/reference tool. Install requirements-reference.txt into an
isolated environment; no Python or Cantera dependency enters the solver.
Every case is repeated with tighter tolerances on the exact same output grid.
"""
import argparse
import hashlib
import json
from pathlib import Path

from import_h2o2 import HERE, SOURCE_NAME, SOURCE_SHA256, VERSION, load_mechanism

VOLUME = 1e-6
TIME_FRACTIONS = [0, 1e-6, 1e-5, 1e-4, 1e-3, .003, .01, .03, .1, .2, .4, .6, .8, 1.0]
CONTROLS = {"high_accuracy": {"rtol": 1e-11, "atol": 1e-20},
            "stricter": {"rtol": 1e-12, "atol": 1e-22}}


def case_definitions():
    cases = []
    for temperature, horizon in ((1000, 1e-3), (1200, 1e-4), (1500, 5e-5)):
        for atmospheres in (1, 10):
            cases.append({"name": "nitrogen_"+str(temperature)+"K_"+str(atmospheres)+"atm",
                          "initial_temperature": temperature, "initial_pressure": atmospheres*101325.0,
                          "composition": {"H2": 2, "O2": 1, "N2": 3.76},
                          "requested_horizon": horizon, "actual_horizon": horizon})
    cases += [{"name": "argon_1200K_1atm", "initial_temperature": 1200, "initial_pressure": 101325.0,
               "composition": {"H2": 2, "O2": 1, "AR": 7}, "requested_horizon": 1e-4, "actual_horizon": 1e-4},
              {"name": "seeded_relaxation_2000K_1atm", "initial_temperature": 2000, "initial_pressure": 101325.0,
               "composition": {"H2": .01, "H": 1e-5, "O": 1e-5, "O2": .01, "OH": .01, "H2O": .3,
                               "HO2": 1e-5, "H2O2": 1e-5, "AR": .1, "N2": .6},
               "requested_horizon": 1e-4, "actual_horizon": 1e-4}]
    return cases


def integrate(ct, model, definition, controls):
    gas = ct.Solution(str(HERE/SOURCE_NAME), "ohmech")
    gas.TPX = definition["initial_temperature"], definition["initial_pressure"], definition["composition"]
    reactor = ct.IdealGasReactor(gas, energy="on", volume=VOLUME)
    network = ct.ReactorNet([reactor])
    network.rtol, network.atol = controls["rtol"], controls["atol"]
    network.max_steps = 200000
    states = []
    for fraction in TIME_FRACTIONS:
        time = definition["actual_horizon"]*fraction
        if time:
            network.advance(time)
        phase = reactor.thermo
        masses = (reactor.mass*phase.Y).tolist()
        elements = [sum(masses[s]/model["thermo"][s]["molarMass"]*model["thermo"][s]["atoms"][e]
                        for s in range(10)) for e in range(4)]
        states.append({"time": time, "temperature": reactor.T, "pressure": phase.P,
                       "density": phase.density, "volume": reactor.volume, "mass": reactor.mass,
                       "massFractions": phase.Y.tolist(), "speciesMass": masses,
                       "specificInternalEnergy": phase.int_energy_mass,
                       "internalEnergy": reactor.mass*phase.int_energy_mass, "elementMoles": elements})
    return states


def generate():
    import cantera as ct
    if ct.__version__ != VERSION:
        raise ValueError("exactly Cantera " + VERSION + " is required")
    model = load_mechanism(HERE/SOURCE_NAME)
    cases = []
    for definition in case_definitions():
        high = integrate(ct, model, definition, CONTROLS["high_accuracy"])
        strict = integrate(ct, model, definition, CONTROLS["stricter"])
        max_t = max(abs(a["temperature"]-b["temperature"]) for a, b in zip(high, strict))
        max_y = max(abs(ya-yb) for a, b in zip(high, strict) for ya, yb in zip(a["massFractions"], b["massFractions"]))
        if max_t >= 1e-4 or max_y >= 1e-8:
            raise ValueError("reference did not converge sufficiently for " + definition["name"] + ": " + repr((max_t, max_y)))
        if any(not (300 <= state["temperature"] <= 3500) for state in high+strict):
            raise ValueError("case leaves common NASA7 validity interval; shorten the declared actual horizon: " + definition["name"])
        midpoint_jump = 0.0
        if definition["initial_temperature"] == 1000:
            # Pinned NASA7 branches are not perfectly continuous at 1000 K.
            # Cantera integrates dT/dt but reports energy from piecewise NASA h.
            for index, sp in enumerate(model["thermo"]):
                def energy(coefficients):
                    t = 1000.0
                    a = coefficients
                    return 8.31446261815324*t*(a[0]-1+a[1]*t/2+a[2]*t*t/3+a[3]*t**3/4+a[4]*t**4/5+a[5]/t)/sp["molarMass"]
                midpoint_jump += strict[0]["speciesMass"][index]*(energy(sp["coefficients"][7:])-energy(sp["coefficients"][:7]))
        energy0 = strict[0]["internalEnergy"]
        conservation = {
            "initial_composition_nasa7_midpoint_energy_jump_J": midpoint_jump,
            "max_abs_reported_energy_change_J": max(abs(state["internalEnergy"]-energy0) for state in strict),
            "max_abs_energy_change_after_midpoint_jump_accounting_J": max(abs(state["internalEnergy"]-energy0-(midpoint_jump if state["temperature"] > 1000 else 0)) for state in strict),
            "note": "At initial T=1000 K the pinned low/high NASA7 branches have a finite energy jump. This diagnostic records it without changing the published Cantera states. Exact-U integration should conserve its initial U rather than reproduce this jump." if midpoint_jump else "No NASA7 midpoint crossing in this case."
        }
        case = dict(definition)
        case.update({"volume": VOLUME, "output_times": [state["time"] for state in strict],
                     "reference_controls": CONTROLS, "conservation_diagnostics": conservation, "states": strict, "high_accuracy_states": high,
                     "sampled_temperature_min": min(state["temperature"] for state in strict),
                     "sampled_temperature_max": max(state["temperature"] for state in strict),
                     "comparison": {"max_abs_temperature_difference": max_t, "max_abs_mass_fraction_difference": max_y}})
        cases.append(case)
        print(definition["name"], "max T", case["sampled_temperature_max"], "tightening dT", max_t, "dY", max_y)
    reference = {"cantera_version": VERSION, "source_sha256": SOURCE_SHA256, "species": model["species"],
                 "elements": model["elements"], "species_order_hash": str(model["speciesOrderHash"]),
                 "thermo_hash": str(model["thermoHash"]), "mechanism_hash": str(model["mechanismHash"]),
                 "reactor": "IdealGasReactor: closed, constant volume, energy on, no walls or flow devices",
                 "units": {"time": "s", "temperature": "K", "pressure": "Pa", "density": "kg/m^3", "volume": "m^3",
                           "mass": "kg", "speciesMass": "kg", "specificInternalEnergy": "J/kg", "internalEnergy": "J", "elementMoles": "mol atoms"},
                 "common_thermo_interval": [300, 3500], "reference_controls": CONTROLS,
                 "published_states": "stricter controls; high_accuracy_states retained for independent convergence check",
                 "cases": cases}
    return json.dumps(reference, sort_keys=True, indent=2)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"h2o2.cantera-reactor-reference.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = generate()
    if args.check:
        if not args.output.exists() or args.output.read_bytes() != content.encode():
            raise SystemExit("stale reactor reference: " + str(args.output))
    else:
        args.output.write_text(content)
    print("verified" if args.check else "generated", args.output.name, "sha256", hashlib.sha256(content.encode()).hexdigest())


if __name__ == "__main__":
    main()
