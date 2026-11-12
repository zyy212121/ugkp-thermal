#!/usr/bin/env python3
"""Offline case-input migration. This module never imports or launches a solver.

Only explicitly listed numerical controls move. Physical/model/precision inputs
retain their original interface. Original dictionary bytes and each moved value
are retained in solver-control-migration.json, outside constant/scheme. This
provenance permits a legacy baseline copy and precision-aware old effective
tolerances without adding any production runtime fallback.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

HEADER = "FoamFile { version 2.0; format ascii; class dictionary; object scheme; }\n"
MANIFEST = "solver-control-migration.json"
EPS32 = 2.0 ** -23
TOKEN = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|[{};()[\]]|[^\s{};()[\]"]+')

class Entry:
    def __init__(self, key, start, end, value="", children=None, parent=None):
        self.key, self.start, self.end = key, start, end
        self.value, self.children, self.parent = value, children, parent

class Document:
    def __init__(self, text):
        self.text = text
        self.tokens = [m for m in TOKEN.finditer(text)
                       if not m.group().startswith(("//", "/*"))]
        self.entries, _ = self._parse(0, None)

    def _parse(self, index, parent):
        result = []
        tokens = self.tokens
        while index < len(tokens):
            token = tokens[index]
            if token.group() == "}":
                return result, index + 1
            if token.group() == ";":
                index += 1
                continue
            key = token.group().strip('"')
            end = token.end()
            index += 1
            # OpenFOAM operator names such as div(phi,(p|rho)) are one key.
            # Preserve adjacent punctuation in an unquoted dictionary key.
            if not token.group().startswith('"'):
                while index < len(tokens) and tokens[index].start() == end and tokens[index].group() not in ("{", "}", ";"):
                    key += tokens[index].group()
                    end = tokens[index].end()
                    index += 1
            node = Entry(key, token.start(), end, parent=parent)
            if index < len(tokens) and tokens[index].group() == "{":
                node.children, index = self._parse(index + 1, node)
                node.end = tokens[index - 1].end()
                if index < len(tokens) and tokens[index].group() == ";":
                    node.end = tokens[index].end()
                    index += 1
            else:
                begin = index
                depth = 0
                while index < len(tokens):
                    value = tokens[index].group()
                    if value in ("(", "[", "{"):
                        depth += 1
                    elif value in (")", "]", "}"):
                        if depth == 0:
                            break
                        depth -= 1
                    if value == ";" and depth == 0:
                        break
                    index += 1
                if index > begin:
                    node.value = self.text[tokens[begin].start():tokens[index - 1].end()]
                    node.end = tokens[index - 1].end()
                if index < len(tokens) and tokens[index].group() == ";":
                    node.end = tokens[index].end()
                    index += 1
                elif index == begin and index < len(tokens):
                    # Preserve unsupported directives as text, without looping.
                    index += 1
            result.append(node)
        return result, index

    def find(self, path):
        nodes = self.entries
        result = None
        for part in path:
            result = next((n for n in reversed(nodes) if n.key == part), None)
            if result is None:
                return None
            nodes = result.children or []
        return result

    def scalar(self, node):
        value = node.value.strip()
        seen = set()
        while re.fullmatch(r"\$[A-Za-z_]\w*", value):
            key = value[1:]
            if key in seen:
                return value
            seen.add(key)
            parent = node.parent
            found = None
            while True:
                nodes = self.entries if parent is None else parent.children
                found = next((n for n in reversed(nodes) if n.key == key and n.children is None), None)
                if found or parent is None:
                    break
                parent = parent.parent
            if found is None:
                return value
            value = found.value.strip()
        return value

def tree(text):
    def convert(nodes):
        return {(chr(34)+n.key+chr(34) if text[n.start:n.start+1] == chr(34) else n.key): convert(n.children) if n.children is not None else n.value
                for n in nodes if n.key != "FoamFile" and not n.key.startswith("#")}
    return convert(Document(text).entries)

def merge(left, right):
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(left.get(key), dict):
            merge(left[key], value)
        else:
            left[key] = value
    return left

def render(values, indent=0):
    lines = []
    for key, value in values.items():
        pad = " " * indent
        if isinstance(value, dict):
            lines.extend((pad + key, pad + "{", render(value, indent + 4), pad + "}"))
        else:
            lines.append(pad + key + " " + str(value) + ";")
    return "\n".join(lines)

def get(values, path, default=None):
    for key in path:
        if not isinstance(values, dict) or key not in values:
            return default
        values = values[key]
    return values

def put(values, path, value):
    for key in path[:-1]:
        values = values.setdefault(key, {})
    values[path[-1]] = str(value)

def numeric(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default

def effective(value, rule, precision, angles=0):
    number = numeric(value)
    if precision != 32 or number is None:
        return value
    floor = {"gasEnergy": 32*EPS32, "gasTemperature": 8*EPS32,
             "wallRelative": 1e-4, "wallAbsolute": 8*EPS32,
             "solidNonlinear": 1e-5, "solidEnergy": 1e-4,
             "radiation": 32*EPS32,
             "phase": max(32, 8*angles)*EPS32}.get(rule)
    return format(max(number, floor), ".17g") if floor is not None else value

def precision_of(case, default=64):
    p = case/"constant/precisionProperties"
    if p.is_file():
        n = Document(p.read_text()).find(("precision",))
        if n and n.value.strip() in ("32", "64"):
            return int(n.value.strip())
    return default

def migrate_case(case, precision=None, write=True):
    case = Path(case)
    precision = precision or precision_of(case)
    scheme_file = case/"constant/scheme"
    values = tree(scheme_file.read_text()) if scheme_file.is_file() else {}
    mp = case/MANIFEST
    old = json.loads(mp.read_text()) if mp.is_file() else {}
    report = dict(schema=1, precision=precision, original_files=old.get("original_files", {}),
                  moves=old.get("moves", []), discarded=old.get("discarded", []),
                  notes=old.get("notes", []), input_sha256=old.get("input_sha256", {}))
    # Re-select only values with known legacy provenance, never arbitrary new
    # scheme controls. Source overlays retain the raw value used before migration.
    for row in report["moves"]:
        if "rule" in row:
            # The saved migration manifest is provenance, not a live control.
            # Preserve later explicit scheme edits; only reconstruct missing keys.
            if get(values, row["target"]) is None:
                put(values, row["target"], row["raw"])
            precision_target = ("precisionOverrides", "fp32", *row["target"])
            if get(values, precision_target) is None:
                put(values, precision_target, effective(row["raw"], row["rule"], 32, row.get("angles", 0)))
    cache = {}
    removed = {}
    def document(name):
        if name not in cache:
            p = case/name
            cache[name] = Document(p.read_text()) if p.is_file() else None
        return cache[name]
    def record(name, node, target, raw=None, rule=None, note=None, angles=0):
        doc = document(name)
        raw = doc.scalar(node) if raw is None else str(raw)
        # References that cannot be resolved retain their exact original bytes.
        # Such a value is not transplanted into a different lexical scope.
        if "$" in raw or raw.startswith("#"):
            report["notes"].append("Unresolved reference retained: "+name+":"+node.key)
            return
        result = effective(raw, rule, precision, angles)
        put(values, target, raw)
        if rule:
            put(values, ("precisionOverrides","fp32",*target), effective(raw, rule, 32, angles))
        removed.setdefault(name, set()).add((node.start, node.end))
        source_path=[node.key]
        ancestor=node.parent
        while ancestor is not None:
            source_path.insert(0,ancestor.key); ancestor=ancestor.parent
        row = dict(source=name, key=node.key, source_path=source_path, target=list(target), raw=raw, value=result,
                   old_requested=raw, old_effective_fp64=raw,
                   old_effective_fp32=effective(raw,rule,32,angles), new_scheme_fp64=raw,
                   new_scheme_fp32=effective(raw,rule,32,angles))
        if rule:
            row.update(rule=rule, angles=angles)
        if note:
            row["note"] = note
        report["moves"] = [r for r in report["moves"] if r["target"] != list(target)]
        report["moves"].append(row)
    def move(name, parent, keys, target, aliases=None, rules=None):
        doc = document(name)
        if not doc:
            return
        for key in keys:
            node = doc.find(tuple(parent)+(key,))
            if node is not None and node.children is None:
                record(name, node, tuple(target)+((aliases or {}).get(key, key),),
                       rule=(rules or {}).get(key))
    def discard(name, parent, keys):
        doc = document(name)
        if not doc:
            return
        for key in keys:
            node = doc.find(tuple(parent)+(key,))
            if node:
                removed.setdefault(name, set()).add((node.start, node.end))
                report["discarded"].append(dict(source=name, key=key, source_path=list(parent)+[key], raw=node.value, reason="No production consumer; not activated"))

    # fvSchemes is still an OpenFOAM framework dictionary. Retain its structure,
    # but the UGKP numerical reader consumes only this copied scheme section.
    fvname = "system/fvSchemes"
    fvdoc = document(fvname)
    if fvdoc:
        for key in ("fluxScheme", "gasLimiter", "ddtSchemes", "gradSchemes", "divSchemes",
                    "laplacianSchemes", "interpolationSchemes", "snGradSchemes"):
            node = fvdoc.find((key,))
            if node is not None and key not in values.get("gasNumerics", {}):
                values.setdefault("gasNumerics", {})[key] = tree(fvdoc.text[node.start:node.end]).get(key)
                report["original_files"].setdefault(fvname,fvdoc.text)
                report["input_sha256"].setdefault(fvname,hashlib.sha256(fvdoc.text.encode()).hexdigest())
                report["notes"].append("Copied system/fvSchemes:"+key+" to gasNumerics; framework dictionary retained, not a UGKP fallback")

    if fvdoc:
        for key in ("fluxScheme","gasLimiter"):
            node=fvdoc.find((key,))
            if node is not None:
                removed.setdefault(fvname,set()).add((node.start,node.end))

    for path in sorted((case/"system").glob("*/fvSchemes")):
        name=str(path.relative_to(case))
        doc=document(name)
        for key in ("laplacianSchemes","gradSchemes","interpolationSchemes","ddtSchemes"):
            node=doc.find((key,))
            if node is None:continue
            target=("regions",path.parent.name,"solidDiscretization",key)
            if get(values,target) is None:
                region=values.setdefault("regions",{}).setdefault(path.parent.name,{})
                region.setdefault("solidDiscretization",{})[key]=tree(doc.text[node.start:node.end])[key]
                report["original_files"].setdefault(name,doc.text)
                report["input_sha256"].setdefault(name,hashlib.sha256(doc.text.encode()).hexdigest())
                report["notes"].append("Copied "+name+":"+key+" to regions."+path.parent.name+".solidDiscretization; framework structure retained")

    gm = "constant/gasModelProperties"
    move(gm, ("chemistryControls",), ("relativeTolerance", "absoluteMassFractionTolerance", "newtonTolerance",
         "maximumNewtonIterations", "maximumLineSearchIterations"), ("gasChemistry",))
    move(gm, ("chemistryControls", "thermo"), ("absoluteEnergyTolerance", "relativeEnergyTolerance",
         "absoluteTemperatureTolerance", "relativeTemperatureTolerance", "maximumIterations"), ("gasThermo",),
         rules={"relativeEnergyTolerance":"gasEnergy", "relativeTemperatureTolerance":"gasTemperature"})
    discard(gm, ("chemistryControls",), ("initialStep", "minimumStep", "maximumStep", "maximumSteps",
            "maximumRejectedSteps", "absoluteMassTolerance", "relativeConservationTolerance",
            "absoluteElementTolerance", "absoluteTemperatureTolerance"))
    fluid = "constant/fluidProperties"
    for path in (("turbulence", "boundaryLayer"), ("turbulence", "RAS", "kOmegaSSTCoeffs", "boundaryLayer")):
        move(fluid, path, ("maxIterations", "relativeTolerance", "absoluteTolerance", "rtol", "atol", "nodes", "stretch", "quadratureRefinement"),
             ("wallLayer",), aliases={"rtol":"relativeTolerance", "atol":"absoluteTolerance"},
             rules={"relativeTolerance":"wallRelative", "rtol":"wallRelative",
                    "absoluteTolerance":"wallAbsolute", "atol":"wallAbsolute"})
    ctrl = "system/controlDict"
    move(ctrl, ("steadyGas",), ("maxCo", "maxDeltaT", "boundaryTime", "relativeTolerance", "absoluteTolerance",
         "residualInterval", "minimumIterations", "referenceTime"), ("steadyGas",))
    move(ctrl, ("steadyGas",), ("samplingDeltaT", "boundaryTime", "windowSamples", "minimumWindows",
         "stableWindows", "meanTolerance", "inventoryTolerance"), ("steadyParticle",))
    move(ctrl, (), ("maxCo", "maxDeltaT", "adjustTimeStep"), ("gasTimeStep",))
    move("constant/schedulingProperties", (), ("gpuResidentCourantUpdateInterval", "gpuResidentMaxDeltaTGrowth"),
         ("gasTimeStep",), aliases={"gpuResidentCourantUpdateInterval":"courantUpdateInterval",
                                   "gpuResidentMaxDeltaTGrowth":"maximumTimeStepGrowth"})
    move("constant/schedulingProperties", (), ("gpuResidentMaxFaceWalkHops",),
         ("particle","tracking"), aliases={"gpuResidentMaxFaceWalkHops":"maxFaceWalkHops"})
    for path in (("turbulence", "boundaryLayer"), ("turbulence", "RAS", "kOmegaSSTCoeffs", "boundaryLayer")):
        discard(fluid,path,("geometryTolerance","maxRayCells"))
    solid = "constant/solidRegionProperties"
    move(solid,("solidThermalCoupling",),("couplingInterval","solidContactCouplingInterval","radiationCouplingFrequency"),("thermalCoupling",))
    move(solid, ("solidThermalCoupling",), ("nonlinearMaxIterations", "nonlinearRelativeTolerance",
         "energyRelativeTolerance", "energyAbsoluteTolerance", "nonlinearAitken", "nonlinearInitialRelaxation",
         "nonlinearMinimumRelaxation", "nonlinearMaximumRelaxation"), ("solid", "nonlinear"),
         rules={"nonlinearRelativeTolerance":"solidNonlinear", "energyRelativeTolerance":"solidEnergy"})
    move(solid, ("solidThermalCoupling",), ("solidInterfaceMaxIterations",
         "solidInterfaceTemperatureRelativeTolerance", "solidInterfaceEnergyRelativeTolerance"), ("solid", "interface"))
    for p in sorted((case/"system").glob("*/fvSolution")):
        name = str(p.relative_to(case))
        doc = document(name)
        solvers = doc.find(("solvers",))
        if not solvers:
            continue
        def selected(field):
            exact = next((n for n in solvers.children if n.key == field and n.children is not None), None)
            if exact:
                return exact
            for n in reversed(solvers.children):
                try:
                    if n.children is not None and re.fullmatch(n.key, field):
                        return n
                except re.error:
                    pass
            return None
        def inherited(node, seen=()):
            if not node or node.key in seen:
                return {}
            result = {}
            for n in node.children:
                if n.key.startswith("$"):
                    ref = next((x for x in solvers.children if x.key == n.key[1:] and x.children is not None), None)
                    result.update(inherited(ref, seen+(node.key,)))
                elif n.children is None:
                    result[n.key] = n
            return result
        normal = selected("TsolidCandidate")
        final = selected("TsolidCandidateFinal")
        for mode, node in (("normal", normal), ("final", final or normal)):
            for key, leaf in inherited(node).items():
                if key in ("maxIter", "minIter", "tolerance", "relTol"):
                    record(name, leaf, ("regions", p.parent.name, "solid", "linear", mode, key),
                           note="Resolved OpenFOAM solver selection/inheritance before removing numerical keys")
    rad = "constant/radiationProperties"
    # The radiation dictionary's frequency was the final authoritative override.
    move(rad,(),("couplingFrequency",),("thermalCoupling",),aliases={"couplingFrequency":"radiationCouplingFrequency"})
    doc = document(rad)
    rparent = ("model", "do")
    if doc and doc.find(rparent):
        ntheta = doc.find(rparent+("nTheta",))
        nphi = doc.find(rparent+("nPhi",))
        angles = int(numeric(ntheta.value, 0) if ntheta else 0)*int(numeric(nphi.value, 0) if nphi else 0)
        source = doc.find(rparent+("sourceRelativeTolerance",))
        source_raw = doc.scalar(source) if source else "1e-10"
        for key in ("maxSourceIterations", "sourceRelativeTolerance", "phaseBalanceTolerance",
                    "maxPhaseBalanceIterations", "radiationActiveEpsilonMin", "angleParallelThreads",
                    "sourceRelaxation", "linearSolverTolerance", "linearSolverRelTol", "linearSolverMaxIterations"):
            node = doc.find(rparent+(key,))
            if node:
                rule = "phase" if key=="phaseBalanceTolerance" else "radiation" if key in ("sourceRelativeTolerance","linearSolverTolerance") else None
                record(rad, node, ("radiation","do",key), rule=rule, angles=angles)
        if not doc.find(rparent+("linearSolverTolerance",)) and source:
            old_default = min(1e-12, numeric(source_raw, 1e-10)*.01)
            record(rad, source, ("radiation","do","linearSolverTolerance"), raw=format(old_default,".17g"),
                   rule="radiation", note="Old derived linear default min(1e-12,sourceRelativeTolerance*.01)", angles=angles)
        discard(rad, rparent, ("conservationRelativeTolerance", "maxPhaseScalingCorrection"))
    particle = "constant/particleProperties"
    move(particle, ("gpuResidentStuckModel",), ("solidificationIterations",), ("coldWall1D",))
    move(particle, (), ("gpuResidentPackingProjectionIterations",), ("particle","packing"),
         aliases={"gpuResidentPackingProjectionIterations":"iterations"})
    chmt = "constant/chmtProperties"
    if get(values,("thermalCoupling","couplingInterval")) is None:
        move(chmt,("multirate",),("couplingInterval",),("thermalCoupling",))
    else:
        discard(chmt,("multirate",),("couplingInterval",))
        for row in report["discarded"]:
            if row.get("source")==chmt and row.get("source_path")==["multirate","couplingInterval"]:
                row["reason"]="Shadowed by authoritative solidThermalCoupling interval; retained for historical restoration"
    move(chmt, ("cpuMaterial",), ("transportCfl","reactionCfl","reactionMaxSubstep","driveRelativeTolerance",
         "driveSourceFraction","driveTractionScale","maxSubsteps"), ("chmt","material"))
    move(chmt, ("multirate",), ("materialMaxSubstep",), ("chmt","material"), aliases={"materialMaxSubstep":"maxSubstep"})
    move(chmt, ("cpuMaterial",), ("nonlinearMaxIterations","nonlinearRelativeTolerance",
         "energyAbsoluteTolerance","energyRelativeTolerance"), ("chmt","material","conduction"))
    move(chmt, ("tolerances",), tuple(prefix+quantity for prefix in ("absolute","relative")
         for quantity in ("Mass","Energy","Temperature","Geometry")), ("chmt","tolerances"))
    discard(chmt, ("tolerances",), ("maxCouplingIterations","maxRetries"))
    move(chmt, (), ("spatialOrder",), ("chmt","material","reconstruction"))
    move(chmt, (), ("materialReconstruction",), ("chmt","material","reconstruction"), aliases={"materialReconstruction":"method"})
    move(chmt, ("particle",), ("packingIterations",), ("particle","packing"), aliases={"packingIterations":"iterations"})
    move(chmt, ("particle","contactCoeffs"), ("nonlinearIterations",), ("coldWall1D",), aliases={"nonlinearIterations":"solidificationIterations"})
    move(chmt, ("sst","boundaryLayer"), ("maxIterations","relativeTolerance","absoluteTolerance","nodes","stretch","quadratureRefinement"), ("wallLayer",),
         rules={"relativeTolerance":"wallRelative","absoluteTolerance":"wallAbsolute"})
    discard(chmt,("sst","boundaryLayer"),("geometryTolerance","maxRayCells"))
    move(chmt,("particle",),("maxFaceCrossings",),("particle","tracking"),aliases={"maxFaceCrossings":"maxFaceWalkHops"})
    move(chmt, (), ("minDt","cfl"), ("gasTimeStep",), aliases={"minDt":"minDeltaT","cfl":"maxCo"})
    doc = document(chmt)
    if doc:
        maxdt = doc.find(("maxDt",))
        gasdt = doc.find(("multirate","gasMaxDt"))
        control = document(ctrl)
        delta = control.find(("deltaT",)) if control else None
        if maxdt or gasdt:
            raw_max = doc.scalar(maxdt) if maxdt else None
            raw_gas = doc.scalar(gasdt) if gasdt else control.scalar(delta) if delta else None
            if numeric(raw_max) is not None and numeric(raw_gas) is not None:
                result = min(float(raw_max), float(raw_gas))
                record(chmt, maxdt or gasdt, ("gasTimeStep","maxDeltaT"), raw=format(result,".17g"),
                       note="Preserved min(old maxDt, old multirate.gasMaxDt or controlDict.deltaT)")
                if gasdt:
                    removed.setdefault(chmt,set()).add((gasdt.start,gasdt.end))
                report["chmt_caps"] = dict(maxDt=raw_max, gasMaxDt=raw_gas)
            else:
                report["notes"].append("CHMT caps require complete inherited controlDict before migration")
    changed_files = {}
    for name, spans in removed.items():
        doc = document(name)
        report["original_files"].setdefault(name, doc.text)
        report["input_sha256"].setdefault(name, hashlib.sha256(doc.text.encode()).hexdigest())
        text = doc.text
        for start, end in sorted(spans, reverse=True):
            text = text[:start]+text[end:]
        text = "\n".join(line.rstrip() for line in text.splitlines()).rstrip()+"\n"
        if text != doc.text:
            changed_files[name] = text
    report["notes"] = list(dict.fromkeys(report["notes"]))
    scheme = HEADER + "\n" + render(values) + "\n"
    if not scheme_file.is_file() or scheme_file.read_text() != scheme:
        changed_files["constant/scheme"] = scheme
    report["moves"].sort(key=lambda row: row["target"])
    report["output_sha256"] = dict(old.get("output_sha256", {}))
    report["output_sha256"].update({name:hashlib.sha256(text.encode()).hexdigest() for name,text in changed_files.items()})
    if write:
        for name,text in changed_files.items():
            path = case/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        if report["moves"] or report["discarded"] or old:
            mp.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    return report

def write_entry(text, path, value):
    doc=Document(text)
    existing=doc.find(path)
    body=path[-1]+" "+str(value)+";"
    if existing:
        return text[:existing.start]+body+text[existing.end:]
    parent=doc.find(path[:-1]) if len(path)>1 else None
    if parent and parent.children is not None:
        end=text.rfind("}",parent.start,parent.end)
        return text[:end]+"\n"+body+"\n"+text[end:]
    if len(path)>1:
        for key in reversed(path[:-1]):
            body=key+" { "+body+" }"
    return text+"\n"+body+"\n"

def restore_legacy_case(case):
    """Reinsert old controls in a copy, retaining later model/time/field edits."""
    case=Path(case); p=case/MANIFEST
    if not p.is_file():
        return case
    report=json.loads(p.read_text())
    values=tree((case/"constant/scheme").read_text())
    files={}
    for row in report["moves"]+report.get("discarded",[]):
        if row.get("note","").startswith("Old derived"):
            continue
        name=row["source"]; path=case/name
        if name not in files:
            files[name]=path.read_text() if path.is_file() else report["original_files"].get(name,"")
        # Reconstruct old requests, not already-floored FP32 effective values.
        value=get(values,row["target"],row["raw"]) if "target" in row else row["raw"]
        files[name]=write_entry(files[name],row.get("source_path",[row["key"]]),value)
    numerics=values.get("gasNumerics",{})
    if numerics:
        name="system/fvSchemes";path=case/name
        old_values=tree(path.read_text()) if path.is_file() else {}
        merge(old_values,numerics)
        files[name]="FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\n"+render(old_values)+"\n"
    # The old CHMT parser applies two independent caps. Restore both raw caps,
    # rather than rewriting maxDt with the merged minimum used by scheme.
    caps=report.get("chmt_caps",{})
    if caps and "constant/chmtProperties" in files:
        text=files["constant/chmtProperties"]
        if caps.get("maxDt") is not None:text=write_entry(text,("maxDt",),caps["maxDt"])
        if caps.get("gasMaxDt") is not None:text=write_entry(text,("multirate","gasMaxDt"),caps["gasMaxDt"])
        files["constant/chmtProperties"]=text
    for name,text in files.items():
        (case/name).write_text(text)
    return case

def merge_overlay_metadata(target, source, previous_scheme, previous_manifest):
    """copytree replaces files; reconstitute inherited scheme and provenance."""
    target, source = Path(target), Path(source)
    incoming = source/"constant/scheme"
    values = tree(previous_scheme) if previous_scheme else {}
    previous_manifest=previous_manifest or {}
    replaced={name for name in previous_manifest.get("original_files",{}) if (source/name).is_file()}
    for row in previous_manifest.get("moves",[]):
        if row["source"] in replaced:
            remove_path(values,row["target"])
            remove_path(values,("precisionOverrides","fp32",*row["target"]))
    previous_manifest=dict(previous_manifest,
        original_files={k:v for k,v in previous_manifest.get("original_files",{}).items() if k not in replaced},
        input_sha256={k:v for k,v in previous_manifest.get("input_sha256",{}).items() if k not in replaced},
        moves=[r for r in previous_manifest.get("moves",[]) if r["source"] not in replaced])
    if incoming.is_file():
        merge(values, tree(incoming.read_text()))
    if values:
        (target/"constant/scheme").write_text(HEADER+"\n"+render(values)+"\n")
    p = source/MANIFEST
    report = previous_manifest or {}
    if p.is_file():
        new = json.loads(p.read_text())
        originals = dict(report.get("original_files",{})); originals.update(new.get("original_files",{}))
        hashes = dict(report.get("input_sha256",{})); hashes.update(new.get("input_sha256",{}))
        moves = {tuple(x["target"]):x for x in report.get("moves",[])}
        moves.update({tuple(x["target"]):x for x in new.get("moves",[])})
        report = dict(new, original_files=originals, input_sha256=hashes, moves=list(moves.values()),
                      discarded=report.get("discarded",[])+new.get("discarded",[]))
    if report:
        (target/MANIFEST).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

def remove_control(case, section, key):
    p=Path(case)/"constant/scheme"
    data=tree(p.read_text()) if p.is_file() else {}
    path=tuple(section.split("."))+(key,)
    remove_path(data,path)
    remove_path(data,("precisionOverrides","fp32",*path))
    remove_path(data,("precisionOverrides","fp64",*path))
    p.write_text(HEADER+"\n"+render(data)+"\n")
    mp=Path(case)/MANIFEST
    if mp.is_file():
        report=json.loads(mp.read_text())
        report["moves"]=[row for row in report["moves"] if row["target"] != list(path)]
        mp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

def remove_path(values, path):
    parents=[]
    for key in path[:-1]:
        if key not in values or not isinstance(values[key],dict):
            return
        parents.append((values,key))
        values=values[key]
    values.pop(path[-1],None)
    for parent,key in reversed(parents):
        if not parent[key]:
            del parent[key]

def set_control(case, section, key, value):
    p = Path(case)/"constant/scheme"
    data = tree(p.read_text()) if p.is_file() else {}
    target=tuple(section.split("."))+(key,)
    put(data, target, value)
    remove_path(data, ("precisionOverrides","fp32",*target))
    remove_path(data, ("precisionOverrides","fp64",*target))
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(HEADER+"\n"+render(data)+"\n")
    # An explicit new-interface edit supersedes provenance-driven precision floors.
    mp = Path(case)/MANIFEST
    if mp.is_file():
        report=json.loads(mp.read_text())
        target=list(section.split("."))+[key]
        for row in report["moves"]:
            if row["target"] == target:
                row.update(raw=str(value),value=str(value),explicit_scheme_edit=True)
                row.pop("rule",None)
        mp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

def edit_legacy_entry(path, key, value):
    path=Path(path); case=path.parent.parent
    scheme=case/"constant/scheme"
    if not scheme.is_file():
        return False
    target=None
    if path.name=="controlDict":
        if key in ("maxCo","maxDeltaT","adjustTimeStep"): target=("gasTimeStep",key)
        elif key in ("samplingDeltaT","windowSamples","minimumWindows","stableWindows","meanTolerance","inventoryTolerance"): target=("steadyParticle",key)
        elif key in ("relativeTolerance","absoluteTolerance","minimumIterations","residualInterval","referenceTime"): target=("steadyGas",key)
    elif path.name=="schedulingProperties" and key=="gpuResidentMaxFaceWalkHops":
        target=("particle.tracking","maxFaceWalkHops")
    elif path.name=="chmtProperties" and key=="couplingInterval":
        target=("thermalCoupling","couplingInterval")
    elif path.name=="chmtProperties" and key=="maxFaceCrossings":
        target=("particle.tracking","maxFaceWalkHops")
    elif path.name=="chmtProperties":
        aliases={"gasMaxDt":"maxDeltaT","maxDt":"maxDeltaT","minDt":"minDeltaT","cfl":"maxCo"}
        if key in aliases: target=("gasTimeStep",aliases[key])
    if target is None:
        return False
    if path.name=="chmtProperties" and key=="gasMaxDt":
        mp=case/MANIFEST
        if mp.is_file():
            cap=json.loads(mp.read_text()).get("chmt_caps",{}).get("maxDt")
            if numeric(cap) is not None and numeric(value) is not None:
                value=min(float(cap),float(value))
    set_control(case,*target,value)
    if path.name=="controlDict" and key in ("maxCo","maxDeltaT"):
        current=tree(scheme.read_text())
        if get(current,("steadyGas",key)) is not None:
            set_control(case,"steadyGas",key,value)
    return True

def before_solver(command, legacy=False, candidate=None):
    """Used by validation launch helpers, not by production solver binaries."""
    argv=list(map(str,command))
    if "-case" not in argv:
        return
    known=("gasUGKP","gasUGKPSteady","CHT","CHMT","FSH")
    def solver_name(arg):
        return any(Path(arg).name == name or Path(arg).name.startswith(name+".") for name in known)
    if not any(solver_name(arg) for arg in argv[:argv.index("-case")]):
        return
    binary = next(Path(arg).resolve() for arg in argv[:argv.index("-case")] if solver_name(arg))
    # A historical baseline binary retains its historical input contract.
    # This is validation preparation, not a fallback in a production reader.
    if legacy or candidate is False or (candidate is None and not any((parent/"common/FoamSolverScheme.H").is_file() for parent in binary.parents)):
        restore_legacy_case(Path(argv[argv.index("-case")+1]))
        return
    migrate_case(Path(argv[argv.index("-case")+1]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--precision", type=int, choices=(32,64))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    args=parser.parse_args()
    report=migrate_case(args.case, args.precision, not args.dry_run)
    text=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if args.report:
        args.report.write_text(text)
    else:
        print(text,end="")
if __name__=="__main__":
    main()
