"""Exercise one-owner maintenance propagation without compiling CUDA.

A change to an error return in the first application's protocol must reach every
consumer of that protocol. This catches reintroduced application-local copies,
even if they happen to contain the same text at the time of the check.
"""
from pathlib import Path
import re
import unittest

REPO = Path(__file__).resolve().parents[1]
APPLICATIONS = {
    "gas": Path("applications/gasUGKP/private_backend/GpuResidentStrict.cu"),
    "fsh": Path("applications/FSH/private_backend/GpuResidentStrict.cu"),
    "cht": Path("applications/CHT/gpu/GpuResidentStrict.cu"),
}
PROTOCOLS = {
    "advanceGasEulerSubstage": ("gas", "fsh", "cht"),
    "blendGasRungeKuttaStage": ("gas", "fsh", "cht"),
    "advanceGasFluxStage": ("gas", "fsh", "cht"),
    "finaliseGasBoundaryStage": ("gas", "fsh", "cht"),
    "advancePureGasGraph": ("gas", "fsh", "cht"),
    "applyMobilePackingProjection": ("gas", "fsh", "cht"),
    "binParticlesByCell": ("fsh", "cht"),
    "buildSplitPreDirectory": ("fsh", "cht"),
    "launchToolB1CellBundle": ("gas", "fsh", "cht"),
    "launchToolB1FaceBundle": ("gas", "fsh", "cht"),
}


def mask_non_code(text):
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern, lambda m: "".join("\n" if c == "\n" else " " for c in m[0]), text)


def definition(text, name):
    code = mask_non_code(text)
    match = re.search(r"^int\s+" + re.escape(name) + r"\b[^;{]*\{", code, re.M)
    if match is None:
        return None
    start = match.start()
    end = code.index("{", start) + 1
    depth = 1
    while depth:
        depth += (code[end] == "{") - (code[end] == "}")
        end += 1
    return start, end


def resolve_definition(path, name, overrides=None, visited=None):
    overrides = overrides or {}
    visited = visited if visited is not None else set()
    path = path.resolve()
    if path in visited or not path.is_file():
        return None
    visited.add(path)
    text = overrides.get(path)
    if text is None:
        text = path.read_text()
    span = definition(text, name)
    if span is not None:
        return path, span, text
    for include in re.findall(r'^\s*#include\s+"([^"\n]+)"', text, re.M):
        found = resolve_definition(path.parent / include, name, overrides, visited)
        if found is not None:
            return found
    return None


class SharedHostMaintenanceTest(unittest.TestCase):
    def test_single_edit_reaches_every_host_protocol_consumer(self):
        tested = 0
        for name, consumers in PROTOCOLS.items():
            available = [app for app in consumers if (REPO / APPLICATIONS[app]).is_file()]
            if not available:
                continue
            tested += 1
            with self.subTest(protocol=name):
                first = resolve_definition(REPO / APPLICATIONS[available[0]], name)
                self.assertIsNotNone(first, f"cannot locate {name}")
                owner, (start, end), source = first
                body = source[start:end]
                self.assertIn("return 1;", body, f"{name} has no error-return mutation site")
                mutated = body.replace("return 1;", "return 97;", 1)
                overrides = {owner: source[:start] + mutated + source[end:]}
                for app in available:
                    found = resolve_definition(REPO / APPLICATIONS[app], name, overrides)
                    self.assertIsNotNone(found, f"cannot resolve {app}/{name}")
                    app_owner, (app_start, app_end), app_source = found
                    self.assertTrue(
                        "return 97;" in app_source[app_start:app_end],
                        f"{app}/{name} did not consume the edit to {owner.relative_to(REPO)}; "
                        f"its independently maintained definition is {app_owner.relative_to(REPO)}",
                    )
                # One-application distributions must still consume a common owner.
                self.assertTrue(
                    owner.is_relative_to(REPO / "common"),
                    f"{name} is still owned by an application-local implementation",
                )
        self.assertGreater(tested, 0)


if __name__ == "__main__":
    unittest.main()
