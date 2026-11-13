"""Non-executing unified-D planning skeleton. NOT a numerical runner.
Read as a design artifact. Parent must implement final execution only after
all argv, activation predicates and acceptance rules have been frozen.
"""
import json
from pathlib import Path

def read_plan(path):
    return json.loads(Path(path).read_text())

def unresolved(plan):
    items = list(plan.get("pending", []))
    items += [p for p in plan["processes"] if p.get("argv") is None]
    return items

def frozen_execution_design():
    # 1. Parent verifies source/input/binary lineage and grants start.
    # 2. Execute ONLY explicit finalmanifest processes in dependency order.
    # 3. Record argv, exit, stdout, input hashes and actual consumer evidence.
    # 4. Preserve every result; no retries or adaptive overrides.
    # 5. Report inactive/missing modules as uncovered, never automatic PASS.
    # No subprocess import or executable dispatch is intentionally present.
    pass

if __name__ == "__main__":
    print("DRAFT ONLY: this skeleton cannot execute fixtures or solvers.")
