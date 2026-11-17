pass



import json
from pathlib import Path

def read_plan(path):
    return json.loads(Path(path).read_text())

def unresolved(plan):
    items = list(plan.get("pending", []))
    items += [p for p in plan["processes"] if p.get("argv") is None]
    return items

def frozen_execution_design():






    pass

if __name__ == "__main__":
    print("DRAFT ONLY: this skeleton cannot execute fixtures or solvers.")
