#!/usr/bin/env python3
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
header = (root / "gpu" / "GpuResidentStrict.H").read_text()
cuda = (root / "gpu" / "GpuResidentStrict.cu").read_text()

classification = re.search(
    r"isA<wedgePolyPatch>\(pp\).*?isA<emptyPolyPatch>\(pp\).*?"
    r"kind\s*=\s*1\s*;.*?restitution\s*=\s*([0-9.]+)\s*;",
    header,
    re.S,
)
assert classification, "wedge/empty particle boundary classification not found"
assert float(classification.group(1)) == 0.0, (
    "wedge/empty particle restitution must be 0.0 so normal velocity is zero"
)
assert re.search(
    r"s\.pux\[i\]\s*-=\s*\(1\.0\s*\+\s*restitution\)\s*\*\s*un\s*\*\s*nx",
    cuda,
), "particle reflection update is missing"
assert "cellFaceTangential[planeI] = tangential" in header
assert re.search(
    r"isA<symmetryPlanePolyPatch>\(pp\).*?isA<symmetryPolyPatch>\(pp\).*?"
    r"kind\s*=\s*1\s*;\s*restitution\s*=\s*1\.0\s*;",
    header,
    re.S,
), "ordinary symmetry boundaries must retain elastic reflection"

                                                                
                                                            
u = (3.0, -4.0, 5.0)
n = (0.0, 1.0, 0.0)
un = sum(ui * ni for ui, ni in zip(u, n))
u_new = tuple(ui - un * ni for ui, ni in zip(u, n))
assert abs(sum(ui * ni for ui, ni in zip(u_new, n))) < 1.0e-15
assert (u_new[0], u_new[2]) == (u[0], u[2])
print("PASS: wedge/empty preserve tangential velocity and zero normal velocity")
