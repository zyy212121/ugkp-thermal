#!/usr/bin/env python3
"""Emit a diagnostic-only patch for an actual CHMT Backend.cu; do not apply it."""
import argparse
import difflib
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('backend',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
src=a.backend.read_text()
header=(Path(__file__).parent/'clock_trace.H').read_text()
replacements=[
 ('#include <vector>\n', '#include <vector>\n'+header+'\n'),
 ('    static int begin(Backend* b){\n', '    static int begin(Backend* b){\n        fp64_clock_trace::begin(b->program.interval,b->history.records().size()+1,b->microTime,b->microDt);\n'),
 ('    std::vector<GasPrimitive> primitive;std::vector<GasGradient> gradient;std::vector<GasQ> flux;\n',
  '    fp64_clock_trace::wall(b->program.interval,b->record.microSequence,stageTime,b->stageTime,dt,wall);\n    std::vector<GasPrimitive> primitive;std::vector<GasGradient> gradient;std::vector<GasQ> flux;\n'),
 ('b.reserve.commitPrepared(std::move(reservation));',
  'fp64_clock_trace::accepted(b.program.interval,b.record,b.trial);\n    b.reserve.commitPrepared(std::move(reservation));'),
]
modified=src
for old,new in replacements:
    count=modified.count(old)
    if count!=1:
        raise SystemExit(f'Diagnostic patch anchor count {count}, expected one: {old[:90]!r}; source unchanged')
    modified=modified.replace(old,new,1)
path='applications/CHMT/gpu/Backend.cu'
a.output.write_text(''.join(difflib.unified_diff(src.splitlines(True),modified.splitlines(True),fromfile='a/'+path,tofile='b/'+path)))
print(a.output)
