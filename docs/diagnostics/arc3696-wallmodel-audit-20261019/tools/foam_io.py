from pathlib import Path
import re
import numpy as np
def strip_comments(s):return re.sub(r'/\*.*?\*/|//[^\n]*','',s,flags=re.S)
def list_body(s):
 m=re.search(r'\n\s*\d+\s*\(\s*\n(.*)\n\s*\)\s*(?:\n|$)',strip_comments(s),re.S);assert m;return m.group(1)
def read_field(p,expected=None,vector=False):
 s=strip_comments(p.read_text());m=re.search(r'internalField\s+uniform\s+([^;]+);',s)
 if m:
  assert expected is not None;v=np.fromstring(m.group(1).strip(' ()\n'),sep=' ');return np.tile(v,(expected,1))if vector else np.full(expected,v[0])
 kind='vector'if vector else'scalar';m=re.search(r'internalField\s+nonuniform\s+List<'+kind+r'>\s+(\d+)\s*\((.*?)\)\s*;',s,re.S);assert m,p
 a=np.fromstring(m.group(2).replace('(',' ').replace(')',' '),sep=' ');a=a.reshape((-1,3))if vector else a;assert len(a)==int(m[1]);assert expected is None or len(a)==expected;return a
def numeric_dirs(case):
 out={}
 for p in case.iterdir():
  if not p.is_dir():continue
  try:t=float(p.name)
  except ValueError:continue
  out[round(t,12)]=p
 return out
