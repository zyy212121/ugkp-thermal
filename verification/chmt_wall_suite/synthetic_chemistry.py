"""Canonical SI metadata for an irreversible first-order equal-thermo isomerization.

Hashing follows the documented production file-format identity, not a physical
reference calculation. The independent numeric oracle is exponential decay.
"""
import hashlib
import json
from pathlib import Path
import re
import struct

def fnv(data,h=14695981039346656037):
    for b in data:h=((h^b)*1099511628211)&((1<<64)-1)
    return h

def mechanism(names,rate=2.):
    species_hash=fnv(b''.join(n.encode()+b'\0' for n in names));h=species_hash
    def scalar(x):
        nonlocal h;h=fnv(struct.pack('<d',float(x) if x else 0.),h)
    def byte(x):
        nonlocal h;h=fnv(bytes([x]),h)
    for x in [101325,1,-1,1]:scalar(x)
    for x in [0,0,0]:byte(x)
    for x in [0,rate,0,0,0,0,0,1,0,0,0,0]:scalar(x)
    byte(0)
    for x in [1,0,1,1,1,1,0]:scalar(x)
    source={'classification':'controlled_synthetic','reaction':f'{names[0]} -> {names[1]}','rate_per_s':rate,'molar_mass_kg_mol':.028,'equal_calorics':True}
    digest=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    text=f'''schemaVersion 1; units siMolar; sourceSha256 "{digest}";
phase synthetic; species ({' '.join(names)}); referencePressure 101325;
speciesOrderHash {species_hash}; mechanismHash {h};
independentRank 1; stoichiometricBasis (-1 1);
reactions {{ reaction0 {{ type elementary; reversible false; duplicate false; sourceIndex 0;
 reactants (0 1); products (1 1); efficiencies (); highRate ({rate} 0 0); lowRate (0 0 0);
 defaultEfficiency 1; troe (0 0 0 0); troeHasT2 false; }} }}
'''
    return text,source

def enable(case,names,rate=2.):
    case=Path(case);p=case/'constant/gasModelProperties';text=p.read_text().replace('gasMode mixtureFrozen;','gasMode mixtureChemistry;')
    if not re.search(r'\belements\s*\(',text):text+='\nelements (X);\n';text=text.replace('coefficients (1040 0 0);','coefficients (1040 0 0); atoms (1);')
    text+='\nmechanism synthetic.mechanism; phase synthetic;\nchemistryControls { relativeTolerance 1e-7; absoluteMassFractionTolerance 1e-13; absoluteTemperatureTolerance 1e-6; maximumSteps 100000; }\n'
    p.write_text(text);model,source=mechanism(names,rate);(case/'constant/synthetic.mechanism').write_text(model);(case/'synthetic_chemistry.json').write_text(json.dumps(source,indent=2)+'\n')
    props=case/'constant/chmtProperties'
    if props.exists():
        s=re.sub(r'modelFingerprint\s+"[^"]+";','',props.read_text());s+='modelFingerprint "'+hashlib.sha256((s+text+model).encode()).hexdigest()+'";\n';props.write_text(s)
    return source
