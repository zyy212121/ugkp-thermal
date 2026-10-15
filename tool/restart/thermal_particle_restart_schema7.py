from __future__ import annotations
import json,struct,hashlib,sys
from pathlib import Path
import numpy as np
MARKERS={"UGKP_THERMAL_PARTICLES_SCHEMA7_T32_BIN":"<f4","UGKP_THERMAL_PARTICLES_SCHEMA7_T64_BIN":"<f8"}
MAXIMUM_CHUNK_PARTICLES=262144
FLOAT_FIELDS=("px","py","pz","pux","puy","puz","pT","pTheta","pd","pm")
def field_specs(marker):
    if marker not in MARKERS:raise ValueError("unsupported schema7 time marker")
    time=MARKERS[marker]
    return ([(n,"<f8",1) for n in FLOAT_FIELDS]+[("cell","<i4",1),("status","<i4",1),("rng","<u8",1),("orig_id","<u8",1),("pStuck","u1",1),("pStuckFaceId","<i4",1),("pDepositionArea","<f4",1),("pContactDuration",time,1),("pContactMaximumArea","<f4",1),("pContactPeakFraction","<f4",1),("pColdNodeSpecificEnthalpy","<f4",8),("pColdRingSolidMass","<f4",8),("pColdFrozenArea","<f4",1),("pColdContactAge",time,1),("pCold2DNodeSpecificEnthalpy","<f4",64),("pCold2DRingContactAge",time,8),("pCold2DFrozenArea","<f4",1)]+[(n,"<f8",1) for n in ("puxOld","puyOld","puzOld")])
def iter_chunks(path):
    path=Path(path)
    with path.open("rb") as f:
        header=f.readline(256)
        if not header.endswith(b"\n"):raise ValueError("invalid schema7 header terminator or length")
        tokens=header.decode("ascii").split()
        if len(tokens)!=3:raise ValueError("invalid schema7 header")
        marker=tokens[0];specs=field_specs(marker);total,bound=map(int,tokens[1:])
        if total<0 or not 0<bound<=MAXIMUM_CHUNK_PARTICLES:raise ValueError("invalid schema7 count/bound")
        done=0
        while done<total:
            raw=f.read(4)
            if len(raw)!=4:raise ValueError("truncated schema7 chunk count")
            n=struct.unpack("<I",raw)[0]
            if n==0 or n>bound or n>total-done:raise ValueError("invalid schema7 chunk size")
            data={}
            for name,dtype,width in specs:
                a=np.fromfile(f,dtype=dtype,count=n*width)
                if len(a)!=n*width:raise ValueError("truncated schema7 field "+name)
                if a.dtype.kind=="f" and not np.all(np.isfinite(a)):raise ValueError("nonfinite schema7 field "+name)
                data[name]=a if width==1 else a.reshape(n,width)
            done+=n
            yield marker,data
        if f.read(1):raise ValueError("trailing schema7 data")
def summarize(path):
    total=transient=0;mass=0.;marker=None
    for marker,data in iter_chunks(path):
        total+=len(data["pm"]);mass+=float(np.sum(data["pm"],dtype=np.float64));transient+=int(np.count_nonzero((data["pStuck"]==2)|(data["pStuck"]==3)))
    return dict(marker=marker,particles=total,transient_contacts=transient,statistical_mass_kg=mass,saved_velocity_present=True,scope="Structural checkpoint summary; not a whole-system physical conservation certificate")
if __name__=="__main__":print(json.dumps(summarize(Path(sys.argv[1])),indent=2))
