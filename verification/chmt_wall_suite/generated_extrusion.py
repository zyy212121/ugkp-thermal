"""Certify and encode only the suite's declared axis-aligned single-block extrusion.

This is input construction after blockMesh, never a solver-side GCL repair.
Pair identities come from mesh edges, not rounded-coordinate nearest neighbors.
"""
import hashlib,json,math,re,sys
from pathlib import Path


def _list(text):
    match=re.search(r"\n\s*(\d+)\s*\n\(\s*\n(.*?)\n\)",text,re.S)
    if not match:raise ValueError("expected an ASCII mesh list")
    return match,int(match.group(1)),match.group(2)


def canonicalize(case,spec):
    if spec.get("generator")!="flatplate_moving" or not spec.get("mesh_commands"):
        raise ValueError("extrusion encoding is restricted to generated moving single-block plates")
    case=Path(case);p=spec["parameters"];width=p["width"]
    if not math.isfinite(width) or width<=0:raise ValueError("invalid extrusion width")
    records=[];pending=[]
    for region,ny,y0,y1 in [("",p["ny"],0.,p["height"]),("solid",p["solid_ny"],-.05,0.)]:
        mesh=case/"constant"/region/"polyMesh";path=mesh/"points";text=path.read_text()
        match,count,body=_list(text)
        points=[list(map(float,x.split())) for x in re.findall(r"\(([^()]+)\)",body)]
        if count!=2*(p["nx"]+1)*(ny+1) or len(points)!=count or any(len(x)!=3 or not all(map(math.isfinite,x)) for x in points):
            raise ValueError("unexpected generated extrusion point layout")
        normal_tol=64*sys.float_info.epsilon*width
        tangent_tol=64*sys.float_info.epsilon*max(p["length"],abs(y0),abs(y1))
        layers=[]
        for x,y,z in points:
            layer=0 if abs(z)<abs(z-width) else 1
            if abs(z-layer*width)>normal_tol or not -tangent_tol<=x<=p["length"]+tangent_tol or not y0-tangent_tol<=y<=y1+tangent_tol:
                raise ValueError("point differs from declared extrusion beyond representation roundoff")
            layers.append(layer)
        _,nf,body=_list((mesh/"faces").read_text())
        faces=re.findall(r"(\d+)\s*\(([^()]+)\)",body)
        if len(faces)!=nf:raise ValueError("invalid face list")
        mates={}
        for size,vertices in faces:
            ids=list(map(int,vertices.split()))
            if len(ids)!=int(size) or len(ids)!=4 or any(i<0 or i>=count for i in ids):raise ValueError("not a generated hexahedral face")
            for a,b in zip(ids,ids[1:]+ids[:1]):
                if layers[a]==layers[b]:continue
                if a in mates and mates[a]!=b or b in mates and mates[b]!=a:raise ValueError("ambiguous layer pairing")
                if any(abs(points[a][j]-points[b][j])>tangent_tol for j in (0,1)):raise ValueError("nonextruded cross-layer edge")
                mates[a]=b;mates[b]=a
        if len(mates)!=count:raise ValueError("unpaired extrusion points")
        updated=[x[:] for x in points]
        for i,layer in enumerate(layers):
            updated[i][2]=layer*width
            if layer:updated[i][:2]=points[mates[i]][:2]
        new_body="\n".join("("+" ".join(format(v,".17g") for v in x)+")" for x in updated)
        result=text[:match.start(2)]+new_body+text[match.end(2):]
        changes=[dict(point=i,before=a,after=b) for i,(a,b) in enumerate(zip(points,updated)) if a!=b]
        records.append(dict(region=region or "gas",point_count=count,changed_points=len(changes),changes=changes,
            maximum_displacement_m=max(math.dist(a,b) for a,b in zip(points,updated)),
            before_sha256=hashlib.sha256(text.encode()).hexdigest(),after_sha256=hashlib.sha256(result.encode()).hexdigest()))
        pending.append((path,text,result))
    # Validate both regions before any write. Preserve exact original bytes.
    evidence=case/"mesh-construction-evidence";evidence.mkdir(exist_ok=True)
    for (path,text,result),record in zip(pending,records):
        backup=evidence/(record["region"]+"-points-"+record["before_sha256"]+".txt")
        if not backup.exists():backup.write_text(text)
        elif backup.read_text()!=text:raise ValueError("existing input backup does not match")
        path.write_text(result)
    manifest=evidence/(hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()+".json")
    manifest.write_text(json.dumps(records,indent=2)+"\n")
    return records
