#!/usr/bin/env python3
"""Initial conforming tetrahedral native polyMesh fixture (no runtime remeshing).
Each structured block centre is connected to its globally triangulated faces.
The x-normal 2x2 interface uses a full-rank three-plus-one diagonal pattern.
"""
import pathlib,sys,itertools
root=pathlib.Path(sys.argv[1]);nx=int(sys.argv[2]) if len(sys.argv)>2 else 4

def subtract(a,b):return tuple(x-y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def build(x0,x1,n,dest):
 points=[]; index={}
 for k in range(3):
  for j in range(3):
   for i in range(n+1):
    index[i,j,k]=len(points);points.append((x0+(x1-x0)*i/n,.005*j,.005*k))
 cells=[]
 for i,j,k in itertools.product(range(n),range(2),range(2)):
  center=len(points);points.append((x0+(x1-x0)*(i+.5)/n,.005*(j+.5),.005*(k+.5)))
  quads=[]
  for axis in range(3):
   for side in range(2):
    lo=[i,j,k];lo[axis]+=side;tangent=[a for a in range(3) if a!=axis]
    vertices=[]
    for a,b in [(0,0),(1,0),(1,1),(0,1)]:
     p=lo.copy();p[tangent[0]]+=a;p[tangent[1]]+=b;vertices.append(index[tuple(p)])
    flip=axis==0 and j==1 and k==1
    triangles=[(vertices[0],vertices[1],vertices[3]),(vertices[1],vertices[2],vertices[3])] if flip else [(vertices[0],vertices[1],vertices[2]),(vertices[0],vertices[2],vertices[3])]
    for tri in triangles:cells.append((center,)+tri)
 faces={}
 for owner,cell in enumerate(cells):
  for omitted in range(4):
   face=[cell[a] for a in range(4) if a!=omitted]
   a,b,c=[points[v] for v in face]
   if dot(cross(subtract(b,a),subtract(c,a)),subtract(points[cell[omitted]],a))>0:face.reverse()
   key=tuple(sorted(face))
   if key in faces:assert faces[key][2] is None;faces[key][2]=owner
   else:faces[key]=[face,owner,None]
 internal=[]; patches={name:[] for name in ['left','right','sides']}
 for face,owner,neighbor in faces.values():
  if neighbor is not None:internal.append((face,owner,neighbor))
  else:
   xs=[points[v][0] for v in face];kind='left' if all(abs(x-x0)<1e-12 for x in xs) else ('right' if all(abs(x-x1)<1e-12 for x in xs) else 'sides')
   patches[kind].append((face,owner,None))
 internal.sort(key=lambda entry:(entry[1],entry[2]))
 ordered=internal+sum(patches.values(),[]);dest.mkdir(parents=True,exist_ok=True)
 def output(name,cls,items):
  header=f'FoamFile {{ version 2.0; format ascii; class {cls}; object {name}; }}\n'
  (dest/name).write_text(header+str(len(items))+'\n(\n'+'\n'.join(items)+'\n)\n')
 output('points','vectorField',['('+' '.join(format(v,'.17g') for v in p)+')' for p in points])
 output('faces','faceList',['3('+' '.join(map(str,f))+')' for f,o,nbr in ordered])
 output('owner','labelList',[str(o) for f,o,nbr in ordered]);output('neighbour','labelList',[str(nbr) for f,o,nbr in internal])
 start=len(internal);entries=[]
 for name,items in patches.items():
  entries.append(f'{name}\n{{ type wall; nFaces {len(items)}; startFace {start}; }}');start+=len(items)
 output('boundary','polyBoundaryMesh',entries)
 # Empty generated block zones describe a different topology; retain no zones.
 for name in ['cellZones','faceZones','pointZones']:
  if (dest/name).exists():(dest/name).unlink()
 print(f'native tetra fixture {dest}: {len(cells)} cells, {len(points)} points, {len(ordered)} faces')
build(0,.02,nx,root/'constant'/'polyMesh')
build(-.004,0,max(2,nx//3),root/'constant'/'solid'/'polyMesh')
