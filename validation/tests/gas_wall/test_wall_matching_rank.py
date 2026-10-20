"""Roundoff in an unresolved extrusion direction must not become a fit mode."""
import pytest
from test_wall_geometry import compile_run


@pytest.mark.parametrize('noise', [1e-17, 1e-16])
def test_matching_rank_ignores_roundoff_only_extrusion_modes(tmp_path, noise):
    compile_run(tmp_path, r'''
MeshBuilder b;
const double dx=.002,dy=.001,dz=.00025,offset=.7;
for(int k=0;k<5;++k)for(int i=0;i<5;++i){
 const double x=offset+i*dx,z=k*dz;
 V a={x,0,z},bb={x+dx,0,z},c={x+dx,dy,z},d={x,dy,z};
 V e={x,0,z+dz},f={x+dx,0,z+dz},g={x+dx,dy,z+dz},h={x,dy,z+dz};
 b.cell({{a,bb,c,d},{e,f,g,h},{a,bb,f,e},{bb,c,g,f},{c,d,h,g},{d,a,e,h}},
     {x+dx*.5,dy*.5,z+dz*.5},dx*dy*dz);
}
int wall=-1;
for(int f=0;f<int(b.m.owner.size());++f){bool bottom=true;
 for(int j=b.m.faceOffsets[f];j<b.m.faceOffsets[f+1];++j)
     bottom=bottom&&b.m.points[b.m.facePoints[j]][2]==0;
 if(bottom){b.m.physicalWall[f]=1;if(b.m.owner[f]==2)wall=f;}}
assert(wall>=0);
WallGeometryDescriptor<double> reference;WallStatus status;
WallGeometryOptions<double> options;options.reconstructMatching=true;options.constructQuadrature=false;
assert(buildWallGeometry(b.m,wall,reference,status,options));
auto mesh=b.m;
// Coordinates from a real one-cell extrusion contain these double-precision
// differences. They resolve no extra physical direction in the donor stencil.
for(int j=0;j<int(mesh.points.size());++j)
    mesh.points[j][1]+=NOISE*std::sin(double(j)*1.23);
for(int j=0;j<int(mesh.cellCentres.size());++j)
    mesh.cellCentres[j][1]+=NOISE*std::sin(double(j)*.91);
WallGeometryDescriptor<double> actual;
assert(buildWallGeometry(mesh,wall,actual,status,options));
assert(actual.matchingCells==reference.matchingCells);
assert(actual.samplingOrder==2);
double linear=0,quadratic=0,weightChange=0;
for(size_t j=0;j<actual.matchingCells.size();++j){
 const auto& c=mesh.cellCentres[actual.matchingCells[j]];
 const double x=(c[0]-offset)/dx,z=c[2]/dz,w=actual.matchingWeights[j];
 linear+=w*(2+3*x+4*z);
 quadratic+=w*(2+3*x+4*z+.7*(x*x+1./12.)+.9*(z*z+1./12.)+.4*x*z);
 weightChange=std::max(weightChange,std::abs(w-reference.matchingWeights[j]));
}
std::cerr<<"maximum matching weight change="<<weightChange<<"\n";
assert(weightChange<1e-9);
const double x=(actual.matchingPoint[0]-offset)/dx,z=actual.matchingPoint[2]/dz;
assert(near(linear,2+3*x+4*z,1e-11));
assert(near(quadratic,2+3*x+4*z+.7*x*x+.9*z*z+.4*x*z,1e-11));
'''.replace('NOISE', repr(noise)))