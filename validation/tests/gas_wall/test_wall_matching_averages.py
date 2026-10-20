'''Finite-volume matching must reproduce point values from cell moments.'''
from test_wall_geometry import compile_run

def test_normal_quadratic_average_is_not_used_as_point_value(tmp_path):
    compile_run(tmp_path,r'''
auto m=cubes(5);WallGeometryDescriptor<double>d;WallStatus status;WallGeometryOptions<double> opt;opt.reconstructMatching=true;assert(buildWallGeometry(m,0,d,status,opt));
double q=0;
for(size_t j=0;j<d.matchingCells.size();++j){double z=m.cellCentres[d.matchingCells[j]][2];q+=d.matchingWeights[j]*(2+3*z+4*(z*z+1./12.));}
double z=d.matchingPoint[2];assert(near(q,2+3*z+4*z*z,1e-12));
''')

def test_resolved_three_dimensional_quadratic_matches_actual_volume_moments(tmp_path):
    compile_run(tmp_path,r'''
MeshBuilder b;
for(int z=0;z<5;++z)for(int y=0;y<3;++y)for(int x=0;x<3;++x){
V a={double(x),double(y),double(z)},bb=add(a,{1,0,0}),c=add(a,{1,1,0}),d=add(a,{0,1,0});
V e=add(a,{0,0,1}),f=add(a,{1,0,1}),g=add(a,{1,1,1}),h=add(a,{0,1,1});
b.cell({{a,bb,c,d},{e,f,g,h},{a,bb,f,e},{bb,c,g,f},{c,d,h,g},{d,a,e,h}},add(a,{.5,.5,.5}),1);}
int wall=-1;
for(int f=0;f<int(b.m.owner.size());++f){bool bottom=true;for(int j=b.m.faceOffsets[f];j<b.m.faceOffsets[f+1];++j)bottom=bottom&&b.m.points[b.m.facePoints[j]][2]==0;
 if(bottom){b.m.physicalWall[f]=1;if(b.m.owner[f]==4)wall=f;}}
WallGeometryDescriptor<double>d;WallStatus status;WallGeometryOptions<double> opt;opt.reconstructMatching=true;assert(buildWallGeometry(b.m,wall,d,status,opt));
auto value=[](V p){return 3+2*p[0]-p[1]+4*p[2]+.7*p[0]*p[0]+.3*p[1]*p[1]+.9*p[2]*p[2]+.4*p[0]*p[2]-.2*p[1]*p[2];};
double actual=0;for(size_t j=0;j<d.matchingCells.size();++j){V c=b.m.cellCentres[d.matchingCells[j]];actual+=d.matchingWeights[j]*(value(c)+( .7+.3+.9)/12.);}
assert(near(actual,value(d.matchingPoint),1e-11));
''')


def test_volume_closure_ends_at_owner_bulk_interface(tmp_path):
    compile_run(tmp_path,r'''
auto m=cubes(5);WallGeometryDescriptor<double>d;WallStatus status;
WallGeometryOptions<double> opt;opt.reconstructMatching=true;assert(buildWallGeometry(m,0,d,status,opt));
assert(near(d.matchingDistance,d.maximumOwnerDistance,1e-14));
for(auto cell:d.matchingCells)assert(cell!=d.ownerCell);
''')
