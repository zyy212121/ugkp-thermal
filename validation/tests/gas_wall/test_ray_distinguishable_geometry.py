from test_wall_geometry import compile_run


def test_shallow_nonparallel_ray_preserves_real_hit(tmp_path):
    compile_run(tmp_path, r'''
 double time=-99;
 assert(wallGeometryDetail::rayTriangle(V{.25,.25,1e-17},V{1,0,-1e-16},V{0,0,0},V{1,0,0},V{0,1,0},time));
 assert(std::abs(time-.1)<1e-16);
 assert(!wallGeometryDetail::rayTriangle(V{.25,.25,1e-17},V{1,0,0},V{0,0,0},V{1,0,0},V{0,1,0},time));
''')


def test_thin_represented_cells_keep_positive_matching_interval(tmp_path):
    compile_run(tmp_path, r'''
 for(double height:{1e-12,5e-13,2e-13,1e-13,1e-20,1e-100,1e-200}){
  auto m=cubes();for(auto&p:m.points)p[2]*=height;
  for(auto&p:m.cellCentres)p[2]*=height;for(auto&v:m.volumes)v*=height;
  WallGeometryOptions<double> options;options.constructQuadrature=false;
  WallGeometryDescriptor<double>d;WallStatus s;
  assert(buildWallGeometry(m,0,d,s,options));assert(s.code==WallCode::Success);
  assert(d.matchingCells==std::vector<int>{1});
  assert(std::abs(d.matchingDistance/height-1.5)<1e-14);
  assert(d.matchingDistance>d.maximumOwnerDistance);
  assert(d.geometryVolume>0&&d.volume==height);
 }
''')


def test_shared_triangles_and_actual_distinct_face_tie(tmp_path):
    compile_run(tmp_path, r'''
 auto m=cubes();WallGeometryOptions<double>options;options.constructQuadrature=false;
 wallGeometryDetail::Context<double>ctx;WallStatus s;
 assert(wallGeometryDetail::prepare(m,ctx,s,options));
 WallGeometryDescriptor<double>d;d.ownerCell=0;d.wallFace=0;d.origin={.5,.5,0};d.normal={0,0,1};d.maximumOwnerDistance=1;
 // Both diagonal-adjacent triangles of each horizontal face meet this ray.
 assert(wallGeometryDetail::matching(m,ctx,d,s,options,1.));assert(d.matchingCells==std::vector<int>{1});
 d=WallGeometryDescriptor<double>{};d.ownerCell=0;d.wallFace=0;d.origin={.5,.5,0};d.normal={.5-1e-14,0,1};d.maximumOwnerDistance=1;
 // Close but distinct top/right crossings delimit a real positive donor segment.
 assert(wallGeometryDetail::matching(m,ctx,d,s,options,1.));assert(d.matchingCells==std::vector<int>{1});
 assert(d.matchingDistance>1&&d.matchingDistance<1+1e-13);
 d=WallGeometryDescriptor<double>{};d.ownerCell=0;d.wallFace=0;d.origin={.5,.5,0};d.normal={.5,0,1};d.maximumOwnerDistance=1;
 // The ray reaches top and right faces together: there is no unique next cell.
 assert(!wallGeometryDetail::matching(m,ctx,d,s,options,1.));assert(s.code==WallCode::InvalidGeometry);
''')


def test_positive_thin_face_and_actual_invalid_polygon_are_distinct(tmp_path):
    compile_run(tmp_path, r'''
 WallGeometryMesh<double>m;m.faceOffsets={0,4};m.facePoints={0,1,2,3};
 WallGeometryOptions<double>options;wallGeometryDetail::Face<double>face;
 m.points={{0,0,0},{1,0,0},{1,1e-100,0},{0,1e-100,0}};
 assert(wallGeometryDetail::triangulate(m,0,face,options));assert(face.triangles.size()==2);
 face={};m.points={{0,0,0},{2,2,0},{0,2,0},{1,0,0}};
 assert(!wallGeometryDetail::triangulate(m,0,face,options));
 face={};m.points={{0,0,0},{1,0,0},{2,0,0},{3,0,0}};
 assert(!wallGeometryDetail::triangulate(m,0,face,options));
 face={};m.points={{0,0,0},{1,0,0},{1,1,0},{0,1,0}};m.facePoints={0,1,2,1};
 assert(!wallGeometryDetail::triangulate(m,0,face,options));
''')
