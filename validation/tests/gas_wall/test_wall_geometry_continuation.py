"""Real triangulation, descriptor publication and uncapped ray traversal."""
from test_wall_geometry import compile_run


def test_finite_certificates_publish_complete_actual_geometry(tmp_path):
    compile_run(tmp_path, r'''
for(int mutation=0;mutation<4;++mutation)for(bool quadrature:{false,true}){
 auto m=cubes();
 if(mutation==0)m.volumes[0]=.9;
 if(mutation==1)m.cellCentres[0][2]=.4;
 // A warped side face still has a complete existing ear-clipped surface.
 if(mutation==2)m.points[6][0]+=.01;
 if(mutation==3)m.physicalWall[2]=1;
 WallGeometryOptions<double> o;o.constructQuadrature=quadrature;o.relativeTolerance=0;
 WallGeometryDescriptor<double>d;d.wallFace=-99;d.matchingCells={-99};d.distance={-99};WallStatus s;
 assert(buildWallGeometry(m,0,d,s,o));assert(s.code==WallCode::Success);
 assert(d.wallFace==0&&d.ownerCell==0&&d.matchingCells==std::vector<int>{1});
 assert(d.matchingWeights==std::vector<double>{1});assert(std::isfinite(d.firstMoment));
 assert(d.volume==m.volumes[0]&&d.geometryVolume>0);
 if(quadrature){assert(d.distance.size()>4);assert(d.distance.size()==d.volumeWeight.size());assert(d.distance.size()==d.quadraturePoints.size());
  double volume=0,moment=0;for(size_t q=0;q<d.distance.size();++q){volume+=d.volumeWeight[q];moment+=d.distance[q]*d.volumeWeight[q];}
  assert(near(volume,m.volumes[0]));assert(near(moment,d.firstMoment));
 }else assert(d.distance.empty()&&d.volumeWeight.empty()&&d.quadraturePoints.empty());
 if(mutation<2){assert(near(d.geometryVolume,1));assert(near(d.firstMoment,.5*m.volumes[0]));}
}
''')


def test_ray_finishes_beyond_old_budget_or_reports_actual_end(tmp_path):
    compile_run(tmp_path, r'''
auto m=cubes(70);m.omegaConstrained.assign(70,1);m.omegaConstrained.back()=0;
WallGeometryOptions<double> o;o.maxRayCells=1;WallGeometryDescriptor<double>d;WallStatus s;
assert(buildWallGeometry(m,0,d,s,o));assert(d.matchingCells==std::vector<int>{69});assert(near(d.matchingDistance,69.5));
const auto keep=d.matchingCells;const double distance=d.matchingDistance;
m.omegaConstrained.back()=1;assert(!buildWallGeometry(m,0,d,s,o));assert(d.matchingCells==keep&&d.matchingDistance==distance);
m=cubes();m.neighbour[1]=-1;assert(!buildWallGeometry(m,0,d,s,o));assert(d.matchingCells==keep);
''')


def test_geometry_controls_only_affect_consumed_construction(tmp_path):
    compile_run(tmp_path, r'''
auto m=cubes();WallGeometryDescriptor<double>d;WallStatus s;WallGeometryOptions<double>o;
o.constructQuadrature=false;o.relativeTolerance=-1;o.maxRayCells=-99;o.quadratureRefinement=-99;o.profileNodes=-99;o.profileStretch=-99;o.firstSegmentGaussOrder=-99;
assert(buildWallGeometry(m,0,d,s,o));assert(d.distance.empty()&&d.matchingCells[0]==1);
o.constructQuadrature=true;o.profileNodes=3;o.profileStretch=.75;o.firstSegmentGaussOrder=3;
assert(buildWallGeometry(m,0,d,s,o));checkMoments(d,{.5,.5,.5},1);
o.profileNodes=132;o.firstSegmentGaussOrder=6;assert(buildWallGeometry(m,0,d,s,o));checkMoments(d,{.5,.5,.5},1);
std::vector<double>x,w;assert(wallGeometryDetail::gaussRule(1,x,w));assert(x.size()==1&&x[0]==0&&near(w[0],2));
assert(!wallGeometryDetail::gaussRule(0,x,w));
''')


def test_cache_duplicates_and_zero_walls_are_real_publications(tmp_path):
    compile_run(tmp_path, r'''
WallGeometryCache<double>cache;WallStatus s;auto m=cubes();
assert(cache.rebuild(m,{0,0},s));assert(cache.descriptors.size()==2);
for(const auto&d:cache.descriptors)checkMoments(d,{.5,.5,.5},1);
WallGeometryMesh<double>empty;assert(cache.rebuild(empty,{},s));assert(cache.descriptors.empty());
''')


def test_unused_geometry_tail_and_unused_cache_controls_do_not_add_work(tmp_path):
    compile_run(tmp_path, r'''
auto m=cubes();m.points.push_back({std::numeric_limits<double>::quiet_NaN(),0,0});
WallGeometryDescriptor<double>d;WallStatus s;assert(buildWallGeometry(m,0,d,s));checkMoments(d,{.5,.5,.5},1);
WallGeometryCache<double>cache;WallGeometryOptions<double>o;o.constructQuadrature=false;
assert(cache.rebuild(m,{0},s,o));auto* matching=cache.descriptors[0].matchingCells.data();
o.relativeTolerance=-10;o.maxRayCells=-10;o.profileNodes=-10;o.profileStretch=-10;o.quadratureRefinement=-10;o.firstSegmentGaussOrder=-10;
assert(cache.rebuild(m,{0},s,o));assert(cache.descriptors[0].matchingCells.data()==matching);
''')


def test_exhausted_local_iterations_use_evaluated_candidates(tmp_path):



    import subprocess
    from test_wall_geometry import BASE, ROOT
    source=(ROOT/'common/gasWall/WallGeometryBuilder.H').read_text()
    source=source.replace('iteration<32','iteration<1').replace('pass<256','pass<0')
    header=tmp_path/'iteration_budget.H'
    header.write_text(source.replace('"WallModelTypes.H"','"common/gasWall/WallModelTypes.H"'))
    code=BASE.replace('"common/gasWall/WallGeometryBuilder.H"',f'"{header}"')+r'''
int main(){
 std::vector<double>x,w;assert(wallGeometryDetail::gaussRule(4,x,w));
 for(int i=0;i<2;++i){
  const double seed=std::cos(std::acos(-1.)*(i+.75)/4.5);
  auto p=[](double z){return (35*z*z*z*z-30*z*z+3)/8;};
  auto derivative=[](double z){return (140*z*z*z-60*z)/8;};
  const double root=seed-p(seed)/derivative(seed);
  const double weight=2/((1-root*root)*derivative(root)*derivative(root));
  assert(near(x[i],-root,1e-14)&&near(x[3-i],root,1e-14));
  assert(near(w[i],weight,1e-14)&&near(w[3-i],weight,1e-14));
 }
 auto m=cubes();WallGeometryDescriptor<double>d;WallStatus s;
 assert(buildWallGeometry(m,0,d,s));checkMoments(d,{.5,.5,.5},1);
 // Exhausted seed search still publishes its actual signed tetrahedral moments.
 m.cellCentres[0]={3,3,.5};assert(buildWallGeometry(m,0,d,s));
 assert(d.matchingCells==std::vector<int>{1}&&std::isfinite(d.firstMoment));
}
'''
    cpp=tmp_path/'iterations.cpp';cpp.write_text(code);exe=tmp_path/'iterations'
    built=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    ran=subprocess.run([str(exe)],capture_output=True,text=True)
    assert ran.returncode==0,ran.stdout+ran.stderr


def test_mesh_reads_actual_offset_ranges_and_owner_volume(tmp_path):
    compile_run(tmp_path, r'''
auto m=cubes();
// Consumed face ranges need not start at zero. Unconsumed prefix/tail values
// and donor-cell volume storage are not required by the wall construction.
m.facePoints.insert(m.facePoints.begin(),999);for(auto&i:m.faceOffsets)++i;
m.faceOffsets.push_back(999999);m.volumes.resize(1);
WallStatus s;WallGeometryDescriptor<double>d;
assert(buildWallGeometry(m,0,d,s));checkMoments(d,{.5,.5,.5},1);
m.volumes.clear();assert(!buildWallGeometry(m,0,d,s));
''')


def test_consumed_offset_bounds_precede_subtraction_under_ubsan(tmp_path):
    import subprocess
    from test_wall_geometry import BASE, ROOT
    source=tmp_path/'offsets.cpp';binary=tmp_path/'offsets'
    source.write_text(BASE+r'''
int main(){auto original=cubes();WallStatus s;WallGeometryDescriptor<double>d;
assert(buildWallGeometry(original,0,d,s));const auto before=d;
for(int position:{int(original.owner.size()),1}){
 for(int end:{std::numeric_limits<int>::min(),-1,original.faceOffsets[position-1]-1,
              original.faceOffsets[position-1]+2,std::numeric_limits<int>::max()}){
  auto m=original;m.faceOffsets[position]=end;
  assert(!buildWallGeometry(m,0,d,s));assert(s.code==WallCode::InvalidGeometry);
  assert(d.distance==before.distance&&d.volumeWeight==before.volumeWeight);
  assert(d.matchingCells==before.matchingCells&&d.matchingWeights==before.matchingWeights);
  assert(d.matchingDistance==before.matchingDistance&&d.volume==before.volume);
 }
}
auto m=original;m.facePoints.insert(m.facePoints.begin(),999);for(auto&offset:m.faceOffsets)++offset;
m.faceOffsets.push_back(std::numeric_limits<int>::min());
assert(buildWallGeometry(m,0,d,s));checkMoments(d,{.5,.5,.5},1);
}
''')
    command=['g++','-std=c++17','-O1','-fsanitize=undefined','-fno-sanitize-recover=undefined',
             '-I'+str(ROOT),'-I'+str(ROOT/'common'),str(source),'-o',str(binary)]
    built=subprocess.run(command,capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert 'runtime error' not in run.stderr


def test_nonfinite_stage_volume_does_not_publish_invalid_geometry(tmp_path):
    compile_run(tmp_path, r'''
for(bool quadrature:{false,true})for(double volume:{std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}){
 auto m=cubes();m.volumes[0]=volume;
 WallGeometryOptions<double> o;o.constructQuadrature=quadrature;WallGeometryDescriptor<double>d;WallStatus s;
 d.firstMoment=917;d.matchingDistance=918;d.matchingCells={-99};d.distance={-98};
 assert(!buildWallGeometry(m,0,d,s,o));assert(s.code==WallCode::InvalidGeometry);
 assert(d.firstMoment==917&&d.matchingDistance==918);
 assert(d.matchingCells==std::vector<int>{-99}&&d.distance==std::vector<double>{-98});
}
''')


def test_profile_control_diagnostic_completes_existing_quadrature_arithmetic(tmp_path):
    compile_run(tmp_path,r'''
auto m=cubes();WallGeometryOptions<double> o;o.profileNodes=3;o.profileStretch=std::numeric_limits<double>::quiet_NaN();
WallGeometryDescriptor<double>d;WallStatus status;d.firstMoment=917;
assert(buildWallGeometry(m,0,d,status,o));
assert(d.matchingCells==std::vector<int>{1}&&d.matchingDistance==1.5);
assert(!d.distance.empty()&&d.distance.size()==d.volumeWeight.size());
checkMoments(d,{.5,.5,.5},1);
''')
