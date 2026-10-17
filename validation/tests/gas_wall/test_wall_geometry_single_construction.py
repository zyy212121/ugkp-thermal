"""A wall ray starts at its deterministic interior triangle, without retrying origins."""
from test_wall_geometry import compile_run


def test_concave_origin_rule_is_selected_before_tracing_even_when_centroid_is_inside(tmp_path):
    compile_run(tmp_path,r'''
 MeshBuilder builder;
 const std::vector<V> base={{0,0,0},{2,0,0},{2,1,0},{1.5,1,0},{1.5,2,0},{0,2,0}};
 for(int layer=0;layer<3;++layer){
  std::vector<V> low,high;for(auto p:base){p[2]=layer;low.push_back(p);p[2]=layer+1;high.push_back(p);}
  std::vector<std::vector<V>> faces={low,high};
  for(size_t j=0;j<base.size();++j)faces.push_back({low[j],low[(j+1)%base.size()],high[(j+1)%base.size()],high[j]});
  builder.cell(faces,{.5,.5,layer+.5},3.5);builder.m.cellCentres.back()={3.125/3.5,3.25/3.5,layer+.5};
 }
 auto m=builder.m;m.physicalWall[0]=1;WallGeometryDescriptor<double> d;WallStatus s;
 wallGeometryDetail::Context<double> ctx;WallGeometryOptions<double> options;
 assert(wallGeometryDetail::prepare(m,ctx,s,options));
 const auto& face=ctx.faces[0];int largest=0;
 for(int i=1;i<int(face.triangles.size());++i){
  const auto a=face.triangles[i],b=face.triangles[largest];
  if(wallGeometryDetail::norm(cross(sub(m.points[a.b],m.points[a.a]),sub(m.points[a.c],m.points[a.a])))>
     wallGeometryDetail::norm(cross(sub(m.points[b.b],m.points[b.a]),sub(m.points[b.c],m.points[b.a]))))largest=i;
 }
 const auto t=face.triangles[largest];const V expected=mul(add(add(m.points[t.a],m.points[t.b]),m.points[t.c]),1./3);
 assert(buildWallGeometry(m,0,d,s));
 for(int i=0;i<3;++i)assert(near(d.origin[i],expected[i]));
 checkMoments(d,m.cellCentres[0],m.volumes[0]);
 ''')


def test_finite_projection_budget_reports_residual_without_restarting(tmp_path):
    compile_run(tmp_path,r'''
 auto m=cubes();wallGeometryDetail::Context<double> ctx;WallStatus s;WallGeometryOptions<double> options;
 assert(wallGeometryDetail::prepare(m,ctx,s,options));
 // Opposing supports expose the terminal residual of the single projection solve.
 ctx.faces[0].normal={1,0,0};ctx.faces[0].centre={0,0,0};
 ctx.faces[1].normal={-1,0,0};ctx.faces[1].centre={0,0,0};
 ugkp::diagnostics::DeviceContext diagnostics{};V seed;double scale=0;
 assert(wallGeometryDetail::cellSeed(m,ctx,0,seed,scale,&diagnostics));
 assert(!diagnostics.fatalClaim&&diagnostics.warnings[1].count==1);
 assert(diagnostics.warnings[1].sample.sourceSiteId==2830);
 assert(diagnostics.warnings[1].sample.observed>0&&std::isfinite(seed[0]));
 ''')
