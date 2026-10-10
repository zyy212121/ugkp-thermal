"""Profile-aware real-volume SST integration against independent Gauss32."""
import subprocess
import numpy as np
from test_wall_model import BASE, ROOT
from test_wall_geometry import BASE as GEOMETRY_BASE


def test_actual_cube_source_matches_profile_interval_gauss32(tmp_path):
    geometry = GEOMETRY_BASE[GEOMETRY_BASE.index('using V='):GEOMETRY_BASE.index('WallGeometryMesh<double> tetra()')]
    geometry = '\n'.join(line for line in geometry.splitlines() if not line.startswith('bool near('))
    nodes, weights = np.polynomial.legendre.leggauss(32)
    arrays = 'const double glX[32]={' + ','.join(format(x,'.17g') for x in nodes) + '};\n'
    arrays += 'const double glW[32]={' + ','.join(format(x,'.17g') for x in weights) + '};\n'
    body = r'''
 auto mesh=cubes();
 for(auto& p:mesh.points){p[0]*=.01;p[1]*=.01;p[2]=p[2]<=1?p[2]*.008:.008+(p[2]-1)*.004;}
 for(auto& p:mesh.cellCentres){p[0]*=.01;p[1]*=.01;p[2]=p[2]<=1?p[2]*.008:.008+(p[2]-1)*.004;}
 for(size_t i=0;i<mesh.volumes.size();++i)mesh.volumes[i]*=i==0?8e-7:4e-7;
 Model m;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=30;
 m.in.matching.k=.5;m.in.matching.omega=200;m.in.normal[1]=0;m.in.normal[2]=1;
 WallModelConfig<double> c;c.enableSst=true;c.maxIterations=100;
 static WallWorkspace<double,2,128> work;WallOutput<double,2> result;WallStatus status;
 std::cout.precision(17);
 for(double mass:{0.,.001})for(int n:{48,96,128}){
  m.in.massFlux[0]=.7*mass;m.in.massFlux[1]=.3*mass;
  WallGeometryDescriptor<double> descriptor;WallGeometryOptions<double> options;
  options.profileNodes=n;options.profileStretch=c.stretch;options.firstSegmentGaussOrder=8;
  assert(buildWallGeometry(mesh,0,descriptor,status,options));assert(near(descriptor.matchingDistance,.01));
  m.in.quadrature=descriptor.quadrature();m.in.ownerDistance=descriptor.ownerDistance;
  m.in.matchingDistance=descriptor.matchingDistance;c.nodes=n;
  assert(evaluateWallModel(m.in,c,work,result,status));const double geometryIntegral=result.integratedKSource;
  std::vector<double> distance,weight;
  // Independent one-dimensional quadrature uses the cube's exact constant
  // section area; it never calls tetra section construction.
  for(int i=0;i<n-1&&work.y[i]<.008;++i){
   const double lo=work.y[i],hi=std::min(work.y[i+1],.008);
   for(int q=0;q<32;++q){double y,w;
    if(i==0){const double eta=(glX[q]+1)/2;y=hi*eta*eta;w=hi*eta*glW[q];}
    else {y=(lo+hi)/2+(hi-lo)/2*glX[q];w=(hi-lo)/2*glW[q];}
    distance.push_back(y);weight.push_back(1e-4*w);}}
  m.in.quadrature={distance.data(),weight.data(),int(distance.size()),8e-7,3.2e-9};
  detail::LayerContext<double,2> ctx;detail::makeContext(m.in,c,ctx);
  assert(detail::publishLayer(m.in,c,ctx,work,status.iteration,status.residual,result));
  const double reference=result.integratedKSource;
  std::cout<<"mass="<<mass<<" nodes="<<n<<" profile_geometry="<<geometryIntegral<<" independent_gauss32="<<reference<<" points="<<descriptor.distance.size()<<"\n";
  assert(std::abs(geometryIntegral-reference)<1e-6*std::max(std::abs(reference),1e-12));
  // The reconstructed source is not the exact point constitutive field. Report
  // its remaining finite-grid boundary-flux defect on an absolute-source scale.
  double absoluteIntegral=0;
  for(size_t q=0;q<distance.size();++q){const double y=distance[q];int l=0;
   while(l+1<n-1&&work.y[l+1]<y)++l;
   double source;
   if(l==0){const double one=1;auto sample=m.in;sample.quadrature={&y,&one,1,1,y};WallOutput<double,2> point;
    assert(detail::publishLayer(sample,c,ctx,work,0,0.,point));source=point.integratedKSource;}
   else {const double f=(y-work.y[l])/(work.y[l+1]-work.y[l]);source=(1-f)*work.seed[l][0]+f*work.seed[l+1][0];}
   absoluteIntegral+=weight[q]*std::abs(source);
  }
  const double ymax=.008;int lastNode=0;while(lastNode+1<n-1&&work.y[lastNode+1]<ymax)++lastNode;
  const double length=work.y[lastNode+1]-work.y[lastNode],f=(ymax-work.y[lastNode])/length;double state[6];
  for(int j=0;j<6;++j)state[j]=(1-f)*work.state[lastNode][j]+f*work.state[lastNode+1][j];
  detail::LayerPoint<double,2> point,left,right;
  assert(detail::pointState(m.in,c,ctx,state,point));
  assert(detail::pointState(m.in,c,ctx,work.state[lastNode],left));
  assert(detail::pointState(m.in,c,ctx,work.state[lastNode+1],right));
  detail::pointSst(m.in,c,ymax,left,right,length,point);
  const double diffusion=m.in.model.viscosity+point.rho*sstAlphaK(point.f1,m.in.model.sstCoefficients)*point.nut;
  const double upperFlux=mass*point.k-diffusion*(right.k-left.k)/length;
  const double fluxIntegral=1e-4*(upperFlux-result.wallKFlux),defect=reference-fluxIntegral;
  assert(std::isfinite(defect)&&absoluteIntegral>0);
  std::cout<<"source_abs_integral="<<absoluteIntegral<<" boundary_flux_integral="<<fluxIntegral
           <<" finite_grid_defect="<<defect<<" defect_over_abs_source="<<std::abs(defect)/absoluteIntegral<<"\n";
  // In contrast, the discrete nodal FV source telescopes exactly to its shared
  // face fluxes. Do not conflate that algebraic identity with reconstruction.
  double nodalIntegral=0;
  for(int i=1;i<=lastNode;++i){detail::LayerPoint<double,2> node;
   assert(detail::evaluatePoint(m.in,c,ctx,work.state,work.y,i,node));
   nodalIntegral+=(work.y[i+1]-work.y[i-1])*.5*node.sourceK;}
  detail::LayerFlux<double,2> first,last;
  assert(detail::intervalFlux(m.in,c,ctx,work.state[0],work.state[1],work.y[0],work.y[1],first));
  assert(detail::intervalFlux(m.in,c,ctx,work.state[lastNode],work.state[lastNode+1],work.y[lastNode],work.y[lastNode+1],last));
  const double discreteDefect=nodalIntegral-(last.k-first.k);
  assert(std::abs(discreteDefect)<1e-8);
  std::cout<<"discrete_source_flux_defect="<<discreteDefect<<"\n";

 }
'''
    source = tmp_path/'source_geometry.cpp'
    source.write_text(BASE+'\n#include "common/gasWall/WallGeometryBuilder.H"\n#include <map>\n'+geometry+'\nint main(){\n'+arrays+body+'\n}\n')
    exe = source.with_suffix('')
    result = subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT),'-I',str(ROOT/'common'),str(source),'-o',str(exe)], capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    print(result.stdout)
