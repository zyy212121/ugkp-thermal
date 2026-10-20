
#include "validation/tests/gas_wall/fixtures/sst_small_k_column_capture.H"
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cstdio>
int main(int argc,char**argv){using namespace ugkwp::gaswall;using namespace ugkwp::gaswall::detail;
 CapturedSingularWall m;auto&in=m.input;auto&c=m.config;c.nodes=std::atoi(argv[2]);
 WallWorkspace<double,2,128>w;LayerContext<double,2>ctx;makeContext(in,c,ctx);
 std::ifstream file(argv[1]);std::string line;
 while(std::getline(file,line)){if(line.rfind("node,",0)!=0)continue;
  std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);std::string tag;int count,i;
  row>>tag>>count>>i>>w.y[i];for(int j=0;j<6;++j)row>>w.state[i][j];
 }
 double norm;allResidual(in,c,ctx,w.state,w.y,w.residual,norm);
 static double jac[128][3][6][6]{},base[128][6];
 for(int i=0;i<c.nodes;++i)for(int k=0;k<6;++k){base[i][k]=w.residual[i][k];w.delta[i][k]=-base[i][k];}
 for(int i=0;i<c.nodes;++i)assembleJacobianNode(in,c,ctx,w,i);
 for(int i=0;i<c.nodes;++i)for(int row=0;row<6;++row)for(int col=0;col<6;++col){jac[i][0][row][col]=w.lower[i][row][col];jac[i][1][row][col]=w.diagonal[i][row][col];jac[i][2][row][col]=w.upper[i][row][col];}
 bool ok=solveLayerNewtonSystem(c,w);std::printf("solved %d norm %.17g\n",int(ok),norm);
 double err=0;for(int i=0;i<c.nodes;++i)for(int row=0;row<6;++row){double v=base[i][row];
  for(int k=-1;k<=1;++k)if(i+k>=0&&i+k<c.nodes)for(int col=0;col<6;++col)v+=jac[i][k+1][row][col]*w.delta[i+k][col];err=std::max(err,std::abs(v));}
 std::printf("linear_system_residual %.17g\n",err);
 for(double alpha:{1.,.5,.1,.01,.001,.0001,.00001,.000001,.0000001}){
  for(int i=0;i<c.nodes;++i)for(int j=0;j<6;++j)w.trial[i][j]=w.state[i][j]+alpha*w.delta[i][j];
  double n=0;bool v=allResidual(in,c,ctx,w.trial,w.y,w.residual,n);double error=0;int worst=-1,component=-1;
  for(int i=0;i<c.nodes;++i)for(int j=0;j<6;++j){double e=std::abs((w.residual[i][j]-base[i][j])/alpha+base[i][j]);
   if(e>error){error=e;worst=i;component=j;}}
  std::printf("probe %.8g valid %d norm %.17g deriv_error %.17g at %d,%d\n",alpha,int(v),n,error,worst,component);
 }
 int node=c.nodes==48?26:69;std::printf("worst node %d\n",node);
 for(int i=node-1;i<=node+1;++i){
   LayerPoint<double,2>p;evaluatePoint(in,c,ctx,w.state,w.y,i,p);
   std::printf("point %d y %.10g k %.10g omega %.10g nut %.10g f1 %.10g f2 %.10g sourceO %.10g deltaK %.10g deltaZ %.10g\n",i,w.y[i],p.k,p.omega,p.nut,p.f1,p.f2,p.sourceOmega,w.delta[i][4],w.delta[i][5]);
   for(int j=0;j<6;++j){double old=w.state[i][j],scale=std::max(std::abs(old),1e-15);
     for(double relative:{1e-4,1e-5,1e-6,1e-7}){
       double h=relative*scale,plus[6]{},minus[6]{};
       w.state[i][j]=old+h;layerRow(in,c,ctx,w.state,w.y,node,plus);
       w.state[i][j]=old-h;layerRow(in,c,ctx,w.state,w.y,node,minus);w.state[i][j]=old;
       std::printf("derivative j %d col %d rel %.1g prod %.12g central %.12g\n",i,j,relative,jac[node][i-node+1][5][j],(plus[5]-minus[5])/(2*h));
     }
   }
 }
}
