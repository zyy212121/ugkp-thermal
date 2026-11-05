from pathlib import Path
import subprocess
R=Path(__file__).resolve().parents[2]
def test_total_condition_equations_and_programs(tmp_path):
    source=tmp_path/'probe.cpp'
    source.write_text(r"""
#include "gasBoundary/Evaluate.H"
#include "gasBoundary/Wire.H"
#include <cassert>
#include <cstdio>
using namespace ugkp::boundary;
void near(double a,double b){assert(std::abs(a-b)<2e-12*std::max(1.,std::abs(b)));}
int main(){
 int count=0;double maximum=0;
 for(double R:{188.9,237.2,287.05,300.})for(double gamma:{1.1,1.3,1.4,1.67})
 for(double T0:{200.,700.,2000.})for(double speed:{-200.,-10.,0.,50.,250.})
 for(int pressureMode:{0,1})for(double pressureGamma:{1.,1.4}){
  Face f;f.uMode=4;f.pMode=f.tMode=2;f.temperatureGamma=gamma;
  f.compressiblePressure=pressureMode;f.pressureGamma=pressureGamma;
  const double nx=.6,ny=.8,nz=0;
  double value[]={1.,3.,-2.,7.,184325.,T0};
  Primitive owner{1,speed*nx+8*ny,speed*ny-8*nx,10,90000,300},out;
  assert(totalState(f,value,owner,R,nx,ny,nz,0,out));
  near(out.ux*nx+out.uy*ny,speed);
  const bool inflow=speed<0;
  if(inflow){near(out.uz,7);near(out.ux*ny-out.uy*nx,3*ny+2*nx);}
  else {near(out.ux,owner.ux);near(out.uy,owner.uy);near(out.uz,owner.uz);}
  double K=inflow?.5*(out.ux*out.ux+out.uy*out.uy+out.uz*out.uz):0;
  // Independently evaluate the original boundary equations at the returned state.
  double expectedT=T0/(1+K/(R*out.T)*(gamma-1)/gamma);
  near(out.T,expectedT);near(out.rho*R*out.T,out.p);
  const double Kp=K-(inflow?.5*(value[Ux]*value[Ux]+value[Uy]*value[Uy]+value[Uz]*value[Uz]):0);
  double expectedP=value[Pressure]-out.rho*Kp;
  if(pressureMode) {
   if(pressureGamma>1){double g=(pressureGamma-1)/pressureGamma;expectedP=value[Pressure]/std::pow(1+Kp/(R*out.T)*g,1/g);}
   else expectedP=value[Pressure]/(1+Kp/(R*out.T));
  }
  near(out.p,expectedP);maximum=std::max(maximum,std::abs(out.T-expectedT));++count;
 }
 Definition d;d.faces.resize(2);d.functions={{FunctionKind::Table,Bounds::Clamp,0,3,0},{FunctionKind::Polynomial,Bounds::Clamp,3,3,0}};
 d.terms={{0,2},{1,4},{3,10},{2,0},{3,1},{-1,2}};d.faces[1].function[Pressure]=0;validate(d);
 View v{d.faces.data(),d.functions.data(),d.terms.data(),2};
 for(double t:{-1.,0.,.2,1.,2.,3.,4.}) {
  double table=t<=0?2:t>=3?10:t<=1?2+2*t:4+3*(t-1);
  near(evaluate(v,0,t,0),table);near(evaluate(v,1,t,0),2+3*t-t*t);
 }
 std::vector<int> iv;std::vector<double> rv;encode(d,iv,rv);auto restored=decode(iv,rv);
 assert(restored.faces[1].function[Pressure]==0);assert(restored.terms[4].x==3);
 Face invalid;invalid.uMode=4;invalid.tMode=2;invalid.temperatureGamma=1.4;
 double values[]={1,0,0,0,1e5,100};Primitive out;
 assert(!totalState(invalid,values,{1,-10000,0,0,1e5,300},287,1,0,0,0,out));
 printf("equation_cases=%d maximum_temperature_equation_residual=%.17g\n",count,maximum);
}
""")
    exe=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-I'+str(R/'common'),str(source),'-o',str(exe)],check=True)
    subprocess.run([exe],check=True)
