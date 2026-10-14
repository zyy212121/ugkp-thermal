#include "coupling/WindowTime.H"
#include <cstdio>
#include <algorithm>
int main(){
 using namespace chmt;
 for(double origin : {0.,.0005}) {
  double old=origin;WindowTime paid;
  const double end=origin+.0005,cap=1e-5;
  const WindowTime total=windowOffset(origin,end);
  for(int step=1;step<=50;++step){
   const double oldDt=std::min(cap,end-old),newDt=std::min(cap,boundedWindowDuration(total,paid));
   if(step<8 || step>=49)std::printf("origin=%a step=%d old_stamp=%a paid_stamp=%a old_dt=%a paid_dt=%a relative_dt=%.17g\n",origin,step,old,windowAbsolute(origin,paid),oldDt,newDt,(newDt-oldDt)/oldDt);
   old+=oldDt;paid+=newDt;
  }
  std::printf("complete old=%a paid=%La correction=%La residual=%La\n",old,paid.sum,paid.correction,windowDifference(total,paid));
 }
}
