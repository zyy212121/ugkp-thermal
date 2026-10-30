#include <initializer_list>
#include "gasWall/ConstantTransportWall.H"
#include <cstdio>
int main(){for(double a:{-1000.,-200.,-100.,-50.,-10.,-1.,-.001,0.,.001,1.,10.,50.,100.,200.,1000.})for(double pr:{.00000001,.72,2.,10.}){double h=0;bool ok=ugkwp::gaswall::detail::constantWallHeatingFactor(a,a*pr,h);printf("%.17g %.17g %d %.17g\n",a,a*pr,int(ok),h);}}
