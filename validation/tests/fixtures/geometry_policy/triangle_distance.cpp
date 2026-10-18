#include "mesh/Geometry.C"
#include <iostream>
#include <cstdlib>
int main() {
    using namespace chmt;
    const auto check=[](bool ok,const char* message){if(!ok){std::cerr<<"FAIL: "<<message<<'\n';std::exit(1);}};
    const Real distance=triangleDistance({2e99,2e99,1e99},{0,0,0},{1e100,0,0},{0,1e100,0});
    check(chmt::finite(distance) && absValue(distance/1e99-1)<1e-14,"finite triangle distance must not fall back to vertex distance after overflow");
    check(absValue(triangleDistance({5e99,.25,.1},{0,0,0},{1e100,0,0},{1e100,1,0})-.1)<1e-15,"anisotropic triangle retains its actual interior plane distance");
    check(triangleDistance({.5,1,0},{0,0,0},{1,0,0},{2,0,0})==1,"collinear triangle uses closest segment interior");
    check(triangleDistance({.5,1,0},{0,0,0},{0,0,0},{2,0,0})==1,"repeated triangle endpoint uses actual segment");
    check(triangleDistance({3,4,0},{0,0,0},{0,0,0},{0,0,0})==5,"point triangle exact distance");
    check(!chmt::finite(triangleDistance({undefinedValue(),0,0},{0,0,0},{1,0,0},{0,1,0})),"nonfinite input never substitutes a finite vertex distance");
    std::cout<<"PASS triangle distance: finite overflow, degenerate segments, nonfinite propagation\n";
}
