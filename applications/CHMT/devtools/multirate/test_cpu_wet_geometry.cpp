#define main chmtExistingSweepFixtureMain
#include "test_sweep_constraints.cpp"
#undef main
#include "materials/MaterialTransport.H"
int main(){
    HostState base=coupledPair();for(auto& point:base.gasMesh.points)point.z+=.1;std::string error;
    require(rebuildGeometry(base.gasMesh,error),error);base.film.resize(2);
    for(int f=0;f<2;++f){base.film[f].mass=100;base.filmAux[f].thickness=.1;base.filmAux[f].area=1;}
    auto candidate=base.filmAux;for(int f=0;f<2;++f){require(filmThicknessFromMass(120,1,1000,candidate[f].thickness),"updated melting film thickness");candidate[f].solidFront=-.02;}
    HostMesh gas,solid;SurfaceMesh surface;SweepConstraintReport report;
    require(moveCoupledMeshesConstrained(base,candidate,{-.02,-.02},.2,gas,solid,surface,report,error),"actual wet melting motion: "+error);
    HostMesh measured;std::vector<Real> gs,ss;require(makeStageGeometry(base.gasMesh,gas.points,.2,measured,gs,error),error);require(makeStageGeometry(base.solidMesh,solid.points,.2,measured,ss,error),error);
    for(int f=0;f<2;++f)require(filmSweepsCompatible(100,120,1000,gs[base.surface.gasFace[f]],ss[base.surface.solidFace[f]],0,Tolerances{}),"real 3-D melting top/base volume certificate");
    HostState wet=base;wet.gasMesh=gas;wet.solidMesh=solid;wet.surface=surface;wet.filmAux=candidate;for(auto& q:wet.film)q.mass=120;
    auto evaporation=wet.filmAux;for(auto& a:evaporation)require(filmThicknessFromMass(90,1,1000,a.thickness),"updated evaporating thickness");
    require(moveCoupledMeshesConstrained(wet,evaporation,{0,0},.07,gas,solid,surface,report,error),"actual wet evaporation motion: "+error);
    require(makeStageGeometry(wet.gasMesh,gas.points,.07,measured,gs,error),error);require(makeStageGeometry(wet.solidMesh,solid.points,.07,measured,ss,error),error);
    for(int f=0;f<2;++f){require(filmSweepsCompatible(120,90,1000,gs[wet.surface.gasFace[f]],ss[wet.surface.solidFace[f]],0,Tolerances{}),"real 3-D evaporation top/base volume certificate");
        require(!filmSweepsCompatible(120,90,1000,0,0,0,Tolerances{}),"stale old thickness/zero gas sweep cannot conserve updated film mass");}
    std::cout<<"CPU wet geometry: actual 3-D melting/evaporation volume certificates passed\n";
}
