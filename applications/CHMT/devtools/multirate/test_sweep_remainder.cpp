#define main chmtExistingSweepFixtureMain
#include "test_sweep_constraints.cpp"
#undef main
int main(){
    HostState base=coupledPair();
    for(auto& p:base.solidMesh.points)p=p*.001;
    for(auto& p:base.gasMesh.points)p=p*.001;
    std::string error;require(rebuildGeometry(base.solidMesh,error),error);require(rebuildGeometry(base.gasMesh,error),error);
    base.solid.resize(2);for(int c=0;c<2;++c)base.solid[c].condensed[0]=1000*base.solidMesh.volumes[c];
    const auto inventories=base.solid;
    long double physicalTotal[2]={},actualTotal[2]={};bool nonzero=false;
    for(int step=0;step<240;++step){
        // Signed physical transfers alternate removal/deposition without ever
        // replacing them by a geometry-derived mass or changing the inventory.
        const Real sign=(step/40)%2?-1:1;
        const std::vector<Real> physical={-sign*.043e-8*1e-9,-sign*.043e-8*1e-9};
        const auto original=physical;auto aux=base.filmAux;
        // Model the measured one-sided waveform/adoption residual: each guess
        // over-recedes by an individually admissible four-epsilon volume.
        for(int f=0;f<2;++f)aux[f].solidFront+=(physical[f]-4*std::numeric_limits<Real>::epsilon()*base.solidMesh.volumes[f])
            /mag(base.solidMesh.areaVectors[base.surface.solidFace[f]]);
        HostMesh gas,solid,measured;SurfaceMesh surface;SweepConstraintReport report;std::vector<Real> sweeps;
        const auto acceptedRemainder=base.solidSweepRemainder;
        require(moveCoupledMeshesConstrained(base,aux,physical,1,gas,solid,surface,report,error),"carried actual motion: "+error);
        require(base.solidSweepRemainder==acceptedRemainder&&physical==original,"predictor cannot publish carry or alter physical targets");
        require(makeStageGeometry(base.solidMesh,solid.points,1,measured,sweeps,error),error);
        std::vector<Real> next;
        const bool updated=updateMaterialSweepRemainder(base,physical,sweeps,solid,next,error);
        require(updated,"bounded cumulative geometric remainder at step "+std::to_string(step)+": "+error);
        for(int f=0;f<2;++f){physicalTotal[f]+=physical[f];actualTotal[f]+=sweeps[base.surface.solidFace[f]];
            const long double discrepancy=physicalTotal[f]-actualTotal[f];
            require(std::abs(discrepancy-next[f])<1e-29L,"carry equals cumulative physical target minus actual sweep");
            require(std::abs(discrepancy)<=32*std::numeric_limits<Real>::epsilon()*solid.volumes[f],"cumulative remainder cannot grow with window count");
            nonzero=nonzero||next[f]!=0;
            require(base.solid[f].condensed[0]==inventories[f].condensed[0],"numerical carry cannot modify physical mass");
        }
        base.gasMesh=std::move(gas);base.solidMesh=std::move(solid);base.surface=std::move(surface);base.filmAux=std::move(aux);base.solidSweepRemainder=std::move(next);
    }
    require(nonzero,"real finite-precision geometry exercised a nonzero carried remainder");
    const auto accepted=base;HostMesh gas=base.gasMesh,solid=base.solidMesh;SurfaceMesh surface=base.surface;SweepConstraintReport report;
    require(!moveCoupledMeshesConstrained(base,base.filmAux,{-3,-3},1,gas,solid,surface,report,error),"impossible carried trajectory rejects");
    require(base.solidSweepRemainder==accepted.solidSweepRemainder,"rejected trajectory preserves accepted carry");
    std::vector<Real> sentinel={123,456};
    require(!updateMaterialSweepRemainder(base,{0,0},std::vector<Real>(base.solidMesh.owner.size(),1),base.solidMesh,sentinel,error),"large unaccounted sweep rejected");
    require(sentinel==std::vector<Real>({123,456}),"failed carry update preserves output");
    for(int invalid=0;invalid<3;++invalid){auto bad=base;
        if(invalid==0)bad.solidSweepRemainder.pop_back();
        else bad.solidSweepRemainder[0]=invalid==1?undefinedValue():1;
        require(!validateMaterialSweepRemainder(bad,error),"invalid inherited geometric remainder rejected");
    }
    {
        auto start=coupledPair();auto middle=start.solidMesh.points;
        for(auto& p:middle)p.z+=.1*p.x;
        auto finish=middle;for(auto& p:finish)p.x+=.05;
        HostMesh mid,end,chord;std::vector<Real> first,second,direct;
        require(makeStageGeometry(start.solidMesh,middle,1,mid,first,error),error);
        require(makeStageGeometry(mid,finish,1,end,second,error),error);
        require(makeStageGeometry(start.solidMesh,finish,2,chord,direct,error),error);
        auto measured=first;for(std::size_t f=0;f<measured.size();++f)measured[f]+=second[f];
        std::vector<Real> physical;for(int f:start.surface.solidFace)physical.push_back(measured[f]);
        std::vector<Real> remainder;
        require(updateMaterialSweepRemainder(start,physical,measured,end,remainder,error),"actual segmented path remainder");
        require(remainder==std::vector<Real>({0,0}),"segment sums preserve exact supplied physical volumes");
        require(!updateMaterialSweepRemainder(start,physical,direct,chord,remainder,error),"macro chord cannot replace path-dependent actual segment sweeps");
    }
    std::cout<<"PASS: 240-window physical-target/sweep remainder, signed transfers and rollback\n";
}
