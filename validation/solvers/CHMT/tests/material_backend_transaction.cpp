#include "gpu/MaterialBackend.H"
#include "materials/CpuMaterialDriver.H"
#include "materials/DryMaterialAdapter.H"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
using namespace chmt;
void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
void equal(const std::vector<SolidQ>& a,const std::vector<SolidQ>& b){
    require(a.size()==b.size(),"inventory shape");
    for(std::size_t i=0;i<a.size();++i){
        for(int k=0;k<Nc;++k)require(a[i].condensed[k]==b[i].condensed[k],"condensed inventory");
        for(int k=0;k<Ns;++k)require(a[i].pore[k]==b[i].pore[k],"pore inventory");
        for(int k=0;k<Nr;++k)require(a[i].progress[k]==b[i].progress[k],"reaction progress");
        require(a[i].energy==b[i].energy&&a[i].porosity==b[i].porosity,"energy/porosity");
    }
}
HostState fixture(PhysicsConfig& p){
    p.permeability=0;p.enableReactions=false;p.gasMode=ugkwp::GasMode::MixtureFrozen;p.gasConductivity=2;p.gasViscosity=.01;
    for(int k=0;k<Nc;++k){auto& t=p.condensed[k];t.rho=1200+200*k;t.cp0=900+100*k;t.cp1=0;t.conductivity=1;t.Tmin=100;t.Tmax=2000;}
    for(int k=0;k<Ns;++k){auto& t=p.species[k];t.R=280+60*k;t.cp0=1050+100*k;t.cp1=0;t.Tmin=100;t.Tmax=2000;p.poreEffectiveDiffusivity[k]=0;}
    HostState h;auto& m=h.solidMesh;m.cellFaceOffsets.push_back(0);
    for(int c=0;c<2;++c){
        const Vec3 centre{2.*c,0,0};m.volumes.push_back(1);m.cellCentres.push_back(centre);
        const Vec3 normals[]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
        for(auto n:normals){const int f=m.owner.size();m.owner.push_back(c);m.neighbour.push_back(-1);
            m.faceCentres.push_back(centre+n*.5);m.areaVectors.push_back(n);m.boundaryKind.push_back(BoundaryKind::NoSlip);
            m.periodicPartner.push_back(-1);m.cellFaces.push_back(f);m.cellFaceSigns.push_back(1);}
        m.cellFaceOffsets.push_back(m.cellFaces.size());
        SolidQ q;q.condensed[0]=600;q.condensed[1]=280;q.porosity=.3;
        for(int k=0;k<Nc;++k)q.energy+=q.condensed[k]*condensedE(p.condensed[k],650);
        for(int k=0;k<Ns;++k){q.pore[k]=1.;q.energy+=speciesE(p.species[k],650);}
        h.solid.push_back(q);
    }
    return h;
}
int main(){
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
    require(cudaDeviceSetLimit(cudaLimitStackSize,65536)==cudaSuccess,"CUDA stack");
    PhysicsConfig p;auto h=fixture(p);std::string error;bool numerical=false;
    std::unique_ptr<MaterialBackend,decltype(&destroyMaterialBackend)> b(createMaterialBackend(error),destroyMaterialBackend);
    require(bool(b),error);require(beginMaterialWindow(*b,h,p,error),error);
    Real dtLimit=0;require(evaluateResidentMaterialTransport(*b,h.solidMesh,{},.001,.4,dtLimit,error,numerical),error);
    require(!numerical&&dtLimit>0,"successful transport classification");
    ExchangePacket good;good.kind=ExchangeKind::GasSolid;good.solidCell=0;good.pore[0]=.125;good.energy=64;
    ExchangePacket bad=good;bad.solidCell=1;bad.pore[0]=2.;
    CpuMaterialControls controls;
    std::vector<SolidQ> shadow(1),donor(1);shadow[0].energy=123;donor[0].energy=456;
    const auto savedShadow=shadow,savedDonor=donor;
    require(!advanceResidentMaterialSources(*b,h.solidMesh,h.surface,{good,bad},{},{},.001,controls,true,shadow,donor,error,numerical),"overdraw must fail");
    require(error=="material packet batch exceeds physical donor inventory"&&numerical,"original rejection");
    equal(shadow,savedShadow);equal(donor,savedDonor);
    std::vector<SolidQ> expected;require(applyMaterialPacketBatch(h.solid,{good},expected,error,&numerical),error);
    for(int attempt=0;attempt<2;++attempt){
        require(advanceResidentMaterialSources(*b,h.solidMesh,h.surface,{good},{},{},.001,controls,true,shadow,donor,error,numerical),error);
        equal(shadow,expected);std::vector<SolidQ> zero(2);
        for(int c=0;c<2;++c){zero[c]=addSolid(expected[c],h.solid[c],-1);if(!c)zero[c]=addSolid(zero[c],packetDelta(good).solid,-1);}
        equal(donor,zero); // Neither rejected nor uncommitted calls accumulate transfers.
    }
    commitResidentMaterialStep(*b,.001);
    std::vector<SolidQ> expectedSecond;require(applyMaterialPacketBatch(expected,{good},expectedSecond,error,&numerical),error);
    require(advanceResidentMaterialSources(*b,h.solidMesh,h.surface,{good},{},{},.001,controls,true,shadow,donor,error,numerical),error);
    equal(shadow,expectedSecond);
    std::vector<SolidQ> expectedDonor(2);
    for(int c=0;c<2;++c){expectedDonor[c]=addSolid(expectedSecond[c],h.solid[c],-1);
        if(!c)expectedDonor[c]=addSolid(expectedDonor[c],addSolid(packetDelta(good).solid,packetDelta(good).solid),-1);}
    equal(donor,expectedDonor);
    h.gasMesh.owner={0};h.gasMesh.volumes={1};h.gasMesh.areaVectors={{-1,0,0}};h.gasMesh.faceIds={1};
    h.surface.area={1,1};h.surface.gasFace={-1,0};h.surface.normal={{1,0,0},{1,0,0}};
    h.surface.gasDistance={.5,.5};h.surface.solidCell={-1,0};h.surface.solidFace={-1,0};h.surface.solidDistance={.5,.5};
    std::vector<std::vector<GasPrimitive>> bulk(9,std::vector<GasPrimitive>(2));
    std::vector<GasGradient> gradients;std::vector<MaterialSurfaceRequest> requests;
    for(std::size_t i=0;i<bulk.size();++i){auto& gas=bulk[i][1];gas.rho=1;gas.temperature=710+7*i;gas.Y[0]=.7;gas.Y[1]=.3;
        gas.pressure=(.7*p.species[0].R+.3*p.species[1].R)*gas.temperature;
        requests.push_back({&bulk[i],&gradients,nullptr});}
    std::vector<CpuSurfaceResult> batched;
    require(evaluateResidentMaterialSurfaces(*b,h,requests,.001,true,false,batched,error,numerical),error);
    require(batched.size()==bulk.size(),"history result shape");
    for(std::size_t i=0;i<bulk.size();++i){CpuSurfaceResult single;
        require(evaluateResidentMaterialSurface(*b,h,bulk[i],gradients,.001,true,false,single,error,numerical),error);
        for(std::size_t f=0;f<2;++f){const auto& a=batched[i].physics[f];const auto& v=single.physics[f];
            require(a.topTemperature==v.topTemperature&&a.gasEnergy==v.gasEnergy&&a.poreEnergy==v.poreEnergy
                &&a.gasConductive==v.gasConductive&&a.thermalResidual==v.thermalResidual,"history face indexing or shared-state mutation");
            for(int k=0;k<Ns;++k)require(a.gasSpecies[k]==v.gasSpecies[k]&&a.poreSpecies[k]==v.poreSpecies[k],"history species");
        }
    }
    std::puts("PASS: independent history batching equals sequential surface evaluation, including an inactive first face");
    std::puts("PASS: failed and uncommitted steps preserve current inventory, donor sums and caller outputs; commit advances once");
}
