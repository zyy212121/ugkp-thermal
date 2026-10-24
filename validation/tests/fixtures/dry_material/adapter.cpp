#include "materials/DryMaterialAdapter.H"
#include "materials/MaterialTransport.H"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace chmt;
void close(double a,double b,double tol=1e-11){assert(std::abs(a-b)<=tol*std::max(1.,std::max(std::abs(a),std::abs(b))));}
PhysicsConfig config(){PhysicsConfig p;p.spatialOrder=1;p.poreViscosity=1e-5;
    for(auto& c:p.condensed){c.rho=1000;c.cp0=1000;c.conductivity=1;c.Tmin=100;c.Tmax=3000;}
    for(auto& s:p.species){s.R=287;s.cp0=1000;s.Tmin=100;s.Tmax=3000;}
    p.species[1].e0=-2e5;return p;}
void energy(SolidQ& q,const PhysicsConfig& p,double T){q.energy=dry::caloricState(dryAdapter::state(q),dryAdapter::condensed(p),dryAdapter::GasCalorics{p},T).energy;}
HostMesh mesh(){HostMesh m;m.volumes={.001,.001};m.cellCentres={{0,0,0},{.1,0,0}};
    m.owner={0};m.neighbour={1};m.areaVectors={{.01,0,0}};m.faceCentres={{.05,0,0}};
    m.periodicPartner={-1};m.boundaryKind={BoundaryKind::Internal};m.cellFaceOffsets={0,1,2};m.cellFaces={0,0};m.cellFaceSigns={1,-1};return m;}
int main(){
    PhysicsConfig p=config();auto m=mesh();std::vector<SolidQ> q(2);
    for(int c=0;c<2;++c){q[c].condensed[0]=.5;q[c].pore[0]=c?.0002:.0008;q[c].pore[1]=.001-q[c].pore[0];energy(q[c],p,600);}
    p.poreEffectiveDiffusivity[0]=1e-4;p.poreEffectiveDiffusivity[1]=2e-4;
    MaterialTransportResult out;std::string error;
    assert(evaluateMaterialTransport(m,q,p,{},.01,.4,out,error));
    // rho=2, gradient=(-6,+6), C=.0012 and upwindY=(.8,.2).
    close(out.faceFlux[0].pore[0],.01*.00216);close(out.faceFlux[0].pore[1],-.01*.00216);
    close(out.faceFlux[0].energy,.01*.00216*(speciesH(p.species[0],600)-speciesH(p.species[1],600)));
    close(out.rates[0].energy+out.rates[1].energy,0);
    for(int s=0;s<Ns;++s)close(out.rates[0].pore[s]+out.rates[1].pore[s],0);
    auto empty=q;
    for(auto& mass:empty[1].pore)mass=0;energy(empty[1],p,600);
    assert(evaluateMaterialTransport(m,empty,p,{},.01,.4,out,error));
    for(int species=0;species<Ns;++species)close(out.faceFlux[0].pore[species],0);
    for(auto& mass:empty[0].pore)mass=0;energy(empty[0],p,600);
    assert(evaluateMaterialTransport(m,empty,p,{},.01,.4,out,error));close(out.faceFlux[0].energy,0);
    auto noGasTables=p;for(auto& species:noGasTables.species)species=SpeciesThermo{};
    assert(evaluateMaterialTransport(m,empty,noGasTables,{},.01,.4,out,error));close(out.faceFlux[0].energy,0);
    // A thermal gradient alone must not enter explicit transport: native
    // OpenFOAM owns implicit conduction. Set pore diffusion/flow to zero.
    for(auto& d:p.poreEffectiveDiffusivity)d=0;energy(q[1],p,800);
    assert(evaluateMaterialTransport(m,q,p,{},.01,.4,out,error));close(out.faceFlux[0].energy,0);
    // Default-zero diffusion retains historical Darcy advection and enthalpy.
    p.permeability=1e-14;MaterialPrimitive a,b;
    assert(recoverMaterial(q[0],.001,p,a)&&recoverMaterial(q[1],.001,p,b));
    SolidQ expected;a.conductivity=b.conductivity=0;
    assert(darcyConductionFlux(q[0],q[1],a,b,.001,.001,.01,.1,{1,0,0},p,expected));
    assert(evaluateMaterialTransport(m,q,p,{},.01,.4,out,error));
    for(int s=0;s<Ns;++s)close(out.faceFlux[0].pore[s],expected.pore[s]);close(out.faceFlux[0].energy,expected.energy);
    p.spatialOrder=2;
    assert(evaluateMaterialTransport(m,q,p,{},.01,.4,out,error));
    const Real velocity=-p.permeability/p.poreViscosity*(b.pressure-a.pressure)/.1;
    for(int species=0;species<Ns;++species)close(out.faceFlux[0].pore[species],.01*velocity);
    close(out.faceFlux[0].energy,.01*velocity*(speciesH(p.species[0],700)+speciesH(p.species[1],700)));
    // Formation-inclusive constant-U chemistry and progress map to actual
    // calibrated CHMT rates, with no unrelated gas time integrator change.
    p.enableReactions=true;p.nReactions=1;auto& r=p.reactions[0];
    r.A=2;r.order[0]=1;r.condensedNu[0]=-1;r.condensedNu[1]=.3;r.gasNu[0]=.2;r.gasNu[1]=.5;
    SolidQ reacted;std::uint64_t steps=0;
    assert(advanceMaterialReactions(q[0],.001,p,.01,.001,.4,100,reacted,steps,error));assert(steps==10);
    close(reacted.condensed[0],q[0].condensed[0]*std::pow(.998,10));
    const auto extent=q[0].condensed[0]-reacted.condensed[0];
    close(reacted.progress[0],extent);close(reacted.condensed[1],.3*extent);
    close(reacted.pore[0]-q[0].pore[0],.2*extent);close(reacted.pore[1]-q[0].pore[1],.5*extent);close(reacted.energy,q[0].energy);
    // Accepted extensive exchange is consumed once, independent of a new
    // constitutive proposal. Sign convention: gas gains, material loses.
    ExchangePacket packet;packet.kind=ExchangeKind::PoreGas;packet.solidCell=0;packet.energy=12;
    packet.pore[0]=packet.species[0]=packet.mass=1e-5;
    std::vector<SolidQ> accepted;
    assert(applyMaterialPacketBatch(q,{packet},accepted,error));
    close(accepted[0].energy,q[0].energy-12);close(accepted[0].pore[0],q[0].pore[0]-1e-5);
    close(accepted[1].energy,q[1].energy);
    std::cout<<"PASS production dry material adapter: diffusion, Darcy, chemistry, accepted extensive exchange\n";
}
