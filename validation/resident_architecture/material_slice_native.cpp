#include "material_slice_probe.H"
#include <iostream>
#include "materials/MaterialInventoryMath.H"
#include "materials/Reaction.H"
#include <stdexcept>
#include <limits>
#include <iomanip>
using namespace chmt;
using Clock=ugkp::CompensatedTime;
WindowTime native(Clock x){WindowTime t(x.sum);t.add(x.correction);return t;}
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
double maxError=0;std::size_t values=0;
void equal(double a,double b){
    if(a==b||(std::isnan(a)&&std::isnan(b)))return;
    const double relative=std::abs(a-b)/std::max(1e-30,std::max(std::abs(a),std::abs(b)));maxError=std::max(maxError,relative);
    if(!(relative<3e-14)){std::cerr<<std::setprecision(17)<<"expected "<<a<<" actual "<<b<<'\n';throw std::runtime_error("physical integral differs");}
}
void packetEqual(const ExchangePacket& a,const ExchangePacket& b){
    require(a.kind==b.kind&&a.step==b.step&&a.geometry==b.geometry&&a.face==b.face&&a.stage==b.stage&&a.gasCell==b.gasCell&&a.solidCell==b.solidCell&&a.filmFace==b.filmFace&&a.consumerMask==b.consumerMask,"packet identity/order differs");
    equal(a.mass,b.mass);equal(a.energy,b.energy);equal(a.momentum.x,b.momentum.x);equal(a.momentum.y,b.momentum.y);equal(a.momentum.z,b.momentum.z);
    equal(a.conductive,b.conductive);equal(a.advective,b.advective);equal(a.pressureWork,b.pressureWork);equal(a.viscousWork,b.viscousWork);equal(a.radiation,b.radiation);equal(a.liquidKineticAdvection,b.liquidKineticAdvection);
    for(int s=0;s<Ns;++s){equal(a.species[s],b.species[s]);equal(a.pore[s],b.pore[s]);equal(a.poreSweep[s],b.poreSweep[s]);}
    for(int c=0;c<Nc;++c)equal(a.condensed[c],b.condensed[c]);values+=11+3*Ns+Nc;
}
// Deliberately reproduce the existing host publication, not the new virtual
// packet accessor. The separately compiled native IntervalCursor is the oracle.
GasIntervalRecord record(const materialSliceProbe::Test& t,std::size_t frame){
    GasIntervalRecord out;const auto time=t.intervals[frame];out.localBegin=native(time.begin);out.localEnd=native(time.end);out.actualDt=time.actualDt;
    for(std::size_t f=0;f<t.faces;++f){const auto& r=t.surface[frame*t.faces+f];out.packets.push_back(r.primary);
        if(hasPacketTransfer(r.pore))out.packets.push_back(r.pore);
        if(t.particles){ExchangePacket p{};p.kind=ExchangeKind::ParticleWall;p.step=r.primary.step;p.geometry=r.primary.geometry;p.face=r.primary.face;
            p.stage=2;p.gasCell=r.primary.gasCell;p.solidCell=r.primary.solidCell;p.filmFace=r.primary.filmFace;p.consumerMask=ConsumeParticle;
            p.energy=p.conductive=r.particleWallEnergy;if(hasPacketTransfer(p))out.packets.push_back(p);}
        out.radiationEnergy.push_back(r.radiationEnergy);
    }return out;
}
ExchangePacket packet(std::size_t r,std::size_t f,int kind,double scale){
    ExchangePacket p;p.kind=kind?ExchangeKind::PoreGas:ExchangeKind::GasSolid;p.step=r;p.geometry=r+31;p.face=f;p.stage=1;
    p.gasCell=f;p.solidCell=f%3;p.filmFace=f%5;p.consumerMask=ConsumeGas;
    const double v=scale*(1+(r*19+f*7+kind)%41);p.mass=v;p.energy=-13*v;p.momentum={2*v,-3*v,4*v};
    p.conductive=5*v;p.advective=6*v;p.pressureWork=-7*v;p.viscousWork=8*v;p.radiation=-9*v;p.liquidKineticAdvection=10*v;
    for(int s=0;s<Ns;++s){p.species[s]=(s+1)*v;p.pore[s]=-(s+2)*v;p.poreSweep[s]=(s+3)*v;}
    for(int c=0;c<Nc;++c)p.condensed[c]=-(c+4)*v;return p;
}
int main(){try{
    std::size_t cases=0,slices=0,failures=0,rejected=0;
    for(std::size_t faces:{0u,1u,3u,17u,129u})for(int mode=0;mode<12;++mode){
        materialSliceProbe::Test t;t.faces=faces;t.particles=mode%2;t.recessionCase=mode;const int frames=mode==10?0:(mode==11?1:7);
        const double dt=mode%3==0?0.125:0.1;Clock time;
        for(int r=0;r<frames;++r){const Clock end=time+dt;t.intervals.push_back({time,end,dt});time=end;
            for(std::size_t f=0;f<faces;++f){CoupledSurfaceDownload q;q.primary=packet(r,f,0,mode==9?1e20:1e-8);
                if((r+f)%3)q.pore=packet(r,f,1,1e-9);q.particleWallEnergy=(r+f)%2?-3e-5*(f+1):0;
                q.radiationEnergy=((r+f)%2?-1:1)*0.005*(r+1)*(f+1);t.surface.push_back(q);}}
        t.total=time;
        if(mode==2||mode==3||mode==5||mode==7||mode==8||mode==10){t.external.resize(mode==8?faces+1:faces);for(std::size_t f=0;f<t.external.size();++f)t.external[f]=0.001*(f+1);}
        auto add=[&](Clock target,int accepted=1){t.targets.push_back(target);t.accepted.push_back(accepted);};
        add(Clock{});add(Clock(dt/8));add(Clock(dt*3/8),0);add(Clock(dt/2));add(Clock(dt));
        add(Clock(dt*1.5));add(Clock(dt*1.5));add(Clock(dt*3.25));add(Clock(dt*20));
        if(frames){add(time);add(time);}
        if(mode==4){t.targets.clear();t.accepted.clear();Clock x(dt);x.add(-1e-20);add(x);x.add(2e-20);add(x);add(time);}
        if(mode==6&&frames>3){t.intervals[1].end=Clock(dt/2);t.intervals[2].begin=Clock(-dt);}
        IntervalHistory h;std::string error;CouplingInterval w;w.end=time.value();h.begin(w,error);
        for(int r=0;r<frames;++r)h.appendAccepted(record(t,r),error);h.setExternalRadiationEnergy(t.external);auto cursor=h.cursor();
        std::cerr<<"case faces="<<faces<<" mode="<<mode<<"\n";
        const auto got=materialSliceProbe::evaluate(t);require(got.size()==t.targets.size(),"result size");int parity=0;
        for(std::size_t i=0;i<got.size();++i){auto candidate=cursor;std::vector<ExchangePacket> packets;std::vector<Real> radiation;
            const bool ok=candidate.takeThroughLocal(native(t.targets[i]),packets,radiation,error);const auto& g=got[i];
            require(ok==(g.plan.error==ugkp::IntervalSliceError::None),"native/device acceptance differs");
            if(ok){require(packets.size()==g.packets.size()&&radiation.size()==g.radiation.size(),"packet/radiation count differs");
                for(std::size_t j=0;j<packets.size();++j)packetEqual(packets[j],g.packets[j]);
                for(std::size_t f=0;f<radiation.size();++f)equal(radiation[f],g.radiation[f]);
                // Independent original CpuMaterialDriver packet traversal;
                // deliberately do not call the new recession math helper.
                const auto material=materialSliceProbe::recessionInventory(mode);const auto physics=materialSliceProbe::recessionPhysics(mode);
                std::vector<double> targets(faces,0);std::vector<int> errors(faces,0);
                for(const auto& packet:packets)if(packet.kind==ExchangeKind::GasSolid){
                    const auto f=packet.face;if(errors[f])continue;MaterialPrimitive primitive;
                    if(!recoverMaterial(material[packet.solidCell],.001,physics,primitive)){errors[f]=1;continue;}
                    for(int c=0;c<Nc;++c)if(packet.condensed[c]!=0){
                        if(!chmt::finite(packet.condensed[c])||!chmt::finite(physics.condensed[c].rho)||!(physics.condensed[c].rho>0)){errors[f]=2;break;}
                        targets[f]-=packet.condensed[c]/physics.condensed[c].rho/(1-primitive.porosity);
                    }
                }
                for(std::size_t f=0;f<faces;++f){require(errors[f]==g.recessionStatus[f],"recession error differs");equal(targets[f],g.recession[f]);
                    auto front=materialSliceProbe::recessionFront(f);
                    if(!errors[f]&&f%4){const double area=1+.1*f;front.solidNormalVelocity=targets[f]/(area*.125);front.normalVelocity=front.solidNormalVelocity;
                        front.solidFront+=.125*front.solidNormalVelocity;front.gasFront+=.125*front.normalVelocity;}
                    equal(front.solidNormalVelocity,g.fronts[f].solidNormalVelocity);equal(front.normalVelocity,g.fronts[f].normalVelocity);
                    equal(front.solidFront,g.fronts[f].solidFront);equal(front.gasFront,g.fronts[f].gasFront);equal(front.temperature,g.fronts[f].temperature);
                }
                bool sourceOk=true;
                for(int c=0;c<3;++c){SolidQ initial;for(int j=0;j<Nc;++j)initial.condensed[j]=1e25*(c+1);
                    for(int j=0;j<Ns;++j)initial.pore[j]=1e25*(j+1);initial.energy=1e30;
                    std::vector<int> indices;for(std::size_t j=0;j<packets.size();++j)if((requiredConsumers(packets[j].kind)&ConsumeSolid)&&packets[j].solidCell==c)indices.push_back(j);
                    SolidQ result;const auto code=applyMaterialPacketRow(initial,packets.data(),indices.data(),0,indices.size(),result);
                    sourceOk=sourceOk&&code==MaterialInventoryCode::Success;
                    require(int(code)==g.inventoryStatus[c],"resident row consumption status differs");
                    if(code==MaterialInventoryCode::Success){for(int j=0;j<Nc;++j)equal(result.condensed[j],g.inventory[c].condensed[j]);for(int j=0;j<Ns;++j)equal(result.pore[j],g.inventory[c].pore[j]);equal(result.energy,g.inventory[c].energy);}
                }
                if(sourceOk&&t.accepted[i])parity^=1;
                if(t.accepted[i])cursor=candidate;else ++rejected;
            }else ++failures;
            require(parity==g.bindingParity,"accepted/candidate state, mesh or gas-change ownership differs");
            equal(windowDuration(cursor.localTime(),WindowTime{}),g.cursor.time.value());++slices;
        }++cases;
    }
    std::cout<<"PASS "<<cases<<" histories, "<<slices<<" slices, "<<failures<<" native errors, "<<rejected<<" uncommitted candidates, "<<values<<" physical fields; maximum scaled difference "<<std::setprecision(17)<<maxError<<"; three device-only graph replays with stable CSR consumption and device state/geometry ownership\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
