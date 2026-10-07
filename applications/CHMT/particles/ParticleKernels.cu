#include "particles/ParticleKernels.H"
#include "particles/ParticleMath.H"
#include "gas/KernelSupport.cuh"
namespace chmt { namespace {
__device__ Real particleTemperature(const ParticleQ& q,const CondensedThermo& p){return (q.energy/q.mass-p.e0)/p.cp0;}
__device__ int solidReceiver(int face,SurfaceView surface){for(int f=0;f<surface.nFaces;++f)if(surface.gasFace[f]==face)return surface.solidCell[f];return -1;}
__device__ bool attachedMotion(Vec3 point,int face,GeometryView g,Real fraction,Real dt,Vec3& mapped,Vec3& velocity){
    const int begin=g.faceOffsets[face],end=g.faceOffsets[face+1];const int a=g.facePoints[begin];
    for(int k=begin+1;k+1<end;++k){const int b=g.facePoints[k],c=g.facePoints[k+1];
        const Vec3 aa=g.oldPoints[a]+(g.newPoints[a]-g.oldPoints[a])*fraction;
        const Vec3 bb=g.oldPoints[b]+(g.newPoints[b]-g.oldPoints[b])*fraction;
        const Vec3 cc=g.oldPoints[c]+(g.newPoints[c]-g.oldPoints[c])*fraction;
        if(mapTrianglePoint(point,aa,bb,cc,g.newPoints[a],g.newPoints[b],g.newPoints[c],mapped)){
            Vec3 previous;if(!mapTrianglePoint(point,aa,bb,cc,g.oldPoints[a],g.oldPoints[b],g.oldPoints[c],previous))return false;
            velocity=(mapped-previous)/dt;return true;
        }
    }
    return false;
}
__global__ void particleKernel(const ParticleQ* base,const ParticleQ* evaluation,ParticleQ* out,int count,
    GasView gas,SolidView solid,GeometryView geometry,SurfaceView surface,PhysicsConfig physics,Real dt,
    std::uint64_t step,int stage,int offset,ExchangePacket* packets,Real* radiation,Real* body,Real* support,Vec3* supportImpulse,DeviceStatus* status){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count||gasStopped(status))return;
    const ParticleQ b=base[i];ParticleQ e=evaluation[i];if(e.mass==0)e=b;ParticleQ next=b;
    ExchangePacket pg,pw;pg.kind=ExchangeKind::ParticleGas;pg.gasCell=e.cell;pg.particleIndex=i;pg.step=step;pg.stage=stage;pg.geometry=geometry.geometryVersion;pg.face=b.id;
    pw=pg;radiation[i]=body[i]=support[i]=0;supportImpulse[i]={};
    if(b.state==ParticleDeposited||b.state==ParticleEscaped){out[i]=b;packets[offset+2*i]=pg;packets[offset+2*i+1]=pw;return;}
    const auto thermo=physics.condensed[physics.particle.condensedIndex];const Real Tp=particleTemperature(e,thermo);
    if(!(Tp>=thermo.Tmin&&Tp<=thermo.Tmax)||e.mass<=0||e.diameter<=0||e.cell<0||e.cell>=gas.nCells){gasDeviceError(status,ErrorCode::Inventory,i,Tp,ErrorLocation::Particle);return;}
    const Real area=3.14159265358979323846*e.diameter*e.diameter;
    if(b.state==ParticleFlying){const auto w=gas.primitive[e.cell];const Real gasCapacity=gasHeatCapacity(gas.q[e.cell],physics),particleCapacity=e.mass*thermo.cp0;
        const Real conductance=e.state==ParticleFlying?physics.particle.heatTransferCoefficient*area:0;
        const Real drag=e.state==ParticleFlying?3*3.14159265358979323846*physics.gasViscosity*e.diameter:0;
        // Conservative aggregate relaxation bound, including every particle
        // sharing this gas cell. It is explicit, not an AP stiff integrator.
        Real gasHeatRate=0,gasDragRate=0,maxParticleHeat=0,maxParticleDrag=0;
        for(int j=0;j<count;++j)if(evaluation[j].state==ParticleFlying&&evaluation[j].cell==e.cell){const auto q=evaluation[j];const Real a=3.14159265358979323846*q.diameter*q.diameter;
            const Real h=physics.particle.heatTransferCoefficient*a,d=3*3.14159265358979323846*physics.gasViscosity*q.diameter;
            gasHeatRate+=h/gasCapacity;gasDragRate+=d/gas.q[e.cell].mass;maxParticleHeat=maxValue(maxParticleHeat,h/(q.mass*thermo.cp0));maxParticleDrag=maxValue(maxParticleDrag,d/q.mass);}
        const Real radiationTemperature=maxValue(Tp,physics.ambientTemperature);
        const Real radiationRate=physics.enableRadiation?4*physics.particle.emissivity*5.670374419e-8*area*::pow(radiationTemperature,3)/particleCapacity:0;
        if(dt*maxValue(gasHeatRate+maxParticleHeat+radiationRate,gasDragRate+maxParticleDrag)>.5){gasDeviceError(status,ErrorCode::Inventory,i,dt,ErrorLocation::Particle);return;}
        const Vec3 force=(w.velocity-e.velocity)*drag;ExchangePacket exchange;
        if(!makeParticleGasExchange(b,e,w.velocity,w.temperature,gasCapacity,physics,dt,force,conductance*(w.temperature-Tp),next,exchange,status,stage))return;
        exchange.step=step;exchange.stage=stage;exchange.geometry=geometry.geometryVersion;exchange.face=b.id;exchange.particleIndex=i;pg=exchange;
        body[i]=b.mass*dot(physics.gravity,next.position-b.position);
        // Track the midpoint position segment through linearly moving triangles.
        Vec3 start=b.position,end=next.position;Real elapsed=0;int cell=b.cell;int crossings=0;
        while(elapsed<1&&crossings<physics.particle.maxFaceCrossings){Real earliest=2;int hit=-1;Vec3 normal;
            for(int entry=geometry.cellFaceOffsets[cell];entry<geometry.cellFaceOffsets[cell+1];++entry){const int f=geometry.cellFaces[entry];if(geometry.boundaryKind[f]==BoundaryKind::Empty)continue;const int begin=geometry.faceOffsets[f],finish=geometry.faceOffsets[f+1];const int a=geometry.facePoints[begin];
                for(int k=begin+1;k+1<finish;++k){const int bb=geometry.facePoints[k],c=geometry.facePoints[k+1];Real fraction;Vec3 n;
                    auto at=[&](int v){return geometry.oldPoints[v]+(geometry.newPoints[v]-geometry.oldPoints[v])*elapsed;};
                    if(movingTriangleIntersection(start,end,at(a),at(bb),at(c),geometry.newPoints[a],geometry.newPoints[bb],geometry.newPoints[c],fraction,n)&&fraction<earliest){earliest=fraction;hit=f;normal=n;}}
            }
            if(hit<0){next.position=end;next.cell=cell;elapsed=1;break;}
            ++crossings;const Real eventTime=elapsed+(1-elapsed)*earliest;const Vec3 location=start+(end-start)*earliest;
            const int other=geometry.owner[hit]==cell?geometry.neighbour[hit]:geometry.owner[hit];
            if(other>=0){cell=other;start=location;elapsed=eventTime;continue;}
            const auto kind=geometry.boundaryKind[hit];
            if(kind==BoundaryKind::Periodic){const int partner=geometry.periodicPartner[hit];const Vec3 translation=geometry.faceCentre[partner]-geometry.faceCentre[hit];cell=geometry.owner[partner];start=location+translation;end+=translation;elapsed=eventTime;continue;}
            if(kind==BoundaryKind::Outlet){gasDeviceError(status,ErrorCode::Unsupported,i,hit,ErrorLocation::Particle);return;}
            if(kind!=BoundaryKind::NoSlip&&kind!=BoundaryKind::Slip&&kind!=BoundaryKind::Interface){gasDeviceError(status,ErrorCode::Unsupported,i,hit,ErrorLocation::Particle);return;}
            // A contact event terminates the COMMON interval. Retry to its
            // measured time; do not apply a full-step free-flight source after
            // an earlier impact, or lose the post-impact fraction of a step.
            if(particleStageEventNeedsSplit(dt,eventTime,stage)){gasDeviceError(status,ErrorCode::Inventory,i,-dt*eventTime,ErrorLocation::Particle);return;}
            Vec3 wall,attachedEndpoint;if(!attachedMotion(location,hit,geometry,eventTime,dt,attachedEndpoint,wall)){gasDeviceError(status,ErrorCode::InvalidInput,i,hit,ErrorLocation::Particle);return;}Vec3 velocity,impulse;Real work=0;
            if(physics.particle.contactDuration>0||physics.particle.contact==ParticleContactMode::Deposit){const int receiver=solidReceiver(hit,surface);if(receiver<0){gasDeviceError(status,ErrorCode::Unsupported,i,hit,ErrorLocation::Particle);return;}
                next.state=ParticleContact;next.cell=cell;next.face=hit;next.contactAge=(1-eventTime)*dt;next.contactDuration=physics.particle.contactDuration;next.contactArea=.25*area;next.position=attachedEndpoint;
                // A stationary SolidQ substrate has no kinetic inventory. Impact
                // KE is deposited as solid internal energy exactly once.
                if(physics.particle.contact==ParticleContactMode::ElasticRebound){
                    Vec3 captureImpulse;Real captureWork=0;if(!captureElasticContact(next,wall,captureImpulse,captureWork)){gasDeviceError(status,ErrorCode::Inventory,i,0,ErrorLocation::Particle);return;}
                    support[i]+=captureWork;supportImpulse[i]-=captureImpulse;
                }else{
                    pw=pg;pw.kind=ExchangeKind::ParticleWall;pw.gasCell=-1;pw.solidCell=receiver;pw.mass=0;pw.momentum=next.velocity*next.mass;pw.energy=pw.advective=.5*next.mass*dot(next.velocity,next.velocity);pw.conductive=pw.viscousWork=0;next.velocity={};
                }
                elapsed=1;break;
            }
            if(!elasticWallRebound(next.velocity,wall,normal,next.mass,velocity,impulse,work)){gasDeviceError(status,ErrorCode::PropertyRange,i,0,ErrorLocation::Particle);return;}
            next.velocity=velocity;support[i]+=work;supportImpulse[i]-=impulse;
            next.position=location;next.cell=cell;start=location;end=location+velocity*((1-eventTime)*dt);elapsed=eventTime;
        }
        if(elapsed<1){gasDeviceError(status,ErrorCode::Inventory,i,crossings,ErrorLocation::Particle);return;}
        (void)particleCapacity;
    }
    if(b.state==ParticleContact&&b.contactAge<b.contactDuration
        &&particleStageEventNeedsSplit(dt,(b.contactDuration-b.contactAge)/dt,stage)){
        gasDeviceError(status,ErrorCode::Inventory,i,-(b.contactDuration-b.contactAge),ErrorLocation::Particle);return;
    }
    if(b.state==ParticleContact){const int receiver=solidReceiver(b.face,surface);if(receiver<0||receiver>=solid.nCells){gasDeviceError(status,ErrorCode::Unsupported,i,b.face,ErrorLocation::Particle);return;}
        Real wallTemperature=0;if(!recoverSolid(solid.q[receiver],physics,wallTemperature)){gasDeviceError(status,ErrorCode::PropertyRange,i,0,ErrorLocation::Particle);return;}
        const Real conductance=physics.particle.contactHeatTransferCoefficient*b.contactArea;
        Real solidCapacity=0;for(int c=0;c<Nc;++c)solidCapacity+=solid.q[receiver].condensed[c]*minValue(physics.condensed[c].cp0+physics.condensed[c].cp1*physics.condensed[c].Tmin,physics.condensed[c].cp0+physics.condensed[c].cp1*physics.condensed[c].Tmax);
        for(int s=0;s<Ns;++s)solidCapacity+=solid.q[receiver].pore[s]*minValue(physics.species[s].cp0-physics.species[s].R+physics.species[s].cp1*physics.species[s].Tmin,physics.species[s].cp0-physics.species[s].R+physics.species[s].cp1*physics.species[s].Tmax);
        Real totalConductance=0,maximumParticleRate=0;
        for(int j=0;j<count;++j){const auto contact=evaluation[j];
            if(contact.state!=ParticleContact||contact.mass<=0||solidReceiver(contact.face,surface)!=receiver)continue;
            const Real h=physics.particle.contactHeatTransferCoefficient*contact.contactArea;
            totalConductance+=h;
            const Real temperature=particleTemperature(contact,thermo);
            const Real radiatingTemperature=maxValue(temperature,physics.ambientTemperature);
            const Real exposed=maxValue(0,3.14159265358979323846*contact.diameter*contact.diameter-contact.contactArea);
            const Real radiationConductance=physics.enableRadiation?4*physics.particle.emissivity*5.670374419e-8*exposed*::pow(radiatingTemperature,3):0;
            maximumParticleRate=maxValue(maximumParticleRate,(h+radiationConductance)/(contact.mass*thermo.cp0));
        }
        const Real relaxation=aggregateContactRelaxationRate(totalConductance,solidCapacity,maximumParticleRate);
        if(dt*relaxation>.5){gasDeviceError(status,ErrorCode::Inventory,i,dt,ErrorLocation::Particle);return;}
        const Real heat=dt*conductance*(wallTemperature-Tp);next.energy+=heat;next.contactAge+=dt;
        Vec3 wall;if(!attachedMotion(b.position,b.face,geometry,0,dt,next.position,wall)){gasDeviceError(status,ErrorCode::InvalidInput,i,b.face,ErrorLocation::Particle);return;}
        const Vec3 oldVelocity=next.velocity;next.velocity=wall;
        body[i]=next.mass*dot(physics.gravity,next.position-b.position);
        support[i]+=.5*next.mass*(dot(wall,wall)-dot(oldVelocity,oldVelocity))-body[i];
        supportImpulse[i]-=(wall-oldVelocity)*next.mass-physics.gravity*(next.mass*dt);
        pw=pg;pw.kind=ExchangeKind::ParticleWall;pw.gasCell=-1;pw.solidCell=receiver;pw.energy=pw.conductive=-heat;
    }
    if(physics.enableRadiation&&next.mass>0){const Real exposed=e.state==ParticleContact?maxValue(0,area-e.contactArea):area;
        const Real r=physics.particle.emissivity*5.670374419e-8*exposed*(::pow(physics.ambientTemperature,4)-::pow(Tp,4))*dt;next.energy+=r;radiation[i]=r;}
    if(next.state==ParticleContact&&next.contactAge>=next.contactDuration*(1-1e-10)){const int receiver=solidReceiver(next.face,surface);
        if(physics.particle.contact==ParticleContactMode::Deposit){pw.kind=ExchangeKind::ParticleWall;pw.gasCell=-1;pw.solidCell=receiver;pw.mass=next.mass;pw.condensed[physics.particle.condensedIndex]=next.mass;pw.energy+=particleTotalEnergy(next);pw.advective+=particleTotalEnergy(next);next.mass=0;next.energy=0;next.velocity={};next.state=ParticleDeposited;}
        else{
            Vec3 impulse;Real work=0;Vec3 wall,mapped;if(!attachedMotion(next.position,next.face,geometry,1,dt,mapped,wall)){gasDeviceError(status,ErrorCode::InvalidInput,i,next.face,ErrorLocation::Particle);return;}
            if(!releaseElasticContact(next,wall,normalized(geometry.areaVector[next.face]),impulse,work)){gasDeviceError(status,ErrorCode::Inventory,i,0,ErrorLocation::Particle);return;}
            support[i]+=work;supportImpulse[i]-=impulse;next.state=ParticleFlying;next.face=-1;
        }
    }
    if(next.mass>0){const Real T=particleTemperature(next,thermo);if(!finite(T)||T<thermo.Tmin||T>thermo.Tmax){gasDeviceError(status,ErrorCode::PropertyRange,i,T,ErrorLocation::Particle);return;}}
    pg.consumerMask=pw.consumerMask=0;if(!validatePacketMath(pg,physics.tolerances)||!validatePacketMath(pw,physics.tolerances)){gasDeviceError(status,ErrorCode::Packet,i,0,ErrorLocation::Particle);return;}
    out[i]=next;packets[offset+2*i]=pg;packets[offset+2*i+1]=pw;
}
}
int launchParticleExchange(const StateBuffers& base,const StateBuffers& evaluation,StateBuffers& state,GeometryBuffers& geometry,SurfaceBuffers& surface,const PhysicsConfig& p,Real dt,std::uint64_t step,int stage,int offset,DeviceStatus* status,void* stream){
    if(!p.enableParticles||!state.particles.size())return 0;GeometryBuffers absent;
    particleKernel<<<(state.particles.size()+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(base.particles.data(),evaluation.particles.data(),state.particles.data(),state.particles.size(),state.gasView(base,status,geometry.wallDistance.data()),state.solidView(base,status),geometry.view(),surface.view(absent),p,dt,step,stage,offset,state.packets.data(),state.particleRadiation.data(),state.particleBodyWork.data(),state.particleSupportWork.data(),state.particleSupportImpulse.data(),status);return launchError();
}
}
