#ifndef CHMT_PARTICLES_PARTICLEMATH_H
#define CHMT_PARTICLES_PARTICLEMATH_H
#include "materials/Thermodynamics.H"
#include "coupling/ExchangeLedger.H"
#include "gpu/GpuCouplingMath.H"
namespace chmt {
static_assert(sizeof(GpuReal)==sizeof(Real),"CHMT particles require the existing FP64 precision contract");
CHMT_HD inline bool particleEventNeedsSplit(Real interval,Real fraction){return finite(interval)&&interval>0&&finite(fraction)&&fraction>0&&fraction<1-1e-8;}
// Stage 0 is only a predictor of the common interval. Even its endpoint
// event lies inside the requested full step and must shorten that full step
// before a captured/reflected/released velocity can enter midpoint transport.
CHMT_HD inline bool particleStageEventNeedsSplit(Real interval,Real fraction,int stage){
    if(!finite(interval)||interval<=0||!finite(fraction)||fraction<=0)return false;
    return stage==0?fraction<=1+1e-10:particleEventNeedsSplit(interval,fraction);
}
CHMT_HD inline Real particleEventRetryDt(Real requested,Real encodedEvent){return finite(encodedEvent)&&encodedEvent<0?minValue(requested,-encodedEvent):requested*.5;}
CHMT_HD inline Real aggregateContactRelaxationRate(Real receiverConductance,Real receiverCapacity,Real maximumParticleRate){
    if(!finite(receiverConductance)||receiverConductance<0||!finite(receiverCapacity)||receiverCapacity<=0
        ||!finite(maximumParticleRate)||maximumParticleRate<0)return std::numeric_limits<Real>::max();
    return receiverConductance/receiverCapacity+maximumParticleRate;
}
enum ParticleState { ParticleFlying=0,ParticleContact=1,ParticleDeposited=2,ParticleEscaped=3 };
CHMT_HD inline Real particleTotalEnergy(const ParticleQ& q){return q.energy+q.contactMechanicalEnergy+.5*q.mass*dot(q.velocity,q.velocity);}
CHMT_HD inline Real gasHeatCapacity(const GasQ& q,const PhysicsConfig& p){Real capacity=0;for(int s=0;s<Ns;++s)capacity+=q.species[s]*(p.species[s].cp0-p.species[s].R);return capacity;}
// Analytic unit reference; production uses bounded ordinary midpoint rates.
CHMT_HD inline Real finiteCapacityHeatReference(Real gasCapacity,Real particleCapacity,
    Real gasTemperature,Real particleTemperature,Real conductance,Real dt){
    return gpuFiniteCapacityHeatExchange(gasCapacity,particleCapacity,gasTemperature,
        particleTemperature,conductance/particleCapacity,dt);
}
CHMT_HD inline bool makeParticleGasExchange(const ParticleQ& base,const ParticleQ& evaluation,
    Vec3 gasVelocity,Real gasTemperature,Real capacity,const PhysicsConfig& physics,Real dt,
    Vec3 dragForce,Real heatPower,ParticleQ& out,ExchangePacket& packet,DeviceStatus* status=nullptr,int stage=1){
    if((stage!=0&&stage!=1)||!finite(dt)||dt<=0||base.mass<=0||evaluation.mass!=base.mass||!finite(capacity)||capacity<=0
        ||!finite(gasVelocity)||!finite(gasTemperature)||gasTemperature<=0||!finite(dragForce)||!finite(heatPower))
        return failStatus(status,ErrorCode::InvalidInput,base.cell,base.mass,ErrorLocation::Particle);
    ParticleQ next=base;next.velocity=base.velocity+dragForce*(dt/base.mass)+physics.gravity*dt;
    // The provisional half-state follows its frozen-force kinematic chord.
    // This independently integrates gravity work and introduces no gas exchange
    // when drag/heat are absent. The accepted full update keeps midpoint drift.
    next.position=stage==0?base.position+(base.velocity+next.velocity)*(.5*dt)
        :base.position+evaluation.velocity*dt;
    next.energy=base.energy+dt*heatPower;
    if(!finite(next.energy)||!finite(next.velocity)||!finite(next.position))return false;
    ExchangePacket transfer;transfer.kind=ExchangeKind::ParticleGas;transfer.gasCell=evaluation.cell;
    transfer.particleIndex=0;transfer.momentum=dragForce*(-dt);
    transfer.conductive=-dt*heatPower;
    // External gravity work is independently fixed by the midpoint path.
    // The remaining actual kinetic change belongs to the paired drag transfer;
    // do not define body work as a sequential-update closure remainder.
    const Real bodyWork=base.mass*dot(physics.gravity,next.position-base.position);
    transfer.viscousWork=stage==0?-dt*dot(dragForce,(base.velocity+next.velocity)*.5)
        :-.5*base.mass*(dot(next.velocity,next.velocity)-dot(base.velocity,base.velocity))+bodyWork;
    transfer.energy=transfer.conductive+transfer.viscousWork;
    out=next;packet=transfer;return true;
}
CHMT_HD inline bool elasticWallRebound(Vec3 incoming,Vec3 wall,Vec3 normal,Real mass,
    Vec3& outgoing,Vec3& particleImpulse,Real& supportWork){
    if(mass<=0||!finite(incoming)||!finite(wall)||!finite(normal)||absValue(mag(normal)-1)>1e-10)return false;
    outgoing=incoming-normal*(2*dot(incoming-wall,normal));
    particleImpulse=(outgoing-incoming)*mass;
    supportWork=.5*mass*(dot(outgoing,outgoing)-dot(incoming,incoming));
    return finite(outgoing)&&finite(supportWork);
}
CHMT_HD inline bool captureElasticContact(ParticleQ& particle,Vec3 wall,Vec3& impulse,Real& work){
    if(particle.mass<=0||!finite(wall)||!finite(particle.velocity)||particle.contactMechanicalEnergy!=0)return false;
    const Real before=particleTotalEnergy(particle);const Vec3 incoming=particle.velocity;
    particle.contactRelativeVelocity=incoming-wall;
    particle.contactMechanicalEnergy=.5*particle.mass*dot(particle.contactRelativeVelocity,particle.contactRelativeVelocity);
    particle.velocity=wall;impulse=(wall-incoming)*particle.mass;work=particleTotalEnergy(particle)-before;
    return finite(work);
}
CHMT_HD inline bool releaseElasticContact(ParticleQ& particle,Vec3 wall,Vec3 normal,Vec3& impulse,Real& work){
    if(particle.mass<=0||!finite(wall)||!finite(normal)||absValue(mag(normal)-1)>1e-10)return false;
    const Real before=particleTotalEnergy(particle);const Vec3 prior=particle.velocity;
    const Vec3 relative=particle.contactRelativeVelocity-normal*(2*dot(particle.contactRelativeVelocity,normal));
    particle.velocity=wall+relative;particle.contactMechanicalEnergy=0;particle.contactRelativeVelocity={};
    impulse=(particle.velocity-prior)*particle.mass;work=particleTotalEnergy(particle)-before;
    return finite(work);
}
CHMT_HD inline Real cubicValue(const Real* c,Real t){return ((c[3]*t+c[2])*t+c[1])*t+c[0];}
CHMT_HD inline bool inTriangle(Vec3 point,Vec3 a,Vec3 b,Vec3 c,Vec3& normal){
    const Vec3 u=b-a,v=c-a,n=cross(u,v);const Real n2=dot(n,n);
    if(!(n2>0)||!finite(n2))return false;
    const Real beta=dot(cross(point-a,v),n)/n2,gamma=dot(cross(u,point-a),n)/n2;
    if(beta<-1e-11||gamma<-1e-11||beta+gamma>1+1e-11)return false;
    normal=n/::sqrt(n2);return true;
}
CHMT_HD inline bool mapTrianglePoint(Vec3 point,Vec3 a,Vec3 b,Vec3 c,
    Vec3 newA,Vec3 newB,Vec3 newC,Vec3& mapped){
    Vec3 n;if(!inTriangle(point,a,b,c,n))return false;
    const Vec3 u=b-a,v=c-a,area=cross(u,v);const Real denominator=dot(area,area);
    if(absValue(dot(point-a,n))>1e-10*maxValue(mag(u),mag(v)))return false;
    const Real beta=dot(cross(point-a,v),area)/denominator;
    const Real gamma=dot(cross(u,point-a),area)/denominator;
    mapped=newA*(1-beta-gamma)+newB*beta+newC*gamma;return finite(mapped);
}
// Linear particle and vertex trajectories give a cubic signed plane volume.
// Derivative roots partition it into monotone intervals, so moving-face roots
// are bracketed without a static-plane approximation or sampled-time misses.
CHMT_HD inline bool movingTriangleIntersection(Vec3 x0,Vec3 x1,Vec3 a0,Vec3 b0,Vec3 c0,
    Vec3 a1,Vec3 b1,Vec3 c1,Real& fraction,Vec3& normal){
    const Vec3 r=x0-a0,dr=(x1-x0)-(a1-a0),u=b0-a0,du=(b1-a1)-u,v=c0-a0,dv=(c1-a1)-v;
    const Vec3 n0=cross(u,v),n1=cross(du,v)+cross(u,dv),n2=cross(du,dv);
    const Real co[4]={dot(r,n0),dot(dr,n0)+dot(r,n1),dot(dr,n1)+dot(r,n2),dot(dr,n2)};
    Real knots[4]={0,1,0,0};int count=2;const Real A=3*co[3],B=2*co[2],C=co[1];
    if(A!=0){const Real d=B*B-4*A*C;if(d>=0){const Real root=::sqrt(d);const Real t0=(-B-root)/(2*A),t1=(-B+root)/(2*A);
        if(t0>0&&t0<1){knots[count++]=t0;}
        if(t1>0&&t1<1&&t1!=t0){knots[count++]=t1;}}}
    else if(B!=0){const Real t=-C/B;if(t>0&&t<1)knots[count++]=t;}
    for(int i=1;i<count;++i)for(int j=i;j>0&&knots[j]<knots[j-1];--j){const Real t=knots[j];knots[j]=knots[j-1];knots[j-1]=t;}
    Real best=2;Vec3 bestNormal;
    for(int k=0;k+1<count;++k){Real lo=knots[k],hi=knots[k+1],fl=cubicValue(co,lo),fh=cubicValue(co,hi);
        if(fl==0&&fh==0)continue; // coplanar motion is contact, not a crossing.
        if(fl!=0&&fh!=0&&((fl>0)==(fh>0)))continue;
        Real t=fl==0?lo:hi;
        if(fl!=0&&fh!=0){for(int iter=0;iter<64;++iter){const Real mid=.5*(lo+hi),fm=cubicValue(co,mid);if((fm>0)==(fl>0)){lo=mid;fl=fm;}else hi=mid;}t=.5*(lo+hi);}
        if(t<=1e-12||t>=best)continue;
        Vec3 n;if(inTriangle(x0+(x1-x0)*t,a0+(a1-a0)*t,b0+(b1-b0)*t,c0+(c1-c0)*t,n)){best=t;bestNormal=n;}
    }
    if(best>1)return false;
    fraction=best;normal=bestNormal;return true;
}
}
#endif
