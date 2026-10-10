#ifndef CHMT_COUPLING_INTERVALCONTRACTS_H
#define CHMT_COUPLING_INTERVALCONTRACTS_H

#include "core/HostState.H"
#include "coupling/ExchangeLedger.H"
#include <algorithm>
#include <set>
#include <map>
#include <tuple>
#include <utility>

namespace chmt {
// Host-side, CUDA-runtime/OpenFOAM-independent contracts. All transfer amounts
// are extensive. A history is provisional until a synchronized macro commit.
struct IntervalIdentity { std::uint64_t sequence=0,epoch=0; };
struct CouplingInterval { IntervalIdentity identity{}; Real begin=0,end=0; };
struct CouplingControls {
    Real interval=0,minimumInterval=0,gasMaxDt=0,materialMaxSubstep=0;
    Real predictionTolerance=1;
    int maxWindowIterations=8,maxWindowRetries=12;
    bool enforcePredictionAccuracy=false;
};
struct WallFaceSample {
    Real temperature=0; Vec3 velocity{};
    ExchangeKind primaryKind=ExchangeKind::GasSolid;
    Real speciesRate[Ns]{},condensedRate[Nc]{},poreRate[Ns]{},poreSweepRate[Ns]{};
    Real radiationFlux=0,solidNormalVelocity=0,normalVelocity=0;
}; // rates kg/(reference-surface m^2 s), radiation W/m^2; proposals, not transfers.
struct WallKnot {
    Real time=0; std::vector<WallFaceSample> faces{};
    std::vector<Vec3> gasPoints{},solidPoints{};
};
struct MaterialReserveKnot {
    Real time=0;
    std::vector<SolidQ> cumulativeSolid{};
    std::vector<FilmQ> cumulativeFilm{};
}; // signed cumulative INTERNAL component increments; never gas-facing packets.
struct MaterialDonorPlan { CouplingInterval interval{}; std::vector<MaterialReserveKnot> knots{}; };
struct WallProgram { CouplingInterval interval{}; SurfaceMesh surface{}; std::vector<WallKnot> knots{}; MaterialDonorPlan donorPlan{}; };
struct GasWallMatchingSample {
    Real pressure=0;
    ugkwp::gaswall::WallMatchingState<Real,Ns> state{};
}; // actual gas driving snapshot, never an inventory or chemical source.
struct GasIntervalRecord {
    std::uint64_t microSequence=0; Real begin=0,end=0;
    Real gasTraceTime=-1; // one coherent representative gas stage; -1 for isolated legacy fixtures.
    std::uint64_t gasGeometry=0,solidGeometry=0;
    std::vector<ExchangePacket> packets{};
    std::vector<GasPrimitive> gasTrace{}; // OWNER bulk at gasTraceTime, paired with gradient/traction/matching.
    std::vector<GasGradient> gasGradient{};
    std::vector<Vec3> gasTraction{};
    std::vector<GasWallMatchingSample> gasWallMatching{}; // populated only for boundaryLayer.
    std::vector<Real> radiationEnergy{};
    std::vector<unsigned> radiationReceiver{};
}; // trace/traction/radiation have stable SurfaceMesh face order.

inline bool intervalFailure(std::string& error,const char* text){error=text;return false;}
inline bool validCouplingInterval(const CouplingInterval& w){
    return finite(w.begin)&&finite(w.end)&&w.begin>=0&&w.end>w.begin;
}
inline unsigned intervalProducerMask(ExchangeKind kind){
    if(kind==ExchangeKind::ParticleWall)return ConsumeParticle;
    if(kind==ExchangeKind::ParticleGas)return ConsumeParticle|ConsumeGas;
    return ConsumeGas;
}
// Scale only physical integral fields; identity and consumer mask are untouched.
inline ExchangePacket scaledIntervalPacket(const ExchangePacket& p,Real scale){
    ExchangePacket q=p;
    q.mass*=scale;q.momentum=q.momentum*scale;q.energy*=scale;
    q.conductive*=scale;q.advective*=scale;q.pressureWork*=scale;
    q.viscousWork*=scale;q.radiation*=scale;q.liquidKineticAdvection*=scale;
    for(int s=0;s<Ns;++s){q.species[s]*=scale;q.pore[s]*=scale;q.poreSweep[s]*=scale;}
    for(int c=0;c<Nc;++c)q.condensed[c]*=scale;
    return q;
}
inline ExchangePacket subtractIntervalPacket(const ExchangePacket& p,const ExchangePacket& used){
    ExchangePacket q=p;
    q.mass-=used.mass;q.momentum-=used.momentum;q.energy-=used.energy;
    q.conductive-=used.conductive;q.advective-=used.advective;q.pressureWork-=used.pressureWork;
    q.viscousWork-=used.viscousWork;q.radiation-=used.radiation;q.liquidKineticAdvection-=used.liquidKineticAdvection;
    for(int s=0;s<Ns;++s){q.species[s]-=used.species[s];q.pore[s]-=used.pore[s];q.poreSweep[s]-=used.poreSweep[s];}
    for(int c=0;c<Nc;++c)q.condensed[c]-=used.condensed[c];
    return q;
}
inline void addIntervalPacket(ExchangePacket& q,const ExchangePacket& p){
    q.mass+=p.mass;q.momentum+=p.momentum;q.energy+=p.energy;
    q.conductive+=p.conductive;q.advective+=p.advective;q.pressureWork+=p.pressureWork;
    q.viscousWork+=p.viscousWork;q.radiation+=p.radiation;q.liquidKineticAdvection+=p.liquidKineticAdvection;
    for(int s=0;s<Ns;++s){q.species[s]+=p.species[s];q.pore[s]+=p.pore[s];q.poreSweep[s]+=p.poreSweep[s];}
    for(int c=0;c<Nc;++c)q.condensed[c]+=p.condensed[c];
}

class IntervalCursor;
class IntervalHistory {
    CouplingInterval interval_{};
    std::vector<GasIntervalRecord> records_{};
    bool started_=false;
public:
    bool begin(const CouplingInterval& w,std::string& error){
        if(!validCouplingInterval(w))return intervalFailure(error,"invalid coupling interval");
        interval_=w;records_.clear();started_=true;error.clear();return true;
    }
    const CouplingInterval& interval()const{return interval_;}
    const std::vector<GasIntervalRecord>& records()const{return records_;}
    Real end()const{return records_.empty()?interval_.begin:records_.back().end;}
    bool complete()const{return started_&&!records_.empty()&&end()==interval_.end;}
    bool appendAccepted(const GasIntervalRecord& input,std::string& error){
        if(!started_||!finite(input.begin)||!finite(input.end)||input.begin!=end()
            ||input.end<=input.begin||input.end>interval_.end)
            return intervalFailure(error,"noncontiguous or out-of-window gas record");
        if(!records_.empty()&&(records_.back().microSequence==std::numeric_limits<std::uint64_t>::max()
            ||input.microSequence!=records_.back().microSequence+1))
            return intervalFailure(error,"duplicate or noncontiguous gas micro sequence");
        if(!finite(input.gasTraceTime)||(input.gasTraceTime!=-1&&(input.gasTraceTime<input.begin||input.gasTraceTime>input.end)))
            return intervalFailure(error,"gas driving stage time outside accepted interval");
        if(input.gasTraction.size()&&!input.gasTrace.empty()&&input.gasTraction.size()!=input.gasTrace.size())
            return intervalFailure(error,"gas driving trace size mismatch");
        if(!input.gasWallMatching.empty()&&input.gasWallMatching.size()!=input.gasTrace.size())
            return intervalFailure(error,"gas wall matching trace size mismatch");
        for(const auto& sample:input.gasWallMatching){
            if(!finite(sample.pressure)||sample.pressure<=0||!finite(sample.state.temperature)||sample.state.temperature<=0)
                return intervalFailure(error,"invalid gas wall matching thermodynamic trace");
            Real sum=0;for(int s=0;s<Ns;++s){const auto y=sample.state.massFraction[s];if(!finite(y)||y<0)return intervalFailure(error,"invalid matching composition");sum+=y;}
            for(int d=0;d<3;++d)if(!finite(sample.state.velocity[d]))return intervalFailure(error,"invalid matching velocity");
            if(absValue(sum-1)>2e-12||!finite(sample.state.k)||sample.state.k<0||!finite(sample.state.omega)||sample.state.omega<0)
                return intervalFailure(error,"invalid gas wall matching turbulence/composition");
        }
        if(input.radiationEnergy.size()!=input.radiationReceiver.size())
            return intervalFailure(error,"radiation receiver identity missing");
        if(!records_.empty()&&(input.gasTrace.size()!=records_.front().gasTrace.size()
            ||input.gasGradient.size()!=records_.front().gasGradient.size()
            ||input.gasTraction.size()!=records_.front().gasTraction.size()
            ||input.gasWallMatching.size()!=records_.front().gasWallMatching.size()
            ||input.radiationEnergy.size()!=records_.front().radiationEnergy.size()))
            return intervalFailure(error,"surface trace layout changed inside window");
        if(!records_.empty()&&input.radiationReceiver!=records_.front().radiationReceiver)
            return intervalFailure(error,"radiation receiver changed: split coupling window at phase event");
        for(std::size_t f=0;f<input.radiationEnergy.size();++f)
            if(!finite(input.radiationEnergy[f])||(input.radiationReceiver[f]!=ConsumeSolid&&input.radiationReceiver[f]!=ConsumeFilm))
                return intervalFailure(error,"invalid surface radiation integral/receiver");
        if(!input.gasGradient.empty()&&input.gasGradient.size()!=input.gasTrace.size())
            return intervalFailure(error,"gas gradient trace size mismatch");
        for(const auto& g:input.gasGradient){
            if(!finite(g.rho)||!finite(g.temperature)||!finite(g.pressure))return intervalFailure(error,"nonfinite gas scalar gradient");
            for(int j=0;j<3;++j)if(!finite(g.velocity[j]))return intervalFailure(error,"nonfinite gas velocity gradient");
            for(int j=0;j<Ns;++j)if(!finite(g.Y[j]))return intervalFailure(error,"nonfinite gas species gradient");
        }
        for(const auto& t:input.gasTraction)if(!finite(t))return intervalFailure(error,"nonfinite gas traction");
        for(const auto& w:input.gasTrace){
            if(!finite(w.rho)||!finite(w.temperature)||!finite(w.pressure)||!finite(w.velocity)||!finite(w.soundSpeed))
                return intervalFailure(error,"nonfinite gas trace");
            for(int s=0;s<Ns;++s)if(!finite(w.Y[s]))return intervalFailure(error,"nonfinite gas composition trace");
        }
        std::set<std::tuple<std::uint64_t,int,int,int,int,int>> ids;
        GasIntervalRecord record=input;
        for(auto& p:record.packets){
            if(p.stage!=1||p.step!=input.microSequence||p.kind==ExchangeKind::SolidFilm||!validatePacketMath(p,Tolerances{}))
                return intervalFailure(error,"invalid or stale/nonaccepted-stage interval packet");
            if((requiredConsumers(p.kind)&ConsumeGas)&&p.geometry!=input.gasGeometry)
                return intervalFailure(error,"interval packet gas geometry identity mismatch");
            const auto id=std::make_tuple(p.face,static_cast<int>(p.kind),p.gasCell,p.solidCell,p.filmFace,p.particleIndex);
            if(!ids.insert(id).second)return intervalFailure(error,"duplicate interval packet identity");
            const unsigned producer=intervalProducerMask(p.kind);
            if(p.consumerMask!=producer)
                return intervalFailure(error,"interval packet was not applied by its producer");
            // Exact producer ownership is required; importing a zero or already
            // material-consumed mask would lose exactly-once evidence.
        }
        records_.push_back(std::move(record));error.clear();return true;
    }
    bool sampleGasDrive(Real time,std::vector<GasPrimitive>& trace,std::vector<Vec3>& traction,std::string& error)const{
        if(!finite(time)||records_.empty()||time<interval_.begin||time>end())return intervalFailure(error,"gas drive time outside recorded history");
        for(const auto& r:records_)if(time<r.end||(&r==&records_.back()&&time==r.end)){
            trace=r.gasTrace;traction=r.gasTraction;error.clear();return true;}
        return intervalFailure(error,"gas driving history has a gap");
    }
    bool sampleGasDrive(Real time,std::vector<GasPrimitive>& trace,std::vector<GasGradient>& gradient,
        std::vector<Vec3>& traction,std::string& error)const{
        std::vector<GasPrimitive> t;std::vector<Vec3> force;
        if(!sampleGasDrive(time,t,force,error))return false;
        for(const auto& r:records_)if(time<r.end||(&r==&records_.back()&&time==r.end)){
            gradient=r.gasGradient;trace=std::move(t);traction=std::move(force);error.clear();return true;}
        return intervalFailure(error,"gas gradient history has a gap");
    }
    IntervalCursor cursor()const;
};

// The history must outlive this cursor and must not be changed while it is in
// use. Every candidate/replay uses a fresh cursor; failed slice calls are atomic.
class IntervalCursor {
    const IntervalHistory* history_=nullptr;
    Real time_=0;
    std::size_t record_=0;
    std::vector<ExchangePacket> used_{};
    std::vector<Real> radiationUsed_{};
public:
    explicit IntervalCursor(const IntervalHistory& h):history_(&h),time_(h.interval().begin){}
    Real time()const{return time_;}
    bool complete()const{return history_&&history_->complete()&&time_==history_->interval().end;}
    bool takeThrough(Real target,std::vector<ExchangePacket>& packets,std::vector<Real>& radiation,std::string& error){
        if(!history_||!finite(target)||target<=time_||target>history_->end())return intervalFailure(error,"invalid material slice endpoint");
        IntervalCursor candidate=*this;std::vector<ExchangePacket> next;std::vector<Real> rad;
        const auto& records=history_->records();
        while(candidate.time_<target){
            if(candidate.record_>=records.size())return intervalFailure(error,"material slice exceeds gas history");
            const auto& r=records[candidate.record_];
            if(candidate.used_.empty())candidate.used_.resize(r.packets.size());
            if(candidate.radiationUsed_.empty())candidate.radiationUsed_.assign(r.radiationEnergy.size(),0);
            if(rad.empty())rad.assign(r.radiationEnergy.size(),0);
            const Real stop=std::min(target,r.end),fraction=(stop-candidate.time_)/(r.end-r.begin);
            const bool last=stop==r.end;
            for(std::size_t i=0;i<r.packets.size();++i){
                auto part=last?subtractIntervalPacket(r.packets[i],candidate.used_[i]):scaledIntervalPacket(r.packets[i],fraction);
                addIntervalPacket(candidate.used_[i],part);next.push_back(part);
            }
            for(std::size_t f=0;f<r.radiationEnergy.size();++f){
                const Real part=last?r.radiationEnergy[f]-candidate.radiationUsed_[f]:fraction*r.radiationEnergy[f];
                candidate.radiationUsed_[f]+=part;rad[f]+=part;
            }
            candidate.time_=stop;
            if(last){++candidate.record_;candidate.used_.clear();candidate.radiationUsed_.clear();}
        }
        *this=std::move(candidate);packets=std::move(next);radiation=std::move(rad);error.clear();return true;
    }
};
inline IntervalCursor IntervalHistory::cursor()const{return IntervalCursor(*this);}

inline void sampleMaterialDonorPlan(const MaterialDonorPlan& plan,Real time,
    std::vector<SolidQ>& solid,std::vector<FilmQ>& film){
    std::size_t hi=1;
    while(hi+1<plan.knots.size()&&time>plan.knots[hi].time)++hi;
    const auto& a=plan.knots[hi-1];const auto& b=plan.knots[hi];
    const Real fraction=(time-a.time)/(b.time-a.time);
    solid=a.cumulativeSolid;film=a.cumulativeFilm;
    for(std::size_t i=0;i<solid.size();++i){
        for(int c=0;c<Nc;++c)solid[i].condensed[c]+=fraction*(b.cumulativeSolid[i].condensed[c]-a.cumulativeSolid[i].condensed[c]);
        for(int k=0;k<Ns;++k)solid[i].pore[k]+=fraction*(b.cumulativeSolid[i].pore[k]-a.cumulativeSolid[i].pore[k]);
    }
    for(std::size_t i=0;i<film.size();++i){film[i].mass+=fraction*(b.cumulativeFilm[i].mass-a.cumulativeFilm[i].mass);
        for(int k=0;k<Ns;++k)film[i].species[k]+=fraction*(b.cumulativeFilm[i].species[k]-a.cumulativeFilm[i].species[k]);}
}
// A plan is provisional and must be checked against the CPU corrector before
// commit. Without a plan, only macro-base donors and past transfer credits exist.
// Every crossed plan knot is tested, so a later production credit cannot fund
// an earlier gas withdrawal. Internal increments interpolate linearly per slab.
struct DonorRoundoffReport {
    std::uint64_t admittedChecks=0;
    Real maximumDeficit=0,maximumBound=0;
};
// Compensated bookkeeping over the exact supplied binary64 amounts. The
// allowance covers their representation and this arithmetic only; it is not
// a material admissibility tolerance and never modifies an applied packet.
struct IntervalDonorSum {
    long double sum=0,correction=0,absoluteSum=0;
    std::uint64_t operations=0;
    long double value()const{return sum+correction;}
    void add(long double x){
        const long double next=sum+x;
        correction+=std::fabs(sum)>=std::fabs(x)?(sum-next)+x:(x-next)+sum;
        sum=next;absoluteSum+=std::fabs(x);
        if(operations!=std::numeric_limits<std::uint64_t>::max())++operations;
    }
    void merge(const IntervalDonorSum& other){
        const long double oldMagnitude=absoluteSum;
        add(other.value());absoluteSum=oldMagnitude+other.absoluteSum;
        if(other.operations<=std::numeric_limits<std::uint64_t>::max()-operations)operations+=other.operations;
        else operations=std::numeric_limits<std::uint64_t>::max();
    }
    long double bound()const{
        const long double u=std::numeric_limits<long double>::epsilon();
        const long double nu=static_cast<long double>(operations)*u;
        if(nu>=.25L)return std::numeric_limits<long double>::infinity();
        // Neumaier summation: first-order error independent of term count,
        // with the retained operation-count second-order remainder.
        return (2*std::numeric_limits<Real>::epsilon()+u*(4+2*nu)/(1-nu))*absoluteSum;
    }
};
struct DonorReserveWork {
    std::uint64_t lastComponents=0,lastOwners=0,lastPlanSamples=0;
    std::uint64_t totalComponents=0,totalPlanSamples=0;
};
class MaterialDonorReserve {
    std::vector<SolidQ> solid_{};std::vector<FilmQ> film_{};
    std::vector<IntervalDonorSum> accounts_{};
    std::vector<long double> planApplied_{};std::vector<Real> planAccountedTime_{};DonorReserveWork work_{};
    std::uint64_t revision_=0;
    Real time_=0;bool ready_=false;MaterialDonorPlan plan_{};bool hasPlan_=false;
    DonorRoundoffReport roundoff_{};
    std::size_t solidIndex(std::size_t cell,int component)const{return cell*(Nc+Ns)+component;}
    std::size_t filmIndex(std::size_t face,int component)const{return solid_.size()*(Nc+Ns)+face*(1+Ns)+component;}
    static bool admissible(long double balance,long double bound,DonorRoundoffReport& report){
        if(!std::isfinite(balance)||!std::isfinite(bound)||bound<0||balance < -bound)return false;
        if(balance<0){
            if(report.admittedChecks==std::numeric_limits<std::uint64_t>::max())return false;
            ++report.admittedChecks;
            report.maximumDeficit=maxValue(report.maximumDeficit,static_cast<Real>(-balance));
            report.maximumBound=maxValue(report.maximumBound,static_cast<Real>(bound));
        }
        return true;
    }
    void samplePlanComponent(std::size_t index,Real time,long double& value,long double& magnitude)const{
            auto high=std::upper_bound(plan_.knots.begin(),plan_.knots.end(),time,
                [](Real t,const MaterialReserveKnot& knot){return t<knot.time;});
            if(high==plan_.knots.end())--high;
            const auto& b=*high;const auto& a=*(high-1);
            auto get=[&](const MaterialReserveKnot& knot)->Real{
                const auto solidCount=solid_.size()*(Nc+Ns);
                if(index<solidCount){const auto owner=index/(Nc+Ns),component=index%(Nc+Ns);const auto& q=knot.cumulativeSolid[owner];return component<Nc?q.condensed[component]:q.pore[component-Nc];}
                const auto offset=index-solidCount,owner=offset/(1+Ns),component=offset%(1+Ns);const auto& q=knot.cumulativeFilm[owner];return component?q.species[component-1]:q.mass;
            };
            const long double left=get(a),right=get(b);
            const long double fraction=(static_cast<long double>(time)-a.time)/(static_cast<long double>(b.time)-a.time);
            value=left+fraction*(right-left);magnitude=std::fabs(left)+std::fabs(right);
    }
    bool admissibleIdlePrefix(std::size_t index,Real end,DonorRoundoffReport& report,
        std::uint64_t* samples=nullptr)const{
        if(!hasPlan_||end<=planAccountedTime_[index])return true;
        auto knot=std::upper_bound(plan_.knots.begin(),plan_.knots.end(),planAccountedTime_[index],
            [](Real time,const MaterialReserveKnot& value){return time<value.time;});
        auto check=[&](Real time){
            long double value=0,magnitude=0;samplePlanComponent(index,time,value,magnitude);
            if(samples)++*samples;
            const long double change=value-planApplied_[index];
            const long double balance=accounts_[index].value()+change;
            const long double arithmetic=8*std::numeric_limits<long double>::epsilon()*(std::fabs(accounts_[index].value())+std::fabs(change));
            const long double interpolation=8*std::numeric_limits<Real>::epsilon()*(magnitude+std::fabs(planApplied_[index]));
            return admissible(balance,accounts_[index].bound()+arithmetic+interpolation,report);
        };
        for(;knot!=plan_.knots.end()&&knot->time<end;++knot)if(!check(knot->time))return false;
        return check(end);
    }
public:
    class Transaction {
        friend class MaterialDonorReserve;
        struct Update {std::size_t index=0;IntervalDonorSum account{};long double planApplied=0;};
        const MaterialDonorReserve* owner_=nullptr;std::uint64_t revision_=0;
        Real begin_=0,end_=0;std::vector<Update> updates_{};
        DonorRoundoffReport roundoff_{};DonorReserveWork work_{};
    public:
        std::size_t componentCount()const{return updates_.size();}
    };
    // These are signed ACCOUNTING remainders, not physical inventories. A tiny
    // negative remainder is retained and reported, never clipped or spent twice.
    const std::vector<SolidQ>& solid()const{return solid_;}
    const std::vector<FilmQ>& film()const{return film_;}
    const DonorRoundoffReport& roundoff()const{return roundoff_;}
    const DonorReserveWork& work()const{return work_;}
    Real time()const{return time_;}
    bool reset(const std::vector<SolidQ>& solid,const std::vector<FilmQ>& film,Real time,std::string& error){
        if(!finite(time)||time<0||revision_==std::numeric_limits<std::uint64_t>::max())return intervalFailure(error,"invalid donor reserve time/revision");
        for(const auto& q:solid){for(int c=0;c<Nc;++c)if(!finite(q.condensed[c])||q.condensed[c]<0)return intervalFailure(error,"invalid condensed reserve");for(int s=0;s<Ns;++s)if(!finite(q.pore[s])||q.pore[s]<0)return intervalFailure(error,"invalid pore reserve");}
        for(const auto& q:film){if(!finite(q.mass)||q.mass<0)return intervalFailure(error,"invalid film reserve");for(int s=0;s<Ns;++s)if(!finite(q.species[s])||q.species[s]<0)return intervalFailure(error,"invalid film species reserve");}
        std::vector<IntervalDonorSum> accounts(solid.size()*(Nc+Ns)+film.size()*(1+Ns));
        for(std::size_t i=0;i<solid.size();++i){for(int c=0;c<Nc;++c)accounts[i*(Nc+Ns)+c].add(solid[i].condensed[c]);for(int j=0;j<Ns;++j)accounts[i*(Nc+Ns)+Nc+j].add(solid[i].pore[j]);}
        for(std::size_t i=0;i<film.size();++i){const auto start=solid.size()*(Nc+Ns)+i*(1+Ns);accounts[start].add(film[i].mass);for(int j=0;j<Ns;++j)accounts[start+1+j].add(film[i].species[j]);}
        solid_=solid;film_=film;accounts_=std::move(accounts);planApplied_.assign(accounts_.size(),0);planAccountedTime_.assign(accounts_.size(),time);work_=DonorReserveWork{};time_=time;ready_=true;hasPlan_=false;plan_=MaterialDonorPlan{};roundoff_=DonorRoundoffReport{};++revision_;error.clear();return true;
    }
    bool setPlan(const MaterialDonorPlan& plan,std::string& error){
        if(!ready_||revision_==std::numeric_limits<std::uint64_t>::max()||!validCouplingInterval(plan.interval)||plan.interval.begin!=time_
            ||plan.knots.size()<2||plan.knots.front().time!=time_||plan.knots.back().time!=plan.interval.end)
            return intervalFailure(error,"invalid material donor plan interval");
        for(std::size_t k=0;k<plan.knots.size();++k){const auto& knot=plan.knots[k];
            if(!finite(knot.time)||(k&&knot.time<=plan.knots[k-1].time)
                ||knot.cumulativeSolid.size()!=solid_.size()||knot.cumulativeFilm.size()!=film_.size())
                return intervalFailure(error,"invalid material donor plan layout");
            for(const auto& q:knot.cumulativeSolid){
                for(int c=0;c<Nc;++c)if(!finite(q.condensed[c])||(!k&&q.condensed[c]!=0))return intervalFailure(error,"invalid planned condensed increment");
                for(int j=0;j<Ns;++j)if(!finite(q.pore[j])||(!k&&q.pore[j]!=0))return intervalFailure(error,"invalid planned pore increment");
            }
            for(const auto& q:knot.cumulativeFilm){
                if(!finite(q.mass)||(!k&&q.mass!=0))return intervalFailure(error,"invalid planned film increment");
                for(int j=0;j<Ns;++j)if(!finite(q.species[j])||(!k&&q.species[j]!=0))return intervalFailure(error,"invalid planned film species increment");
            }
        }
        plan_=plan;hasPlan_=true;++revision_;error.clear();return true;
    }
    bool prepare(const GasIntervalRecord& r,Transaction& output,std::string& error)const{
        if(!ready_||r.begin!=time_||!finite(r.end)||r.end<=r.begin)return intervalFailure(error,"donor record is not contiguous");
        struct Change {
            IntervalDonorSum withdrawal{},delta{},next{};
            long double internal=0,internalMagnitude=0,planEndpoint=0;
        };
        std::map<std::size_t,Change> changed;
        std::set<std::pair<bool,int>> owners;
        for(const auto& p:r.packets){
            if(p.stage!=1||p.step!=r.microSequence||p.kind==ExchangeKind::SolidFilm||!validatePacketMath(p,Tolerances{}))return intervalFailure(error,"invalid or stale donor packet");
            const auto d=packetDelta(p);const unsigned consumers=requiredConsumers(p.kind);
            if(consumers&ConsumeSolid){
                if(p.solidCell<0||static_cast<std::size_t>(p.solidCell)>=solid_.size())return intervalFailure(error,"solid donor owner out of range");
                owners.insert({false,p.solidCell});
                for(int c=0;c<Nc;++c){auto& q=changed[solidIndex(p.solidCell,c)];q.withdrawal.add(maxValue(0,-d.solid.condensed[c]));q.delta.add(d.solid.condensed[c]);}
                for(int s=0;s<Ns;++s){auto& q=changed[solidIndex(p.solidCell,Nc+s)];
                    if(p.kind==ExchangeKind::PoreGas){q.withdrawal.add(std::max(0.L,static_cast<long double>(p.pore[s])-p.poreSweep[s]));q.withdrawal.add(maxValue(0,p.poreSweep[s]));}
                    else q.withdrawal.add(maxValue(0,-d.solid.pore[s]));
                    q.delta.add(d.solid.pore[s]);}
            }
            if(consumers&ConsumeFilm){
                if(p.filmFace<0||static_cast<std::size_t>(p.filmFace)>=film_.size())return intervalFailure(error,"film donor owner out of range");
                owners.insert({true,p.filmFace});
                auto& q=changed[filmIndex(p.filmFace,0)];q.withdrawal.add(maxValue(0,-d.film.mass));q.delta.add(d.film.mass);
                for(int s=0;s<Ns;++s){auto& z=changed[filmIndex(p.filmFace,1+s)];z.withdrawal.add(maxValue(0,-d.film.species[s]));z.delta.add(d.film.species[s]);}
            }
        }
        DonorReserveWork work=work_;work.lastComponents=changed.size();work.lastOwners=owners.size();work.lastPlanSamples=0;
        std::vector<Real> checks{r.begin};
        if(hasPlan_){
            if(r.end>plan_.interval.end)return intervalFailure(error,"gas record exceeds donor plan");
            auto knot=std::upper_bound(plan_.knots.begin(),plan_.knots.end(),r.begin,
                [](Real time,const MaterialReserveKnot& value){return time<value.time;});
            for(;knot!=plan_.knots.end()&&knot->time<r.end;++knot)checks.push_back(knot->time);
        }
        checks.push_back(r.end);DonorRoundoffReport report=roundoff_;
        // Only touched boundary-owner components are sampled. Internal-plan
        // increments are applied lazily from that component's last visit,
        // so skipping an unrelated gas microstep does not lose production.
        auto sampleComponent=[&](std::size_t index,Real time,long double& value,long double& magnitude){
            samplePlanComponent(index,time,value,magnitude);++work.lastPlanSamples;
        };
        // Validate every skipped plan knot since this component's last visit.
        // An endpoint recovery or later incoming transfer cannot erase an
        // inadmissible earlier prefix. Only the currently touched owners scan.
        if(hasPlan_)for(const auto& entry:changed)
            if(!admissibleIdlePrefix(entry.first,r.begin,report,&work.lastPlanSamples))
                return intervalFailure(error,"deferred material donor prefix was exhausted");
        for(Real t:checks){
            const long double fraction=(static_cast<long double>(t)-r.begin)/(static_cast<long double>(r.end)-r.begin);
            for(auto& entry:changed){const auto j=entry.first;auto& q=entry.second;
                if(hasPlan_){sampleComponent(j,t,q.planEndpoint,q.internalMagnitude);q.internal=q.planEndpoint-planApplied_[j];q.internalMagnitude+=std::fabs(planApplied_[j]);}
                const long double used=fraction*q.withdrawal.value();
                const long double balance=accounts_[j].value()+q.internal-used;
                const long double arithmetic=8*std::numeric_limits<long double>::epsilon()*(std::fabs(accounts_[j].value())+std::fabs(q.internal)+std::fabs(used));
                const long double interpolation=8*std::numeric_limits<Real>::epsilon()*q.internalMagnitude;
                const long double bound=accounts_[j].bound()+fraction*q.withdrawal.bound()+arithmetic+interpolation;
                if(!admissible(balance,bound,report))return intervalFailure(error,"shared material donor prefix exhausted beyond arithmetic roundoff");
            }
        }
        for(auto& entry:changed){const auto j=entry.first;auto& q=entry.second;
            q.next=accounts_[j];q.next.add(q.internal);q.next.merge(q.delta);
            const long double bound=q.next.bound()+8*std::numeric_limits<Real>::epsilon()*q.internalMagnitude;
            if(!admissible(q.next.value(),bound,report))return intervalFailure(error,"resulting material reserve exceeds arithmetic roundoff");
        }
        if(work.lastComponents>std::numeric_limits<std::uint64_t>::max()-work.totalComponents
            ||work.lastPlanSamples>std::numeric_limits<std::uint64_t>::max()-work.totalPlanSamples)
            return intervalFailure(error,"donor work counter overflow");
        work.totalComponents+=work.lastComponents;work.totalPlanSamples+=work.lastPlanSamples;
        if(revision_==std::numeric_limits<std::uint64_t>::max())return intervalFailure(error,"donor revision overflow");
        Transaction candidate;candidate.owner_=this;candidate.revision_=revision_;
        candidate.begin_=r.begin;candidate.end_=r.end;candidate.roundoff_=report;candidate.work_=work;
        candidate.updates_.reserve(changed.size());
        for(const auto& entry:changed){typename Transaction::Update update;
            update.index=entry.first;update.account=entry.second.next;update.planApplied=entry.second.planEndpoint;candidate.updates_.push_back(update);}
        output=std::move(candidate);error.clear();return true;
    }
    bool readyToCommit(const Transaction& transaction)const noexcept{
        return transaction.owner_==this&&transaction.revision_==revision_&&transaction.begin_==time_;
    }
    // Precondition: readyToCommit() was checked and this reserve has not been
    // mutated since. There are no allocations or fallible operations here.
    // This is the publication operation after another owner's history append.
    void commitPrepared(Transaction&& transaction) noexcept{
        for(const auto& update:transaction.updates_){const auto j=update.index;
            accounts_[j]=update.account;if(hasPlan_){planApplied_[j]=update.planApplied;planAccountedTime_[j]=transaction.end_;}
            const Real remaining=static_cast<Real>(update.account.value());const auto solidCount=solid_.size()*(Nc+Ns);
            if(j<solidCount){const auto owner=j/(Nc+Ns),component=j%(Nc+Ns);if(component<Nc)solid_[owner].condensed[component]=remaining;else solid_[owner].pore[component-Nc]=remaining;}
            else{const auto offset=j-solidCount,owner=offset/(1+Ns),component=offset%(1+Ns);if(component)film_[owner].species[component-1]=remaining;else film_[owner].mass=remaining;}
        }
        roundoff_=transaction.roundoff_;work_=transaction.work_;time_=transaction.end_;++revision_;
        transaction.owner_=nullptr;
    }
    bool commit(Transaction&& transaction,std::string& error){
        if(!readyToCommit(transaction))return intervalFailure(error,"stale or foreign material donor transaction");
        commitPrepared(std::move(transaction));error.clear();return true;
    }
    bool apply(const GasIntervalRecord& record,std::string& error){
        Transaction transaction;return prepare(record,transaction,error)&&commit(std::move(transaction),error);
    }
    // Macro-final validation also covers owners that were never touched again.
    // This full-component scan is not called by gas microstep prepare/commit.
    bool validatePlanPrefix(std::string& error,DonorRoundoffReport* output=nullptr)const{
        if(!ready_)return intervalFailure(error,"uninitialized donor prefix validation");
        DonorRoundoffReport report=roundoff_;
        if(hasPlan_)for(std::size_t j=0;j<accounts_.size();++j)
            if(!admissibleIdlePrefix(j,time_,report))return intervalFailure(error,"untouched material donor plan prefix was exhausted");
        if(output)*output=report;
        error.clear();return true;
    }

};

enum class WindowDecision { Accept,Replay,Shorten,Reject };
struct WindowAssessment {
    bool conservative=true,gclValid=true,donorValid=true,regimeConsistent=true,geometryMatches=true;
    bool recoverable=true;
    Real predictionError=0;
};
class CouplingWindowController {
    CouplingInterval interval_{};CouplingControls controls_{};
    int iterations_=0,retries_=0;bool ready_=false;
public:
    const CouplingInterval& interval()const{return interval_;}
    int iterations()const{return iterations_;}int retries()const{return retries_;}
    bool begin(Real time,Real target,std::uint64_t sequence,const CouplingControls& controls,std::string& error){
        if(!finite(time)||time<0||!finite(target)||target<=time||!finite(controls.interval)||controls.interval<=0
            ||!finite(controls.minimumInterval)||controls.minimumInterval<=0||controls.minimumInterval>controls.interval
            ||!finite(controls.gasMaxDt)||controls.gasMaxDt<=0||!finite(controls.materialMaxSubstep)||controls.materialMaxSubstep<0
            ||!finite(controls.predictionTolerance)||controls.predictionTolerance<0||controls.maxWindowIterations<1||controls.maxWindowRetries<0)
            return intervalFailure(error,"invalid independent coupling controls");
        CouplingInterval next;next.identity.sequence=sequence;next.begin=time;next.end=std::min(target,time+controls.interval);
        if(!validCouplingInterval(next))return intervalFailure(error,"coupling endpoint does not advance");
        interval_=next;controls_=controls;iterations_=1;retries_=0;ready_=true;error.clear();return true;
    }
    WindowDecision assess(const WindowAssessment& a)const{
        if(!ready_||!a.recoverable||!finite(a.predictionError)||a.predictionError<0)return WindowDecision::Reject;
        if(!a.conservative||!a.gclValid||!a.donorValid)return WindowDecision::Shorten;
        if(!a.regimeConsistent||!a.geometryMatches)return WindowDecision::Replay;
        if(controls_.enforcePredictionAccuracy&&a.predictionError>controls_.predictionTolerance)return WindowDecision::Replay;
        return WindowDecision::Accept;
    }
    bool replay(std::string& error){
        if(!ready_||iterations_>=controls_.maxWindowIterations||interval_.identity.epoch==std::numeric_limits<std::uint64_t>::max())return intervalFailure(error,"coupling replay limit reached");
        ++iterations_;++interval_.identity.epoch;error.clear();return true;
    }
    bool shorten(Real interval,std::string& error){
        if(!ready_||!finite(interval)||interval<controls_.minimumInterval||interval>=interval_.end-interval_.begin
            ||retries_>=controls_.maxWindowRetries||interval_.identity.epoch==std::numeric_limits<std::uint64_t>::max())return intervalFailure(error,"coupling retry interval/limit invalid");
        const Real endpoint=interval_.begin+interval;
        if(!finite(endpoint)||endpoint<=interval_.begin||endpoint>=interval_.end)return intervalFailure(error,"coupling retry endpoint invalid");
        interval_.end=endpoint;++interval_.identity.epoch;++retries_;iterations_=1;error.clear();return true;
    }
};
} // namespace chmt
#endif
