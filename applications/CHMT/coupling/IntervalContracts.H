#ifndef CHMT_COUPLING_INTERVALCONTRACTS_H
#define CHMT_COUPLING_INTERVALCONTRACTS_H

#include "core/HostState.H"
#include "coupling/WindowTime.H"
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
    Real time=0; WindowTime localTime{-1}; std::vector<WallFaceSample> faces{};
    std::vector<Vec3> gasPoints{},solidPoints{};
};
struct MaterialReserveKnot {
    Real time=0; WindowTime localTime{-1};
    std::vector<SolidQ> cumulativeSolid{};
    std::vector<FilmQ> cumulativeFilm{};
}; // signed cumulative INTERNAL component increments; never gas-facing packets.
struct MaterialDonorPlan { CouplingInterval interval{}; std::vector<MaterialReserveKnot> knots{}; };
struct WallProgram { CouplingInterval interval{}; SurfaceMesh surface{}; std::vector<WallKnot> knots{}; MaterialDonorPlan donorPlan{}; };
template<class Knot> inline WindowTime knotOffset(const CouplingInterval& interval,const Knot& knot){
    return knot.localTime>=WindowTime{}?knot.localTime:windowOffset(interval.begin,knot.time);
}
inline WindowTime intervalDuration(const CouplingInterval& interval){return windowOffset(interval.begin,interval.end);}
struct GasWallMatchingSample {
    Real pressure=0,mechanicalPressure=0; // matching EOS p; same-stage local wall-force p.
    ugkwp::gaswall::WallMatchingState<Real,Ns> state{};
}; // actual gas driving snapshot, never an inventory or chemical source.
struct GasIntervalRecord {
    std::uint64_t microSequence=0; Real begin=0,end=0,actualDt=0;
    WindowTime localBegin{-1},localEnd{-1};
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
        interval_=w;records_.clear();started_=true;error.clear();return true;
    }
    const CouplingInterval& interval()const{return interval_;}
    const std::vector<GasIntervalRecord>& records()const{return records_;}
    Real end()const{return records_.empty()?interval_.begin:records_.back().end;}
    WindowTime localEnd()const{return records_.empty()?WindowTime{}:records_.back().localEnd;}
    bool complete()const{return started_&&!records_.empty()&&localEnd()==intervalDuration(interval_);}
    bool appendAccepted(const GasIntervalRecord& supplied,std::string& error){
        GasIntervalRecord input=supplied;
        if(input.localBegin<WindowTime{})input.localBegin=windowOffset(interval_.begin,input.begin);
        if(input.actualDt==0)input.actualDt=input.end-input.begin;
        if(input.localEnd<WindowTime{})input.localEnd=supplied.actualDt==0?windowOffset(interval_.begin,input.end):input.localBegin+input.actualDt;
        GasIntervalRecord record=std::move(input);
        records_.push_back(std::move(record));error.clear();return true;
    }
    bool sampleGasDriveLocal(WindowTime time,std::vector<GasPrimitive>& trace,std::vector<Vec3>& traction,std::string& error)const{
        if(records_.empty())return intervalFailure(error,"gas drive time outside recorded history");
        for(const auto& r:records_)if(time<r.localEnd||(&r==&records_.back()&&time==r.localEnd)){
            trace=r.gasTrace;traction=r.gasTraction;error.clear();return true;}
        return intervalFailure(error,"gas driving history has a gap");
    }
    bool sampleGasDriveLocal(WindowTime time,std::vector<GasPrimitive>& trace,std::vector<GasGradient>& gradient,
        std::vector<Vec3>& traction,std::string& error)const{
        if(records_.empty())return intervalFailure(error,"gas drive time outside recorded history");
        for(const auto& r:records_)if(time<r.localEnd||(&r==&records_.back()&&time==r.localEnd)){
            trace=r.gasTrace;traction=r.gasTraction;gradient=r.gasGradient;error.clear();return true;}
        return intervalFailure(error,"gas driving history has a gap");
    }
    bool sampleGasDrive(Real time,std::vector<GasPrimitive>& trace,std::vector<Vec3>& traction,std::string& error)const{return sampleGasDriveLocal(windowOffset(interval_.begin,time),trace,traction,error);}
    bool sampleGasDrive(Real time,std::vector<GasPrimitive>& trace,std::vector<GasGradient>& gradient,std::vector<Vec3>& traction,std::string& error)const{return sampleGasDriveLocal(windowOffset(interval_.begin,time),trace,gradient,traction,error);}
    IntervalCursor cursor()const;
};

// The history must outlive this cursor and must not be changed while it is in
// use. Every candidate/replay uses a fresh cursor; failed slice calls are atomic.
class IntervalCursor {
    const IntervalHistory* history_=nullptr;
    WindowTime time_{};
    std::size_t record_=0;
    std::vector<ExchangePacket> used_{};
    std::vector<Real> radiationUsed_{};
public:
    explicit IntervalCursor(const IntervalHistory& h):history_(&h){}
    Real time()const{return windowAbsolute(history_->interval().begin,time_);}
    WindowTime localTime()const{return time_;}
    bool complete()const{return history_&&history_->complete()&&time_==history_->localEnd();}
    bool takeThroughLocal(WindowTime target,std::vector<ExchangePacket>& packets,std::vector<Real>& radiation,std::string& error){
        if(!history_)return intervalFailure(error,"invalid material slice endpoint");
        IntervalCursor candidate=*this;std::vector<ExchangePacket> next;std::vector<Real> rad;
        const auto& records=history_->records();
        while(candidate.time_<target){
            if(candidate.record_>=records.size())return intervalFailure(error,"material slice exceeds gas history");
            const auto& r=records[candidate.record_];
            if(candidate.used_.empty())candidate.used_.resize(r.packets.size());
            if(candidate.radiationUsed_.empty())candidate.radiationUsed_.assign(r.radiationEnergy.size(),0);
            if(rad.empty())rad.assign(r.radiationEnergy.size(),0);
            const WindowTime stop=std::min(target,r.localEnd);
            const Real fraction=windowDuration(stop,candidate.time_)/r.actualDt;
            const bool last=stop==r.localEnd;
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
    bool takeThrough(Real target,std::vector<ExchangePacket>& packets,std::vector<Real>& radiation,std::string& error){return takeThroughLocal(windowOffset(history_->interval().begin,target),packets,radiation,error);}
};
inline IntervalCursor IntervalHistory::cursor()const{return IntervalCursor(*this);}

inline void sampleMaterialDonorPlanLocal(const MaterialDonorPlan& plan,WindowTime localTime,
    std::vector<SolidQ>& solid,std::vector<FilmQ>& film){
    std::size_t hi=1;
    while(hi+1<plan.knots.size()&&localTime>knotOffset(plan.interval,plan.knots[hi]))++hi;
    const auto& a=plan.knots[hi-1];const auto& b=plan.knots[hi];
    const Real fraction=windowFraction(localTime,knotOffset(plan.interval,a),knotOffset(plan.interval,b),windowAbsolute(plan.interval.begin,localTime),a.time,b.time);
    solid=a.cumulativeSolid;film=a.cumulativeFilm;
    for(std::size_t i=0;i<solid.size();++i){
        for(int c=0;c<Nc;++c)solid[i].condensed[c]+=fraction*(b.cumulativeSolid[i].condensed[c]-a.cumulativeSolid[i].condensed[c]);
        for(int k=0;k<Ns;++k)solid[i].pore[k]+=fraction*(b.cumulativeSolid[i].pore[k]-a.cumulativeSolid[i].pore[k]);
    }
    for(std::size_t i=0;i<film.size();++i){film[i].mass+=fraction*(b.cumulativeFilm[i].mass-a.cumulativeFilm[i].mass);
        for(int k=0;k<Ns;++k)film[i].species[k]+=fraction*(b.cumulativeFilm[i].species[k]-a.cumulativeFilm[i].species[k]);}
}
inline void sampleMaterialDonorPlan(const MaterialDonorPlan& plan,Real time,std::vector<SolidQ>& solid,std::vector<FilmQ>& film){sampleMaterialDonorPlanLocal(plan,windowOffset(plan.interval.begin,time),solid,film);}

struct IntervalDonorSum {
    long double sum=0,correction=0;
    long double value()const{return sum+correction;}
    void add(long double x){
        const long double next=sum+x;
        correction+=std::fabs(sum)>=std::fabs(x)?(sum-next)+x:(x-next)+sum;
        sum=next;
    }
    void merge(const IntervalDonorSum& other){add(other.value());}
};
struct DonorReserveWork {
    std::uint64_t lastComponents=0,lastPlanSamples=0;
    std::uint64_t totalComponents=0,totalPlanSamples=0;
};
class MaterialDonorReserve {
    std::vector<SolidQ> solid_{};std::vector<FilmQ> film_{};
    std::vector<IntervalDonorSum> accounts_{};
    std::vector<long double> planApplied_{};DonorReserveWork work_{};
    Real origin_=0;WindowTime time_{};MaterialDonorPlan plan_{};bool hasPlan_=false;
    std::size_t solidIndex(std::size_t cell,int component)const{return cell*(Nc+Ns)+component;}
    std::size_t filmIndex(std::size_t face,int component)const{return solid_.size()*(Nc+Ns)+face*(1+Ns)+component;}
    void samplePlanComponent(std::size_t index,WindowTime time,long double& value)const{
            auto high=std::upper_bound(plan_.knots.begin(),plan_.knots.end(),time,
                [&](WindowTime t,const MaterialReserveKnot& knot){return t<knotOffset(plan_.interval,knot);});
            if(high==plan_.knots.end())--high;
            const auto& b=*high;const auto& a=*(high-1);
            auto get=[&](const MaterialReserveKnot& knot)->Real{
                const auto solidCount=solid_.size()*(Nc+Ns);
                if(index<solidCount){const auto owner=index/(Nc+Ns),component=index%(Nc+Ns);const auto& q=knot.cumulativeSolid[owner];return component<Nc?q.condensed[component]:q.pore[component-Nc];}
                const auto offset=index-solidCount,owner=offset/(1+Ns),component=offset%(1+Ns);const auto& q=knot.cumulativeFilm[owner];return component?q.species[component-1]:q.mass;
            };
            const long double left=get(a),right=get(b);
            const long double fraction=windowDifference(time,knotOffset(plan_.interval,a))/windowDifference(knotOffset(plan_.interval,b),knotOffset(plan_.interval,a));
            value=left+fraction*(right-left);
    }
public:
    class Transaction {
        friend class MaterialDonorReserve;
        struct Update {std::size_t index=0;IntervalDonorSum account{};long double planApplied=0;};
        WindowTime end_{};std::vector<Update> updates_{};
        DonorReserveWork work_{};
    public:
        std::size_t componentCount()const{return updates_.size();}
    };


    const std::vector<SolidQ>& solid()const{return solid_;}
    const std::vector<FilmQ>& film()const{return film_;}
    const DonorReserveWork& work()const{return work_;}
    Real time()const{return windowAbsolute(origin_,time_);}
    bool reset(const std::vector<SolidQ>& solid,const std::vector<FilmQ>& film,Real time,std::string& error){
        std::vector<IntervalDonorSum> accounts(solid.size()*(Nc+Ns)+film.size()*(1+Ns));
        for(std::size_t i=0;i<solid.size();++i){for(int c=0;c<Nc;++c)accounts[i*(Nc+Ns)+c].add(solid[i].condensed[c]);for(int j=0;j<Ns;++j)accounts[i*(Nc+Ns)+Nc+j].add(solid[i].pore[j]);}
        for(std::size_t i=0;i<film.size();++i){const auto start=solid.size()*(Nc+Ns)+i*(1+Ns);accounts[start].add(film[i].mass);for(int j=0;j<Ns;++j)accounts[start+1+j].add(film[i].species[j]);}
        solid_=solid;film_=film;accounts_=std::move(accounts);planApplied_.assign(accounts_.size(),0);work_=DonorReserveWork{};origin_=time;time_=WindowTime{};hasPlan_=false;plan_=MaterialDonorPlan{};error.clear();return true;
    }
    bool setPlan(const MaterialDonorPlan& plan,std::string& error){
        if(plan.knots.size()<2)return intervalFailure(error,"material donor interpolation needs two supplied knots");
        plan_=plan;hasPlan_=true;error.clear();return true;
    }
    bool prepare(const GasIntervalRecord& supplied,Transaction& output,std::string& error)const{
        const GasIntervalRecord& r=supplied;
        const WindowTime end=r.localEnd<WindowTime{}?windowOffset(origin_,r.end):r.localEnd;
        struct Change {
            IntervalDonorSum delta{},next{};
            long double internal=0,planEndpoint=0;
        };
        std::map<std::size_t,Change> changed;
        for(const auto& p:r.packets){
            const auto d=packetDelta(p);const unsigned consumers=requiredConsumers(p.kind);
            if(consumers&ConsumeSolid){
                if(p.solidCell<0||static_cast<std::size_t>(p.solidCell)>=solid_.size())return intervalFailure(error,"solid donor owner out of range");
                for(int c=0;c<Nc;++c){auto& q=changed[solidIndex(p.solidCell,c)];q.delta.add(d.solid.condensed[c]);}
                for(int s=0;s<Ns;++s){auto& q=changed[solidIndex(p.solidCell,Nc+s)];
                    q.delta.add(d.solid.pore[s]);}
            }
            if(consumers&ConsumeFilm){
                if(p.filmFace<0||static_cast<std::size_t>(p.filmFace)>=film_.size())return intervalFailure(error,"film donor owner out of range");
                auto& q=changed[filmIndex(p.filmFace,0)];q.delta.add(d.film.mass);
                for(int s=0;s<Ns;++s){auto& z=changed[filmIndex(p.filmFace,1+s)];z.delta.add(d.film.species[s]);}
            }
        }
        DonorReserveWork work=work_;work.lastComponents=changed.size();work.lastPlanSamples=0;


        for(auto& entry:changed){const auto j=entry.first;auto& q=entry.second;
            if(hasPlan_){samplePlanComponent(j,end,q.planEndpoint);++work.lastPlanSamples;q.internal=q.planEndpoint-planApplied_[j];}
            q.next=accounts_[j];q.next.add(q.internal);q.next.merge(q.delta);
        }
        work.totalComponents+=work.lastComponents;work.totalPlanSamples+=work.lastPlanSamples;
        Transaction candidate;candidate.end_=end;candidate.work_=work;
        candidate.updates_.reserve(changed.size());
        for(const auto& entry:changed){typename Transaction::Update update;
            update.index=entry.first;update.account=entry.second.next;update.planApplied=entry.second.planEndpoint;candidate.updates_.push_back(update);}
        output=std::move(candidate);error.clear();return true;
    }

    void commitPrepared(Transaction&& transaction) noexcept{
        for(const auto& update:transaction.updates_){const auto j=update.index;
            accounts_[j]=update.account;if(hasPlan_){planApplied_[j]=update.planApplied;}
            const Real remaining=static_cast<Real>(update.account.value());const auto solidCount=solid_.size()*(Nc+Ns);
            if(j<solidCount){const auto owner=j/(Nc+Ns),component=j%(Nc+Ns);if(component<Nc)solid_[owner].condensed[component]=remaining;else solid_[owner].pore[component-Nc]=remaining;}
            else{const auto offset=j-solidCount,owner=offset/(1+Ns),component=offset%(1+Ns);if(component)film_[owner].species[component-1]=remaining;else film_[owner].mass=remaining;}
        }
        work_=transaction.work_;time_=transaction.end_;
    }
    bool commit(Transaction&& transaction,std::string& error){
        commitPrepared(std::move(transaction));error.clear();return true;
    }
    bool apply(const GasIntervalRecord& record,std::string& error){
        Transaction transaction;return prepare(record,transaction,error)&&commit(std::move(transaction),error);
    }

};

enum class WindowDecision { Accept,Replay,Shorten,Reject };
struct WindowAssessment {
    bool regimeConsistent=true,geometryMatches=true;
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
        CouplingInterval next;next.identity.sequence=sequence;next.begin=time;next.end=std::min(target,time+controls.interval);
        interval_=next;controls_=controls;iterations_=1;retries_=0;ready_=true;error.clear();return true;
    }
    WindowDecision assess(const WindowAssessment& a)const{
        if(!ready_||!a.recoverable||!finite(a.predictionError)||a.predictionError<0)return WindowDecision::Reject;
        if(!a.regimeConsistent||!a.geometryMatches)return WindowDecision::Replay;
        if(controls_.enforcePredictionAccuracy&&a.predictionError>controls_.predictionTolerance)return WindowDecision::Replay;
        return WindowDecision::Accept;
    }
    bool replay(std::string& error){
        if(!ready_||iterations_>=controls_.maxWindowIterations)return intervalFailure(error,"coupling replay limit reached");
        ++iterations_;++interval_.identity.epoch;error.clear();return true;
    }
    bool shorten(Real interval,std::string& error){
        if(!ready_||retries_>=controls_.maxWindowRetries)return intervalFailure(error,"coupling retry limit reached");
        const Real endpoint=interval_.begin+interval;
        interval_.end=endpoint;++interval_.identity.epoch;++retries_;iterations_=1;error.clear();return true;
    }
};
} // namespace chmt
#endif
