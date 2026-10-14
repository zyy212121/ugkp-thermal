// Read-only historical checkpoint probe. No production code is modified.
// This uses real checkpoint codec and real CPU film operators; reconstructed
// drive and frozen geometry are explicitly narrower than a full owner replay.
#include "restart/Checkpoint.C"
#include "film/CpuFilmDriver.H"
#include "materials/Thermodynamics.H"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>

namespace chmt {
#include "checkpoint_json_type_names.inc"
struct JsonObserver {
    struct Frame { std::vector<std::string> names; std::size_t count=0; bool object=false; };
    std::ostream& out;std::vector<Frame> frames;
    explicit JsonObserver(std::ostream& stream):out(stream){out<<std::setprecision(17);}
    void quoted(const std::string& text){out<<'"';for(char c:text){if(c=='"'||c=='\\')out<<'\\'<<c;else if(c=='\n')out<<"\\n";else out<<c;}out<<'"';}
    void prefix(){if(frames.empty())return;auto& f=frames.back();if(f.count)out<<',';if(f.object){quoted(f.names.at(f.count));out<<':';}++f.count;}
    void operator()(const Real& x){prefix();if(std::isfinite(x))out<<x;else quoted(std::isnan(x)?"NaN":x<0?"-Infinity":"Infinity");}
    void operator()(const int& x){prefix();out<<x;}
    void operator()(const unsigned& x){prefix();out<<x;}
    void operator()(const std::uint64_t& x){prefix();out<<x;}
    void operator()(const bool& x){prefix();out<<(x?"true":"false");}
    void operator()(const std::string& x){prefix();quoted(x);}
    template<class T> void array(const T& x){prefix();out<<'[';frames.push_back({{},0,false});for(const auto& y:x)(*this)(y);frames.pop_back();out<<']';}
    template<class T,std::size_t N>void operator()(const T(&x)[N]){array(x);}
    template<class T,std::size_t N>void operator()(const std::array<T,N>& x){array(x);}
    template<class T>void operator()(const std::vector<T>& x){array(x);}
    template<class T>typename std::enable_if<std::is_enum<T>::value>::type operator()(const T& x){(*this)(static_cast<int>(x));}
    template<class T>typename std::enable_if<!std::is_enum<T>::value>::type operator()(const T& x){
        prefix();out<<'{';std::string name=jsonTypeName(x),schema=std::string(";")+FieldSchema;
        if(name=="ChemistryAudit")name="ChemistryAudit";
        const auto begin=schema.find(";"+name+":")+name.size()+2,end=schema.find(';',begin);
        std::istringstream fields(schema.substr(begin,end-begin));Frame frame;frame.object=true;std::string field;while(fields>>field)frame.names.push_back(field);
        frames.push_back(frame);visit(*this,x);frames.pop_back();out<<'}';
    }
};

bool loadStoredCheckpoint(const std::string& directory,ModelConfig& config,HostState& state,std::string& error){
    // Public production loader supplies the independently decoded state.
    if(!readCheckpoint(directory,config,state,error))return false;
    std::ifstream file(directory+"/state.chmt",std::ios::binary|std::ios::ate);
    const auto size=file.tellg();Bytes bytes(static_cast<std::size_t>(size));file.seekg(0);file.read(reinterpret_cast<char*>(bytes.data()),size);
    Decoder decoder(bytes);decoder.offset=48;const auto configSize=decoder.integer(8);decoder.offset=72;
    decoder(config);std::cout<<"MODEL_DECODE bytes="<<decoder.offset-72<<" declared="<<configSize<<"\n";
    return true;
}
}


void recordCandidate(const std::string& label,const chmt::CpuFilmCandidate& c,std::ostream& snapshots){
    using namespace chmt;Encoder e;e(c.film);e(c.filmAux);e(c.report.budgetDelta);e(c.report.internalEnergyBefore);e(c.report.internalEnergyAfter);e(c.report.reducedEnergyResidual);e(c.report.kineticDefect);e(c.report.numericalMassRoundoff);e(c.report.phaseEvent);e(c.report.edgeFlux);
    for(const auto& b:c.report.rateBudget){e(b.bodyPower);e(b.supportPower);e(b.prescribedTopPower);e(b.radiationPower);e(b.edgeEnergyOutflow);e(b.edgeKineticOutflow);e(b.interfaceKineticOutflow);e(b.dissipationPower);}
    Real mass=0,enthalpy=0;for(const auto& q:c.film){mass+=q.mass;enthalpy+=q.enthalpy;}
    const auto hash=checksum(e.data.data(),e.data.size());std::cout<<"SNAPSHOT label="<<label<<" checksum="<<hash<<" mass="<<mass<<" enthalpy="<<enthalpy<<" boundaryMass="<<c.report.budgetDelta.boundaryMass<<" boundaryEnergy="<<c.report.budgetDelta.boundaryEnergy<<" residual="<<c.report.reducedEnergyResidual<<"\n";
    snapshots<<std::setprecision(17)<<"{\"label\":\""<<label<<"\",\"checksum\":"<<hash<<",\"film\":";JsonObserver jf(snapshots);jf(c.film);snapshots<<",\"filmAux\":";JsonObserver ja(snapshots);ja(c.filmAux);snapshots<<",\"budget\":";JsonObserver jb(snapshots);jb(c.report.budgetDelta);snapshots<<",\"edgeFlux\":";JsonObserver je(snapshots);je(c.report.edgeFlux);snapshots<<",\"internalEnergyBefore\":"<<c.report.internalEnergyBefore<<",\"internalEnergyAfter\":"<<c.report.internalEnergyAfter<<",\"reducedEnergyResidual\":"<<c.report.reducedEnergyResidual<<",\"kineticDefect\":"<<c.report.kineticDefect<<",\"numericalMassRoundoff\":"<<c.report.numericalMassRoundoff<<",\"phaseEvent\":"<<(c.report.phaseEvent?"true":"false")<<",\"rateBudget\":[";
    bool first=true;for(const auto& b:c.report.rateBudget){if(!first)snapshots<<',';first=false;snapshots<<"{\"bodyPower\":"<<b.bodyPower<<",\"supportPower\":"<<b.supportPower<<",\"prescribedTopPower\":"<<b.prescribedTopPower<<",\"radiationPower\":"<<b.radiationPower<<",\"edgeEnergyOutflow\":"<<b.edgeEnergyOutflow<<",\"edgeKineticOutflow\":"<<b.edgeKineticOutflow<<",\"interfaceKineticOutflow\":"<<b.interfaceKineticOutflow<<",\"dissipationPower\":"<<b.dissipationPower<<'}';}snapshots<<"]}\n";
}

int main(int argc,char** argv){
    using namespace chmt;
    if(argc<3){std::cerr<<"Usage: checkpoint_film_probe CHECKPOINT_DIR FULL_JSON\n";return 2;}
    ModelConfig model;HostState base;std::string error;
    if(!loadStoredCheckpoint(argv[1],model,base,error)){std::cerr<<"LOAD "<<error<<'\n';return 3;}
    std::ofstream full(argv[2]);full<<"{\"model\":";JsonObserver modelJson(full);modelJson(model);full<<",\"state\":";JsonObserver stateJson(full);stateJson(base);full<<"}\n";
    std::cout<<std::setprecision(17)<<"MODEL liquidRho="<<model.physics.liquid.rho<<" liquidE0="<<model.physics.liquid.e0<<" liquidReferencePressure="<<model.physics.liquidReferencePressure<<" gasViscosity="<<model.physics.gasViscosity<<" wallFamily="<<int(model.physics.wallModel.family)<<"\n";
    std::ofstream snapshots(std::string(argv[2])+".snapshots.jsonl");
    const Real fullDt=0x1.0624dd2f1a800p-23;
    Real packetMass[4]{},packetEnergy[4]{},phaseHeatRate[4]{};
#include "checkpoint_film_packet_values.inc"
    for(const std::string mode:{"checkpoint_aux","checkpoint_bulk","uniform_pressure_control"}){
        HostState filmState=base;
        // Match the material owner's initial film-evaluation geometry reset.
        filmState.surface.sweptEdgeArea.assign(base.surface.edgeOwner.size(),0);
        std::vector<CpuFilmDrive> drive(base.film.size());
        for(std::size_t f=0;f<drive.size();++f){const auto& a=base.filmAux[f];auto& d=drive[f];const Vec3 n=base.surface.normal[f];
            d.pressure=mode=="uniform_pressure_control"?base.filmAux[0].pressure:a.pressure;
            if(mode=="checkpoint_bulk"){const int cell=base.gasMesh.owner[base.surface.gasFace[f]];GasPrimitive gas;const bool ok=recoverGas(base.gas[cell],base.gasMesh.volumes[cell],model.physics,gas);std::cout<<"BULK f="<<f<<" ok="<<ok<<" pressure="<<gas.pressure<<"\n";d.pressure=gas.pressure;}
            d.endpointPressure=d.pressure;d.shear=Vec3{};d.hasCoupledNormalTrace=true;
            d.bottomNormalVelocity=2*dot(a.meanVelocity,n)-dot(a.topVelocity,n);d.topNormalVelocity=dot(a.topVelocity,n);d.interfaceNormalVelocity=a.normalVelocity;d.solidNormalVelocity=a.solidNormalVelocity;
            std::cout<<"DRIVE mode="<<mode<<" f="<<f<<" pressure="<<d.pressure<<" bottomNormal="<<d.bottomNormalVelocity<<" topNormal="<<d.topNormalVelocity<<" interfaceNormal="<<d.interfaceNormalVelocity<<"\n";
        }
        {std::vector<FilmQ> rates;std::vector<FilmRateBudget> budgets;std::vector<FilmAux> profiles;std::vector<Vec3> gradients;std::vector<Real> side;bool numerical=false;
            const bool ok=detail::completeCpuFilmTransport(filmState,model.physics,fullDt,drive,rates,budgets,error,&profiles,&gradients,&side,&numerical);
            std::cout<<"PROFILE mode="<<mode<<" ok="<<ok<<" error="<<error<<"\n";
            for(std::size_t f=0;f<profiles.size();++f)std::cout<<"GRADIENT mode="<<mode<<" f="<<f<<" xyz="<<gradients[f].x<<","<<gradients[f].y<<","<<gradients[f].z<<" mean="<<profiles[f].meanVelocity.x<<","<<profiles[f].meanVelocity.y<<","<<profiles[f].meanVelocity.z<<"\n";
            for(std::size_t edge=0;edge<filmState.surface.edgeOwner.size();++edge){const auto& m=filmState.surface;const int owner=m.edgeOwner[edge],neighbour=m.edgeNeighbour[edge];const Vec3 velocity=neighbour>=0?(profiles[owner].meanVelocity+profiles[neighbour].meanVelocity)*.5:profiles[owner].meanVelocity;const Real rate=m.edgeLength[edge]*dot(velocity,m.edgeConormal[edge]);std::cout<<"EDGE mode="<<mode<<" e="<<edge<<" owner="<<owner<<" neighbour="<<neighbour<<" lengthRate="<<rate<<"\n";}
        }
        for(int power:{0,1,10,30}){const Real dt=std::ldexp(fullDt,-power);std::vector<CpuFilmForcing> forcing(base.film.size());
            for(std::size_t f=0;f<forcing.size();++f){auto& a=forcing[f];a.mass=-packetMass[f]*(dt/fullDt);a.species[0]=a.mass;a.withdrawnMass=-a.mass;a.withdrawnSpecies[0]=-a.mass;a.energy=-packetEnergy[f]*(dt/fullDt);a.phaseEnergy=phaseHeatRate[f]*dt;a.topWorkIncluded=a.bottomWorkIncluded=true;}
            std::vector<FilmQ> rate;std::vector<FilmRateBudget> budget;std::vector<FilmAux> aux;std::vector<Vec3> gradient;std::vector<Real> side;bool numerical=false;
            bool ok=evaluateCpuFilmTransport(filmState,model.physics,dt,drive,rate,budget,error,&aux,&gradient,&side,&numerical);
            std::cout<<"RESULT mode="<<mode<<" half="<<power<<" dt="<<dt<<" op=evaluate ok="<<ok<<" numerical="<<numerical<<" error="<<error<<"\n";
            numerical=false;Real limited=dt;ok=estimateCpuFilmStep(filmState,model.physics,forcing,drive,dt,dt,limited,error,&numerical);
            std::cout<<"RESULT mode="<<mode<<" half="<<power<<" dt="<<dt<<" op=estimate ok="<<ok<<" numerical="<<numerical<<" limited="<<limited<<" error="<<error<<"\n";
            numerical=false;CpuFilmCandidate next;ok=advanceCpuFilmCandidate(filmState,model.physics,dt,forcing,drive,next,error,&numerical);
            std::cout<<"RESULT mode="<<mode<<" half="<<power<<" dt="<<dt<<" op=advance ok="<<ok<<" numerical="<<numerical<<" error="<<error<<"\n";
            if(ok)recordCandidate(mode+"-advance-half"+std::to_string(power),next,snapshots);
            numerical=false;ok=detail::completeCpuFilmCandidate(filmState,model.physics,dt,forcing,drive,next,error,&numerical);
            std::cout<<"RESULT mode="<<mode<<" half="<<power<<" dt="<<dt<<" op=complete ok="<<ok<<" numerical="<<numerical<<" error="<<error<<"\n";
            if(ok)recordCandidate(mode+"-complete-half"+std::to_string(power),next,snapshots);
        }
    }
    for(const std::string control:{"mass_donor_exhausted","species_donor_exhausted","missing_coupled_trace","missing_film_aux","nonfinite_length_rate","ordinary_outward"}){
        HostState h=base;h.surface.sweptEdgeArea.assign(h.surface.edgeOwner.size(),0);std::vector<CpuFilmDrive> d(h.film.size());std::vector<CpuFilmForcing> f(h.film.size());
        for(std::size_t i=0;i<d.size();++i){const auto& a=h.filmAux[i];const Vec3 n=h.surface.normal[i];d[i].pressure=d[i].endpointPressure=base.filmAux[0].pressure;d[i].hasCoupledNormalTrace=true;d[i].bottomNormalVelocity=2*dot(a.meanVelocity,n)-dot(a.topVelocity,n);d[i].topNormalVelocity=dot(a.topVelocity,n);d[i].interfaceNormalVelocity=a.normalVelocity;d[i].solidNormalVelocity=a.solidNormalVelocity;f[i].topWorkIncluded=f[i].bottomWorkIncluded=true;}
        if(control=="mass_donor_exhausted"){f[0].mass=f[0].species[0]=-2*h.film[0].mass;f[0].withdrawnMass=f[0].withdrawnSpecies[0]=-f[0].mass;}
        if(control=="species_donor_exhausted"){f[0].mass=f[0].species[1]=-.5*h.film[0].mass;f[0].withdrawnMass=f[0].withdrawnSpecies[1]=-f[0].mass;}
        if(control=="missing_coupled_trace")d[0].hasCoupledNormalTrace=false;
        if(control=="missing_film_aux")h.filmAux.clear();
        if(control=="nonfinite_length_rate")h.surface.sweptEdgeArea[0]=std::numeric_limits<Real>::max();
        if(control=="ordinary_outward"){
            const Vec3 v=h.surface.edgeConormal[0]*.2;for(auto& x:h.surface.baseVelocity)x=v;
            h.surface.edgeOwner.resize(1);h.surface.edgeNeighbour.resize(1);h.surface.edgeLength.resize(1);h.surface.edgeConormal.resize(1);h.surface.edgeOwnerOffset.resize(1);h.surface.edgeNeighbourOffset.resize(1);h.surface.sweptEdgeArea.resize(1);
        }
        CpuFilmCandidate next;bool numerical=false;const bool ok=advanceCpuFilmCandidate(h,model.physics,fullDt,f,d,next,error,&numerical);
        std::cout<<"CONTROL name="<<control<<" ok="<<ok<<" numerical="<<numerical<<" error="<<error<<"\n";if(ok)recordCandidate("control-"+control,next,snapshots);
    }

}
