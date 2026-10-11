#pragma once
#include "../../../common/gasTransport/GasBudgetAudit.H"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

// Host-only diagnostic owner. No output is a solver acceptance gate: a physical
// mismatch is retained as a residual, never folded into a fabricated source.
struct GasBudgetHostLedger
{
    FILE* output=nullptr;
    bool initialized=false,requestActive=false;
    double originTime=0,acceptedTime=0,requestedEnd=0;
    std::size_t sequence=0,deviceBytes=0,hostReadBytes=0;
    double hostSeconds=0;
    std::string identity;
    std::vector<double> volumes,elementMassFactors;
    std::vector<long double> origin;
    std::vector<long double> cumulativeBoundary,cumulativeChemistry;
    int speciesCount=0,elementCount=0;
    ~GasBudgetHostLedger(){if(output)std::fclose(output);}
};
template<class State,class=void>struct GasBudgetHostCapability:std::false_type{};
template<class State>struct GasBudgetHostCapability<State,std::void_t<decltype(std::declval<State>().gasBudgetHost),decltype(std::declval<State>().gasBudgetAudit)>>:std::true_type{};
namespace gasBudgetDetail
{
inline std::string quoted(const std::string& value)
{
    std::ostringstream out;out<<'"';
    for(unsigned char c:value){if(c=='"'||c=='\\')out<<'\\'<<char(c);else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;else out<<char(c);}
    out<<'"';return out.str();
}
template<class T>std::string array(const std::vector<T>& values)
{
    std::ostringstream out;out<<std::setprecision(std::numeric_limits<long double>::max_digits10)<<'[';
    for(std::size_t i=0;i<values.size();++i){if(i)out<<',';const long double value=static_cast<long double>(values[i]);if(!std::isfinite(value))throw std::runtime_error("nonfinite gas budget value");out<<value;}
    out<<']';return out.str();
}
inline std::string names(const std::vector<std::string>& values)
{std::string out="[";for(std::size_t i=0;i<values.size();++i){if(i)out+=',';out+=quoted(values[i]);}return out+"]";}
inline void append(GasBudgetHostLedger& ledger,const std::string& line)
{
    const int fd=::fileno(ledger.output);const auto offset=::lseek(fd,0,SEEK_CUR);const std::string record=line+'\n';
    if(offset<0)throw std::runtime_error("gas budget output position unavailable");
    std::size_t written=0;
    while(written<record.size())
    {
        const auto count=::write(fd,record.data()+written,record.size()-written);
        if(count<=0)
        {
            if(::ftruncate(fd,offset)!=0 || ::lseek(fd,offset,SEEK_SET)<0)
                throw std::runtime_error("gas budget write and output rollback failed");
            throw std::runtime_error("gas budget accepted record write failed");
        }
        written+=std::size_t(count);
    }
}
template<class State>std::vector<double> read(State*s,const double* pointer,std::size_t count)
{
    std::vector<double> data(count);
    if(copyToHost(data.data(),pointer,count,"read opt-in gas budget"))throw std::runtime_error("gas budget device read failed");
    s->gasBudgetHost->hostReadBytes+=count*sizeof(double);
    for(double value:data)if(!std::isfinite(value))throw std::runtime_error("nonfinite accepted gas budget state");
    return data;
}
inline long double sum(const std::vector<double>& data,std::size_t offset,std::size_t count,const std::vector<double>* volumes=nullptr)
{
    long double total=0;
    for(std::size_t c=0;c<count;++c)total+=static_cast<long double>(data[offset+c])*(volumes?static_cast<long double>((*volumes)[c]):1.L);
    return total;
}
template<class State>std::vector<long double> inventory(State*s)
{
    auto& h=*s->gasBudgetHost;const std::size_t n=s->nCells;
    const auto mass=read(s,s->rho,n),energy=read(s,s->rhoE,n),species=read(s,s->gasSpecies.rho,n*h.speciesCount);
    std::vector<long double> result(h.speciesCount+2);result[0]=sum(mass,0,n,&h.volumes);result[1]=sum(energy,0,n,&h.volumes);
    for(int k=0;k<h.speciesCount;++k)result[k+2]=sum(species,k*n,n,&h.volumes);
    return result;
}
inline std::vector<long double> elements(const GasBudgetHostLedger&h,const std::vector<long double>& species)
{
    std::vector<long double> result(h.elementCount);
    for(int e=0;e<h.elementCount;++e)for(int k=0;k<h.speciesCount;++k)result[e]+=species[k]*h.elementMassFactors[k*h.elementCount+e];
    return result;
}
}

template<class State>void releaseGasBudget(State*s)
{
    if constexpr(GasBudgetHostCapability<State>::value)
    {
        releaseSharedGasPointer(s->gasBudgetAudit.transport);releaseSharedGasPointer(s->gasBudgetAudit.initialTransport);
        releaseSharedGasPointer(s->gasBudgetAudit.chemistrySpecies);s->gasBudgetAudit={};
        delete s->gasBudgetHost;s->gasBudgetHost=nullptr;
    }
}
template<class State,class Model>int configureGasBudget(State*s,const Model&model)
{
    if constexpr(GasBudgetHostCapability<State>::value)
    {
        const char* path=std::getenv("UGKP_GAS_BUDGET_OUTPUT");if(!path||!*path)return 0;
        if(path[0]!='/' || s->hostGravityActive || s->particleCapacity || s->nBoundarySources
            || s->gasSpecies.mode==ugkwp::GasMode::SingleLegacy || s->gasBudgetHost)
        {setLastErrorText("gas budget requires a fresh absolute path and fixed pure mixture gas without gravity/particles");return 1;}
        try
        {
            auto* h=new GasBudgetHostLedger;s->gasBudgetHost=h;
            h->output=std::fopen(path,"wx");if(!h->output)throw std::runtime_error("gas budget output must be a new writable file");
            h->speciesCount=int(model.speciesNames.size());h->elementCount=int(model.elementNames.size());
            h->cumulativeBoundary.resize(h->speciesCount+2);h->cumulativeChemistry.resize(h->speciesCount);
            h->elementMassFactors=model.elementComposition;
            for(int k=0;k<h->speciesCount;++k)for(int e=0;e<h->elementCount;++e)h->elementMassFactors[k*h->elementCount+e]/=model.species[k].molarMass;
            std::ostringstream id;id<<"\"species\":"<<gasBudgetDetail::names(model.speciesNames)<<",\"elements\":"<<gasBudgetDetail::names(model.elementNames)
                <<",\"species_order_hash\":"<<gasBudgetDetail::quoted(std::to_string(s->gasSpecies.thermo.speciesOrderHash))
                <<",\"thermo_hash\":"<<gasBudgetDetail::quoted(std::to_string(s->gasSpecies.thermo.thermoHash))
                <<",\"mechanism_hash\":"<<gasBudgetDetail::quoted(std::to_string(s->gasSpecies.mechanism.mechanismHash))
                <<",\"element_mass_factors_mol_per_kg\":"<<gasBudgetDetail::array(h->elementMassFactors);h->identity=id.str();
            auto&a=s->gasBudgetAudit;const std::size_t n=s->nCells,channels=h->speciesCount+2;
            if(allocateSharedGasZero(a.transport,n*channels)||allocateSharedGasZero(a.initialTransport,n*channels)
                ||allocateSharedGasZero(a.chemistrySpecies,n*h->speciesCount))throw std::runtime_error("gas budget allocation failed");
            a.speciesCount=h->speciesCount;a.enabled=true;h->deviceBytes=sizeof(double)*n*(2*channels+h->speciesCount);
            if(cudaMemcpy(reinterpret_cast<char*>(s->deviceState)+offsetof(State,gasBudgetAudit),&a,sizeof(a),cudaMemcpyHostToDevice)!=cudaSuccess)
                throw std::runtime_error("gas budget device view upload failed");
        }
        catch(const std::exception&e){setLastErrorText(e.what());if(s->gasBudgetAudit.enabled)s->gasModelPoisoned=true;releaseGasBudget(s);return 1;}
    }
    return 0;
}
// API guards reject unsupported opted-in paths before any state mutation.
// When disabled they perform no allocations, transfers, launches or validation.
template<class State>int checkGasBudgetAdvanceScope(State*s,bool pureGasOnly)
{
    if constexpr(GasBudgetHostCapability<State>::value)
    {
        const char* path=std::getenv("UGKP_GAS_BUDGET_OUTPUT");
        if(!s->gasBudgetAudit.enabled&&(!path||!*path))return 0;
        if(!pureGasOnly||s->gasSpecies.mode==ugkwp::GasMode::SingleLegacy||!s->gasBudgetAudit.enabled)
        {setLastErrorText("gas budget requires configured pure mixture gas requested-interval advance");return 1;}
    }
    return 0;
}
template<class State>int checkGasBudgetMeshUpload(State*s)
{
    if constexpr(GasBudgetHostCapability<State>::value)
        if(s->gasBudgetAudit.enabled&&s->gasBudgetHost->initialized)
        {setLastErrorText("fixed-grid gas budget segment forbids subsequent mesh upload");return 1;}
    return 0;
}
template<class State>int prepareGasBudgetRequest(State*s,double dt,double time)
{
    if constexpr(GasBudgetHostCapability<State>::value)if(s->gasBudgetAudit.enabled)
    {
        auto&h=*s->gasBudgetHost;const auto start=std::chrono::steady_clock::now();
        try
        {
            if(h.requestActive||!std::isfinite(time)||!std::isfinite(dt)||time<0||dt<=0||!std::isfinite(time+dt)||time+dt==time)
                throw std::runtime_error("invalid gas budget requested interval");
            if(s->hostGravityActive||s->particleCapacity||s->nBoundarySources)
                throw std::runtime_error("gas budget source scope changed after configuration");
            if(!h.initialized)
            {
                h.volumes=gasBudgetDetail::read(s,s->V,s->nCells);
                for(double volume:h.volumes)if(volume<=0)throw std::runtime_error("nonpositive gas budget volume");
                h.origin=gasBudgetDetail::inventory(s);h.originTime=h.acceptedTime=time;
                std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":1,\"record\":\"NEW_SEGMENT\",\"continuity\":\"UNVERIFIED_NEW_SEGMENT\",\"accepted_initial_time\":"<<time
                    <<",\"initial_inventory\":"<<gasBudgetDetail::array(h.origin)<<','<<h.identity
                    <<",\"channels\":\"mass_kg,formation_inclusive_total_energy_J,species_mass_kg\",\"source_scope\":\"fixed_grid_no_gravity_no_particles; chemistry changes species only; boundary mechanical work included in total-energy flux; no separate body-work source; GCL=0\"}";
                gasBudgetDetail::append(h,out.str());h.initialized=true;
            }
            if(time!=h.acceptedTime)throw std::runtime_error("gas budget time does not follow last accepted interval");
            h.requestedEnd=time+dt;h.requestActive=true;
        }
        catch(const std::exception&e){setLastErrorText(e.what());return 1;}
        h.hostSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
    return 0;
}
template<class State>int beginGasBudgetInterval(State*s)
{
    if constexpr(GasBudgetHostCapability<State>::value)if(s->gasBudgetAudit.enabled)
    {
        if(!s->gasBudgetHost->requestActive){setLastErrorText("gas budget interval not prepared");return 1;}
        auto&a=s->gasBudgetAudit;const std::size_t n=s->nCells,channels=a.speciesCount+2;
        if(cudaMemset(a.transport,0,n*channels*sizeof(double))!=cudaSuccess
            ||cudaMemset(a.initialTransport,0,n*channels*sizeof(double))!=cudaSuccess
            ||cudaMemset(a.chemistrySpecies,0,n*a.speciesCount*sizeof(double))!=cudaSuccess)return 1;
    }
    return 0;
}
template<class State>void cancelGasBudgetInterval(State*s)
{if constexpr(GasBudgetHostCapability<State>::value)if(s->gasBudgetAudit.enabled)s->gasBudgetHost->requestActive=false;}
template<class State>int commitGasBudgetInterval(State*s)
{
    if constexpr(GasBudgetHostCapability<State>::value)if(s->gasBudgetAudit.enabled)
    {
        auto&h=*s->gasBudgetHost;const auto start=std::chrono::steady_clock::now();
        try
        {
            if(!h.requestActive)throw std::runtime_error("gas budget interval was not pending");
            const auto current=gasBudgetDetail::inventory(s);
            const auto transport=gasBudgetDetail::read(s,s->gasBudgetAudit.transport,std::size_t(s->nCells)*(h.speciesCount+2));
            const auto chemistry=gasBudgetDetail::read(s,s->gasBudgetAudit.chemistrySpecies,std::size_t(s->nCells)*h.speciesCount);
            auto boundary=h.cumulativeBoundary,chemical=h.cumulativeChemistry;
            for(int k=0;k<h.speciesCount+2;++k)boundary[k]+=gasBudgetDetail::sum(transport,std::size_t(k)*s->nCells,s->nCells);
            for(int k=0;k<h.speciesCount;++k)chemical[k]+=gasBudgetDetail::sum(chemistry,std::size_t(k)*s->nCells,s->nCells);
            std::vector<long double> residual(h.speciesCount+2),elementSpecies(h.speciesCount);
            for(int k=0;k<h.speciesCount+2;++k)residual[k]=static_cast<long double>(current[k])-h.origin[k]-boundary[k]-(k>=2?chemical[k-2]:0.L);
            for(int k=0;k<h.speciesCount;++k)elementSpecies[k]=static_cast<long double>(current[k+2])-h.origin[k+2]-boundary[k+2];
            const auto elementResidual=gasBudgetDetail::elements(h,elementSpecies),chemicalElements=gasBudgetDetail::elements(h,chemical);
            h.hostSeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::ostringstream out;out<<std::setprecision(17)<<"{\"schema\":1,\"record\":\"ACCEPTED_INTERVAL\",\"sequence\":"<<h.sequence+1<<",\"interval_start\":"<<h.acceptedTime
                <<",\"accepted_time\":"<<h.requestedEnd<<",\"inventory\":"<<gasBudgetDetail::array(current)
                <<",\"cumulative_boundary_transport\":"<<gasBudgetDetail::array(boundary)<<",\"cumulative_chemistry_species\":"<<gasBudgetDetail::array(chemical)
                <<",\"constraint_or_roundoff_residual\":"<<gasBudgetDetail::array(residual)<<",\"element_conservation_residual_mol\":"<<gasBudgetDetail::array(elementResidual)
                <<",\"chemistry_element_drift_mol\":"<<gasBudgetDetail::array(chemicalElements)<<",\"element_status\":\""<<(h.elementCount?"DECLARED":"UNAVAILABLE_NO_ELEMENT_BASIS")
                <<"\",\"audit_device_bytes\":"<<3*h.deviceBytes<<",\"audit_host_read_bytes\":"<<h.hostReadBytes<<",\"audit_extra_kernel_launches\":0,\"audit_host_seconds\":"<<h.hostSeconds
                <<",\"cost_scope\":\"host boundary inventory reads/reductions; device bytes include both rollback snapshots; allocation/reset/snapshot/fused-kernel/file-IO time excluded; not complete overhead\"}";
            gasBudgetDetail::append(h,out.str());h.cumulativeBoundary=std::move(boundary);h.cumulativeChemistry=std::move(chemical);
            h.acceptedTime=h.requestedEnd;++h.sequence;h.requestActive=false;
        }
        catch(const std::exception&e){setLastErrorText(e.what());return 1;}
    }
    return 0;
}
