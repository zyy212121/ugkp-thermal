#pragma once
#include "IOdictionary.H"
#include "OSspecific.H"
#include "Switch.H"
#include "Time.H"

namespace ugkp { namespace solverControls {

inline Foam::IOobject regionalInputObject(const char* name,const Foam::Time& time,
    const Foam::objectRegistry& region,Foam::IOobject::readOption option)
{
    Foam::IOobject local(name,time.constant(),region,option,Foam::IOobject::NO_WRITE,false);
    Foam::IOobject global(name,time.constant(),time,option,Foam::IOobject::NO_WRITE,false);
    return Foam::isFile(local.objectPath(false)) || !Foam::isFile(global.objectPath(false)) ? local : global;
}

inline Foam::word rasModel(const Foam::dictionary& ras,const Foam::word& fallback=Foam::word::null)
{
    if (!ras.lookupOrDefault<Foam::Switch>("turbulence", true)) return "laminar";
    return fallback.empty() ? Foam::word(ras.lookup("model")) : ras.lookupOrDefault<Foam::word>("model",fallback);
}

struct FoamSstCoefficients {
    Foam::scalar sigmaK1=0.85,sigmaK2=1.0,sigmaOmega1=0.5,sigmaOmega2=0.856;
    Foam::scalar beta1=0.075,beta2=0.0828,betaStar=0.09,gamma1=5.0/9.0,gamma2=0.44;
    Foam::scalar a1=0.31,productionLimit=10.0,minimumK=1e-12,minimumOmega=1e-6;
};

template<class Coefficients>
inline void readSstCoefficients(const Foam::dictionary& input,Coefficients& sst)
{
    auto read = [&](const char* key,auto& value) {
        value=input.lookupOrDefault<Foam::scalar>(key,value);
    };
    read("alphaK1",sst.sigmaK1);
    read("alphaK2",sst.sigmaK2);
    read("alphaOmega1",sst.sigmaOmega1);
    read("alphaOmega2",sst.sigmaOmega2);
    read("beta1",sst.beta1);
    read("beta2",sst.beta2);
    read("betaStar",sst.betaStar);
    read("gamma1",sst.gamma1);
    read("gamma2",sst.gamma2);
    read("a1",sst.a1);
    read("c1",sst.productionLimit);
    read("kMin",sst.minimumK);
    read("omegaMin",sst.minimumOmega);
}

inline Foam::dictionary readMomentumTransport(const Foam::Time& time,const Foam::objectRegistry& region,
    const Foam::dictionary& fluid,const Foam::dictionary* coupled=nullptr)
{
    const auto input=regionalInputObject("momentumTransport",time,region,Foam::IOobject::MUST_READ);
    Foam::dictionary result;
    Foam::string source("default");
    if (Foam::isFile(input.objectPath(false))) {
        result=Foam::IOdictionary(input);
        source=input.objectPath(false);
    } else if (coupled && coupled->found("sst")) {
        Foam::dictionary ras;
        ras.set("model",Foam::word("kOmegaSST"));
        ras.set("turbulence",Foam::Switch(true));
        ras.set("kOmegaSSTCoeffs",coupled->subDict("sst"));
        result.set("simulationType",Foam::word(coupled->lookupOrDefault<Foam::Switch>("enableSst",true) ? "RAS" : "laminar"));
        result.set("RAS",ras);
        source="chmtProperties/sst";
    } else if (fluid.found("turbulence")) {
        result=fluid.subDict("turbulence");
        source="fluidProperties/turbulence";
    }
    Foam::Info<<"MOMENTUM_INPUT region="<<region.name()<<" source="<<source
        <<" simulationType="<<result.lookupOrDefault<Foam::word>("simulationType","laminar")<<Foam::nl;
    return result;
}
}}
