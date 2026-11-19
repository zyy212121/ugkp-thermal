#pragma once
#include "configuration/InputValues.H"
#include "FoamMomentumTransportInput.H"
#include <cmath>
#include <cstdint>
namespace chmt { namespace io {


using ugkp::solverControls::regionalInputObject;

inline Foam::dictionary sharedParticleInput(const Foam::dictionary& input) {
    Foam::dictionary out(input);
    if(input.found("gpuResidentRandomSeed")){
        Foam::token value(input.lookup("gpuResidentRandomSeed"));
        if(value.isLabel()||value.isScalar()){
            const double number=value.isLabel()?double(value.labelToken()):double(value.scalarToken());
            require(std::isfinite(number)&&number>=0&&number<18446744073709551616.0&&std::floor(number)==number,
                "gpuResidentRandomSeed must contain a nonnegative integer within uint64 range");
            out.set("gpuResidentRandomSeed",Foam::string(std::to_string(static_cast<std::uint64_t>(number))));
        }
    }
    const auto number=[&](const char* current,const char* previous){
        if(!out.found(current)&&input.found(previous))out.set(current,readNumber(input,previous));
    };
    const auto boolean=[&](const char* current,const char* previous){
        if(!out.found(current)&&input.found(previous))out.set(current,Foam::Switch(flag(input,previous)));
    };
    number("TpMin","particleTMin");number("TpMax","particleTMax");
    number("collisionalRestitution","gpuResidentCollisionalRestitution");
    number("pressureKickFraction","gpuResidentPressureKickFraction");
    number("packingFraction","gpuResidentPackingFraction");
    boolean("solveParticleTemperature","particleTemperatureTransport");
    boolean("collisionalPressure","gpuResidentCollisionalPressure");
    boolean("packingPressure","gpuResidentJammingPressure");
    return out;
}

inline Foam::dictionary coupledSurfaceInput(const Foam::dictionary& coupled,
    const Foam::dictionary& shared) {
    Foam::dictionary surface(coupled.subOrEmptyDict("surface"));
    if(shared.found("coupledInterfaces")){
        const Foam::PtrList<Foam::dictionary> pairs(shared.lookup("coupledInterfaces"));
        Foam::wordList solid(pairs.size()),gas(pairs.size());
        forAll(pairs,i){solid[i]=Foam::word(text(pairs[i],"solidPatch"));gas[i]=Foam::word(text(pairs[i],"fluidPatch"));}
        surface.set("patches",solid);surface.set("gasPatches",gas);
    }else{
        if(!surface.found("patches")&&!surface.found("patch"))surface.set("patch",Foam::word(text(shared,"solidPatch")));
        if(!surface.found("gasPatches")&&!surface.found("gasPatch"))surface.set("gasPatch",Foam::word(text(shared,"fluidPatch")));
    }
    return surface;
}
}}
