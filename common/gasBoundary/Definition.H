#pragma once
#include <cmath>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <string>

#if defined(__CUDACC__)
#define UGKP_RESIDENT_BOUNDARY_HD __host__ __device__
#else
#define UGKP_RESIDENT_BOUNDARY_HD
#endif

namespace ugkp { namespace boundary {
// Physical mesh roles are independent of scalar/vector boundary operators.
// Values are retained for existing mesh checkpoints.
enum class Kind { Internal,Periodic,Slip,NoSlip,Inlet,Outlet,Interface,Empty };
enum class Geometry : int { Standard=0,Symmetry=1,Wall=2,Processor=3,Empty=4,Cyclic=5,
    standard=Standard,symmetry=Symmetry,wall=Wall,processor=Processor,empty=Empty,cyclic=Cyclic };
enum class Scalar : int { Extrapolated=0,Fixed=1,Total=2 };
enum class Velocity : int { unsupported=-1,Extrapolated=0,Fixed=1,InletOutlet=2,CharacteristicTotal=3,PressureInletOutlet=4,
    zeroGradient=Extrapolated,fixedValue=Fixed,inletOutlet=InletOutlet };
enum class InputScalar : int { unsupported=-1,zeroGradient=0,fixedValue=1,waveTransmissive=2,calculated=3,total=4 };
inline InputScalar scalarOperator(const std::string& name,bool fixesValue,bool wave) {
    if(name=="totalPressure"||name=="totalTemperature")return InputScalar::total;
    if(wave)return InputScalar::waveTransmissive;
    if(fixesValue)return InputScalar::fixedValue;
    if(name=="zeroGradient")return InputScalar::zeroGradient;
    if(name=="calculated")return InputScalar::calculated;
    return InputScalar::unsupported;
}
inline Velocity velocityOperator(const std::string& name,bool fixesValue,bool inletOutlet) {
    if(name=="pressureInletOutletVelocity")return Velocity::PressureInletOutlet;
    if(inletOutlet)return Velocity::InletOutlet;
    if(fixesValue)return Velocity::Fixed;
    if(name=="zeroGradient")return Velocity::Extrapolated;
    return Velocity::unsupported;
}
enum class FunctionKind : int { Constant,Table,Polynomial };
enum class Bounds : int { Clamp,Repeat };
enum Component : int { Density,Ux,Uy,Uz,Pressure,Temperature,Components };
struct Term { double x=0,y=0; };
struct Function {
    FunctionKind kind=FunctionKind::Constant;
    Bounds bounds=Bounds::Clamp;
    int begin=0,count=0;
    double constant=0;
};
struct Face {
    int geometry=0,rhoMode=0,uMode=0,pMode=0,tMode=0;
    int wave=0,compositionMode=-1;
    double waveGamma=0,waveFieldInf=0,waveLInf=-1e30;
    double value[Components]{};
    // Explicit gamma and pressure branch preserve the selected total-condition law.
    double pressureGamma=1,temperatureGamma=1;
    int compressiblePressure=0;
    int function[Components]={-1,-1,-1,-1,-1,-1};
};

inline bool prescribesInflow(const Face& f) {
    return f.geometry==0&&(f.uMode==1||f.uMode==2||f.uMode==3||f.uMode==4);
}
struct View {
    const Face* faces=nullptr;
    const Function* functions=nullptr;
    const Term* terms=nullptr;
    int count=0;
};
struct Definition {
    std::vector<Face> faces;
    std::vector<Function> functions;
    std::vector<Term> terms;
};
inline bool needsCheckpoint(const Definition& d) {
    if(!d.functions.empty())return true;
    for(const auto& f:d.faces)if(f.pMode==2||f.tMode==2||f.uMode==4)return true;
    return false;
}
inline Kind physicalKind(const std::string& name) {
    if(name=="Internal")return Kind::Internal;
    if(name=="Periodic")return Kind::Periodic;
    if(name=="Slip")return Kind::Slip;
    if(name=="NoSlip")return Kind::NoSlip;
    if(name=="Inlet")return Kind::Inlet;
    if(name=="Outlet")return Kind::Outlet;
    if(name=="Interface")return Kind::Interface;
    if(name=="Empty")return Kind::Empty;
    throw std::runtime_error("unsupported physical boundary kind "+name);
}
UGKP_RESIDENT_BOUNDARY_HD inline int geometry(Kind k) {
    return k==Kind::Periodic?5:k==Kind::Slip?1:
        (k==Kind::NoSlip||k==Kind::Interface)?2:k==Kind::Empty?4:0;
}
// General prescribed-state interface, also used by programmatic mesh clients.
inline Face prescribed(Kind kind,bool fixedTemperature,double rho,double ux,double uy,
    double uz,double p,double T) {
    Face f;f.geometry=geometry(kind);
    f.compositionMode=kind==Kind::Inlet;
    f.rhoMode=kind==Kind::Inlet;f.uMode=kind==Kind::Inlet||kind==Kind::NoSlip;
    f.pMode=kind==Kind::Inlet||kind==Kind::Outlet;f.tMode=fixedTemperature;
    f.value[Density]=rho;f.value[Ux]=ux;f.value[Uy]=uy;f.value[Uz]=uz;
    f.value[Pressure]=p;f.value[Temperature]=T;return f;
}
inline void validate(const Definition& d) {
    for(const auto& f:d.functions) {
        if(f.begin<0||f.count<0||std::size_t(f.begin)+f.count>d.terms.size())
            throw std::runtime_error("boundary function term range is invalid");
        if(f.kind==FunctionKind::Table&&f.count<1)
            throw std::runtime_error("boundary table is empty");
        if(!std::isfinite(f.constant))throw std::runtime_error("nonfinite boundary function");
        for(int j=0;j<f.count;++j) {
            const auto& t=d.terms[f.begin+j];
            if(!std::isfinite(t.x)||!std::isfinite(t.y))throw std::runtime_error("nonfinite boundary term");
            if(f.kind==FunctionKind::Table&&j&&t.x<=d.terms[f.begin+j-1].x)
                throw std::runtime_error("boundary table times must increase strictly");
        }
    }
    for(const auto& f:d.faces) {
        for(int c=0;c<Components;++c) {
            if(!std::isfinite(f.value[c]))throw std::runtime_error("nonfinite boundary value");
            if(f.function[c]<-1||f.function[c]>=int(d.functions.size()))
                throw std::runtime_error("boundary function index is invalid");
        }
        if(f.tMode==int(Scalar::Total)&&(!(f.temperatureGamma>1)||!std::isfinite(f.temperatureGamma)))
            throw std::runtime_error("totalTemperature requires finite gamma > 1");
        if(f.pMode==int(Scalar::Total)&&(!(f.pressureGamma>0)||!std::isfinite(f.pressureGamma)))
            throw std::runtime_error("totalPressure requires finite positive gamma");
    }
}
UGKP_RESIDENT_BOUNDARY_HD inline double evaluate(const View& v,int index,double time,double constant) {
    if(index<0)return constant;
    const auto& f=v.functions[index];
    if(f.kind==FunctionKind::Constant)return f.constant;
    const auto* a=v.terms+f.begin;
    if(f.kind==FunctionKind::Polynomial) {
        double value=0;for(int i=0;i<f.count;++i)value+=a[i].x*::pow(time,a[i].y);
        return value;
    }
    if(f.bounds==Bounds::Repeat&&f.count>1) {
        const double span=a[f.count-1].x-a[0].x;
        if(time<a[0].x||time>a[f.count-1].x) {
            time=a[0].x+::fmod(time-a[0].x,span);if(time<a[0].x)time+=span;
        }
    }
    if(time<=a[0].x)return a[0].y;
    if(time>=a[f.count-1].x)return a[f.count-1].y;
    int lo=0,hi=f.count-1;
    while(lo+1<hi){const int mid=(lo+hi)/2;if(a[mid].x<=time)lo=mid;else hi=mid;}
    return a[lo].y+(time-a[lo].x)/(a[hi].x-a[lo].x)*(a[hi].y-a[lo].y);
}
template<class...>using Void=void;
template<class State,class=void>struct Capability:std::false_type{};
template<class State>struct Capability<State,Void<decltype(std::declval<State>().gasBoundaryProgram)>>:std::true_type{};
}}
