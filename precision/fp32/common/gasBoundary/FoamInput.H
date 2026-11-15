#pragma once
#include "Definition.H"
#include "fvCFD.H"
#include "emptyPolyPatch.H"
#include "wedgePolyPatch.H"
#include "symmetryPolyPatch.H"
#include "symmetryPlanePolyPatch.H"
#include "wallPolyPatch.H"
#include "cyclicPolyPatch.H"
#include "inletOutletFvPatchField.H"
#include "waveTransmissiveFvPatchField.H"
#include "OStringStream.H"
#include "IStringStream.H"

namespace ugkp { namespace boundary {
namespace foam {
using namespace Foam;
template<class Patch> dictionary configuration(const Patch& field) {
    OStringStream os;field.write(os);IStringStream is(os.str());return dictionary(is);
}
inline int scalarMode(const word& type,bool fixes,bool strict) {
    if(type=="totalPressure"||type=="totalTemperature")return int(Scalar::Total);
    if(type=="fixedValue"||type=="uniformFixedValue")return int(Scalar::Fixed);
    if(type=="zeroGradient"||type=="calculated"||type=="waveTransmissive")return 0;
    if(!strict)return fixes?1:0;
    throw std::runtime_error("unsupported resident scalar boundary: "+std::string(type));
}
template<class Type> void function(Definition& out,const dictionary& dict,const word& key,
    int firstComponent,std::vector<Face>& faces,int start,int count) {
    if(!dict.found(key))return;
    const dictionary* sub=dict.isDict(key)?&dict.subDict(key):nullptr;
    word kind;ITstream* input=nullptr;
    if(sub)kind=word(sub->lookup("type"));
    else {input=&dict.lookup(key);*input>>kind;}
    Function fn;fn.begin=int(out.terms.size());
    Type constant=pTraits<Type>::zero;
    List<Tuple2<scalar,Type>> rows;
    List<Tuple2<scalar,scalar>> powers;
    if(kind=="constant") {
        if(sub)sub->lookup("value")>>constant;else *input>>constant;
    } else if(kind=="table"||kind=="tableFile") {
        fn.kind=FunctionKind::Table;
        if(kind=="tableFile") {
            if(!sub)throw std::runtime_error("tableFile requires a function dictionary");
            fileName path(sub->lookup("file"));path.expand();IFstream file(path);
            if(!file.good())throw std::runtime_error("cannot read boundary table "+std::string(path));
            file>>rows;
        } else if(sub)sub->lookup("values")>>rows;else *input>>rows;
        if(sub) {
            const word interpolation=sub->lookupOrDefault<word>("interpolationScheme","linear");
            if(interpolation!="linear")throw std::runtime_error("resident table supports linear interpolation");
            const word bounds=sub->lookupOrDefault<word>("outOfBounds","clamp");
            if(bounds=="repeat")fn.bounds=Bounds::Repeat;
            else if(bounds!="clamp"&&bounds!="warn")throw std::runtime_error("resident table supports clamp or repeat bounds");
        }
        fn.count=rows.size();
    } else if(kind=="polynomial") {
        if(pTraits<Type>::nComponents!=1)throw std::runtime_error("resident polynomial requires scalar coefficients");
        fn.kind=FunctionKind::Polynomial;
        if(sub)sub->lookup("coeffs")>>powers;else *input>>powers;
        fn.count=powers.size();
    } else throw std::runtime_error("unsupported resident function: "+std::string(kind));
    for(int c=0;c<pTraits<Type>::nComponents;++c) {
        fn.begin=int(out.terms.size());fn.constant=GpuReal(Foam::component(constant,c));
        for(const auto& row:rows)out.terms.push_back(Term::table(row.first(),GpuReal(Foam::component(row.second(),c))));
        for(const auto& term:powers)out.terms.push_back(Term::polynomial(GpuReal(term.first()),GpuReal(term.second())));
        const int id=int(out.functions.size());out.functions.push_back(fn);
        for(int i=0;i<count;++i)faces[start+i].function[firstComponent+c]=id;
    }
}
// EpsilonS previously supplied an evaluated startup snapshot. Keep that
// behavior for Function1 programs outside the existing resident evaluator,
// rather than expanding its rejection policy to otherwise usable old cases.
inline bool residentEpsilonFunctionSupported(const dictionary& dict,const word& key) {
    if(!dict.found(key))return false;
    dictionary input(dict);
    const dictionary* sub=input.isDict(key)?&input.subDict(key):nullptr;
    word kind;
    if(sub) {
        if(!sub->found("type"))return false;
        kind=word(sub->lookup("type"));
    } else input.lookup(key)>>kind;
    if(kind=="constant"||kind=="polynomial")return true;
    if(kind!="table"&&kind!="tableFile")return false;
    if(kind=="tableFile"&&!sub)return false;
    if(sub) {
        const word interpolation=sub->lookupOrDefault<word>("interpolationScheme","linear");
        const word bounds=sub->lookupOrDefault<word>("outOfBounds","clamp");
        if(interpolation!="linear"||(bounds!="clamp"&&bounds!="warn"&&bounds!="repeat"))return false;
    }
    return true;
}
inline scalar epsilonStartupSnapshot(const scalar value,const int mode) {
    // Owner-following modes never use a startup value. For legacy snapshots,
    // nonfinite source rates were already suppressed by finiteOr(rate,0).
    const GpuReal stored=GpuReal(value);
    return mode!=0&&std::isfinite(stored)?scalar(stored):scalar(0);
}
// Inspect only the newly parsed epsilonS program. Structural/parser errors
// still reach their existing diagnostics; nonfinite legacy inputs keep the
// evaluated startup snapshot instead of extending gas validation to epsilonS.
inline bool retainFiniteEpsilonProgram(Definition& out,std::size_t functions,
    std::size_t terms,int start,int count) {
    bool finite=true;
    for(std::size_t i=functions;i<out.functions.size();++i) {
        const auto& f=out.functions[i];
        finite=finite&&std::isfinite(f.constant);
        for(int j=0;j<f.count;++j) {
            const auto& t=out.terms[f.begin+j];
            if((f.kind==FunctionKind::Polynomial?!std::isfinite(t.coefficient):!std::isfinite(t.time))||!std::isfinite(t.value))finite=false;
        }
    }
    if(!finite) {
        out.functions.resize(functions);out.terms.resize(terms);
        for(int i=0;i<count;++i)out.faces[start+i].function[SolidFraction]=-1;
    }
    return finite;
}
inline void readComposition(const fvMesh& mesh,const PtrList<volScalarField>& Y,Definition& d) {
    forAll(mesh.boundaryMesh(),patch) {
        if(Y.empty())continue;
        const auto& pp=mesh.boundaryMesh()[patch];
        const bool fixed=Y[0].boundaryField()[patch].fixesValue();
        forAll(Y,k)if(Y[k].boundaryField()[patch].fixesValue()!=fixed)
            throw std::runtime_error("species must share one boundary composition policy on "+std::string(pp.name()));
        for(int f=0;f<pp.size();++f)if(d.faces[pp.start()+f].compositionMode<0)d.faces[pp.start()+f].compositionMode=fixed?1:0;
    }
}
template<class Field> void initialiseStaticPatches(Field& field) {
    forAll(field.boundaryField(),patch) {
        const auto type=field.boundaryField()[patch].type();
        if(type=="totalPressure"||type=="totalTemperature"||type=="pressureInletOutletVelocity"
            ||type=="uniformFixedValue"||type=="inletOutlet"||type=="waveTransmissive")continue;
        field.boundaryFieldRef()[patch].evaluate();
    }
}
inline Definition read(const fvMesh& mesh,const volScalarField& rho,const volVectorField& U,
    const volScalarField& p,const volScalarField& T,bool strict=true,
    const std::vector<Kind>* prescribedKinds=nullptr) {
    Definition out;out.faces.resize(mesh.nFaces());
    forAll(mesh.boundaryMesh(),patch) {
        const auto& pp=mesh.boundaryMesh()[patch];const int start=pp.start(),size=pp.size();
        int geometry=0;
        if(isA<emptyPolyPatch>(pp))geometry=4;
        else if(isA<cyclicPolyPatch>(pp))geometry=5;
        else if(pp.coupled())throw std::runtime_error("resident boundary requires serial or translational cyclic geometry: "+std::string(pp.name()));
        else if(isA<wedgePolyPatch>(pp)||isA<symmetryPolyPatch>(pp)||isA<symmetryPlanePolyPatch>(pp))geometry=1;
        else if(isA<wallPolyPatch>(pp))geometry=2;
        if(geometry==4) {for(int i=0;i<size;++i)out.faces[start+i].geometry=4;continue;}
        const auto& rp=rho.boundaryField()[patch];const auto& up=U.boundaryField()[patch];
        const auto& ppv=p.boundaryField()[patch];const auto& tp=T.boundaryField()[patch];
        if(rp.size()!=size||up.size()!=size||ppv.size()!=size||tp.size()!=size)
            throw std::runtime_error("resident boundary field size mismatch");
        const bool check=strict&&(geometry==0||geometry==2);
        const auto rd=configuration(rp),ud=configuration(up),pd=configuration(ppv),td=configuration(tp);
        int rm=scalarMode(rp.type(),rp.fixesValue(),check),pm=scalarMode(ppv.type(),ppv.fixesValue(),check),tm=scalarMode(tp.type(),tp.fixesValue(),check);
        int um=0;
        const bool inletOutlet=isA<inletOutletFvPatchField<vector>>(up);
        if(up.type()=="pressureInletOutletVelocity")um=int(Velocity::PressureInletOutlet);
        else if(inletOutlet)um=int(Velocity::InletOutlet);
        else if(up.type()=="fixedValue"||up.type()=="uniformFixedValue"||up.type()=="noSlip")um=int(Velocity::Fixed);
        else if(up.type()!="zeroGradient"&&check)throw std::runtime_error("unsupported resident velocity boundary: "+std::string(up.type()));
        else if(!check&&up.fixesValue())um=1;
        const bool wave=isA<waveTransmissiveFvPatchField<scalar>>(ppv);
        scalarField p0,T0;
        if(pm==2)p0=scalarField("p0",pd,size);
        if(tm==2)T0=scalarField("T0",td,size);
        for(int i=0;i<size;++i) {
            auto& f=out.faces[start+i];f.geometry=geometry;
            f.rhoMode=rm;f.uMode=um;f.pMode=pm;f.tMode=tm;
            f.value[Density]=rp[i];f.value[Pressure]=pm==2?p0[i]:ppv[i];f.value[Temperature]=tm==2?T0[i]:tp[i];
            vector velocity=up[i];
            if(inletOutlet)velocity=refCast<const inletOutletFvPatchField<vector>>(up).refValue()[i];
            if(um==4)velocity=vector::zero; // optional tangential program is projected on device
            f.value[Ux]=velocity.x();f.value[Uy]=velocity.y();f.value[Uz]=velocity.z();
            if(pm==2) {
                f.pressureGamma=pd.lookupOrDefault<scalar>("gamma",1);
                f.compressiblePressure=pd.lookupOrDefault<word>("psi","none")!="none";
            }
            if(tm==2)f.temperatureGamma=readScalar(td.lookup("gamma"));
            if(wave) {
                const auto& w=refCast<const waveTransmissiveFvPatchField<scalar>>(ppv);
                f.wave=1;f.waveGamma=w.gamma();f.waveFieldInf=w.fieldInf();f.waveLInf=w.lInf();
            }
            if(prescribedKinds&&pm!=2&&tm!=2&&um!=4
                &&rp.type()!="uniformFixedValue"&&up.type()!="uniformFixedValue"
                &&ppv.type()!="uniformFixedValue"&&tp.type()!="uniformFixedValue") {
                const Kind kind=prescribedKinds->at(start+i);
                f=prescribed(kind,tm==1,rp[i],up[i].x(),up[i].y(),up[i].z(),ppv[i],tp[i]);
            }
        }
        if(rp.type()=="uniformFixedValue")function<scalar>(out,rd,"uniformValue",Density,out.faces,start,size);
        if(up.type()=="uniformFixedValue")function<vector>(out,ud,"uniformValue",Ux,out.faces,start,size);
        if(ppv.type()=="uniformFixedValue")function<scalar>(out,pd,"uniformValue",Pressure,out.faces,start,size);
        if(tp.type()=="uniformFixedValue")function<scalar>(out,td,"uniformValue",Temperature,out.faces,start,size);
        if(um==4)function<vector>(out,ud,"tangentialVelocity",Ux,out.faces,start,size);
    }

    // Capture epsilonS policy and table once. Extrapolated values are read from
    // the current device owner cell at injection, never from this startup field.
    if(mesh.foundObject<volScalarField>("epsilonS")) {
        const auto& eps=mesh.lookupObject<volScalarField>("epsilonS");
        forAll(mesh.boundaryMesh(),patch) {
            const auto& pp=mesh.boundaryMesh()[patch];
            // Empty poly patches own geometry but their fvPatchField has no
            // values. Match the gas boundary loop and never index that storage.
            if(isA<emptyPolyPatch>(pp))continue;
            const auto& field=eps.boundaryField()[patch];
            const auto config=configuration(field);
            const int mode=field.fixesValue()?1:0;
            for(int i=0;i<pp.size();++i) {
                auto& face=out.faces[pp.start()+i];
                face.solidMode=mode;face.value[SolidFraction]=epsilonStartupSnapshot(field[i],mode);
            }
            if(field.type()=="uniformFixedValue"
                &&residentEpsilonFunctionSupported(config,"uniformValue")) {
                // No catch-all: supported table syntax/data errors retain their
                // real diagnostics instead of silently becoming a snapshot.
                const auto functions=out.functions.size(),terms=out.terms.size();
                function<scalar>(out,config,"uniformValue",SolidFraction,out.faces,pp.start(),pp.size());
                if(retainFiniteEpsilonProgram(out,functions,terms,pp.start(),pp.size())) {
                    for(int i=0;i<pp.size();++i)
                        if(out.faces[pp.start()+i].function[SolidFraction]>=0)
                            out.faces[pp.start()+i].value[SolidFraction]=0;
                } else {
                    WarningInFunction
                        << "epsilonS patch " << pp.name()
                        << ": nonfinite resident uniformValue data; retaining the legacy startup snapshot "
                        << "(not a GPU time-varying program)." << nl;
                }
            } else if(field.type()=="uniformFixedValue") {
                WarningInFunction
                    << "epsilonS patch " << pp.name()
                    << ": this uniformValue program is outside the resident evaluator; "
                    << "retaining the legacy startup snapshot (not a GPU time-varying program)."
                    << nl;
            }
        }
    }
    validate(out);return out;
}
}}
}
