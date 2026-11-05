#pragma once
// Included after primitive/EOS helpers, within the application's kernel namespace.
template<class State> __device__ bool evaluateResidentBoundary(State& s,int f,GPU_OPERATOR_TIME time)
{
    if constexpr (ugkp::boundary::Capability<State>::value) {
        const auto& program=s.gasBoundaryProgram;
        if(!program.faces||f>=program.count)return false;
        const auto& b=program.faces[f];
        const bool total=b.pMode==2||b.tMode==2||b.uMode==4;
        bool varying=false;
        for(int c=0;c<ugkp::boundary::Components;++c)varying=varying||b.function[c]>=0;
        if(!varying&&!total)return false;
        double value[ugkp::boundary::Components];
        for(int c=0;c<ugkp::boundary::Components;++c)
            value[c]=ugkp::boundary::evaluate(program,b.function[c],double(time),b.value[c]);
        if(varying&&!total) {
#define SET(component,field) if(b.function[ugkp::boundary::component]>=0) {s.gasBoundary##field[f]=value[ugkp::boundary::component];s.riemannBoundary##field[f]=value[ugkp::boundary::component];}
            SET(Density,Rho);SET(Ux,Ux);SET(Uy,Uy);SET(Uz,Uz);SET(Pressure,P);SET(Temperature,T);
#undef SET
            return false;
        }
        const int own=s.faceOwner[f];const auto owner=gasCellPrimitive(s,own);
        GPU_OPERATOR_REAL R=s.Rgas;
        if constexpr(ugkwp::GasStateTraits<State>::speciesCount>0)if(ugkwp::mixtureGasActive(s)) {
            constexpr int Ns=ugkwp::GasStateTraits<State>::speciesCount;GPU_OPERATOR_REAL y[Ns];
            for(int k=0;k<Ns;++k)y[k]=s.gasSpecies.compositionBoundaryFixed[f]
                ?s.gasSpecies.boundaryMassFraction[k*s.nFaces+f]:s.gasSpecies.rho[k*s.nCells+own]/s.rho[own];
            R=ugkwp::mixtureGasConstant(y,s.gasSpecies.thermo);
        }
        GPU_OPERATOR_REAL meshNormal=0;
        if(!gasBoundaryMeshNormalSpeed(s,f,meshNormal))return true;
        const double area=s.magSf[f];ugkp::boundary::Primitive result;
        if(!ugkp::boundary::totalState(b,value,{owner.rho,owner.ux,owner.uy,owner.uz,owner.p,owner.T},
            R,s.Sfx[f]/area,s.Sfy[f]/area,s.Sfz[f]/area,meshNormal,result)) {
            using namespace ugkp::diagnostics;
            recordFatal(s.diagnostics,makeRecord(Code::NonFinite,2603,26,0,EntityKind::Face,f,-1,result.T));
            return true;
        }
#define STORE(field,member) s.gasBoundary##field[f]=result.member;s.riemannBoundary##field[f]=result.member
        STORE(Rho,rho);STORE(P,p);STORE(T,T);STORE(Ux,ux);STORE(Uy,uy);STORE(Uz,uz);
#undef STORE
        return true;
    }
    return false;
}
