#include "GpuPrecisionTypes.H"
#pragma once

  
                                                              
  
                                         
                                                                      
                                                                             
                                                                        
                                                                         
                                                             
  
                                           
  
                                                                   
                                                                             
                                                                           
                                                            
   

#include <cmath>
#include <limits>
#include "Diagnostics.H"

#if defined(__CUDACC__)
#define UGKP_WALL_HD __host__ __device__ __forceinline__
#else
#define UGKP_WALL_HD inline
#endif

namespace ugkpwall
{

struct SpaldingWallState
{
    GpuReal uTau;
    GpuReal yPlus;
    GpuReal nut;
};

struct WallSubgridTransport
{
    GpuReal dynamicViscosity;
    GpuReal thermalConductivity;
};

struct OmegaWallFunctionState
{
    GpuReal omega;
    GpuReal production;
    GpuReal yPlus;
};

struct JayatillekeWallHeatState
{
    GpuReal heatFlux;
    GpuReal temperaturePlus;
    GpuReal yPlusThermal;
    int valid;
};

UGKP_WALL_HD GpuReal maximum(const GpuReal a, const GpuReal b)
{
    return a > b ? a : b;
}

UGKP_WALL_HD GpuReal minimum(const GpuReal a, const GpuReal b)
{
    return a < b ? a : b;
}

UGKP_WALL_HD GpuReal wallYPlusLaminar
(
    const GpuReal kappa = GPU_R(0.41),
    const GpuReal E = GPU_R(9.8),
    ugkp::diagnostics::DeviceContext* diagnostics = nullptr,
    const int entity = -1
)
{
    using namespace ugkp::diagnostics;
    if(!std::isfinite(kappa)||!std::isfinite(E)||kappa<=0||E<=0){
        const GpuReal value=!std::isfinite(kappa)||kappa<=0?kappa:E;
        if(diagnostics)recordFatal(diagnostics,makeRecord(!std::isfinite(value)?Code::NonFinite:Code::InvalidPhysicalState,
            2829,32,0,EntityKind::Face,entity,-1,value));
        return std::numeric_limits<GpuReal>::quiet_NaN();
    }
    constexpr GpuReal small = GPU_R(2.2204460492503131e-16);
    GpuReal yPlus = GPU_R(11.0);
    for (int iteration = 0; iteration < 10; ++iteration)
    {
        const GpuReal argument = maximum(E*yPlus, GPU_R(1.0) + small);
        const GpuReal function = yPlus - log(argument)/kappa;
        const GpuReal derivative = GPU_R(1.0) - GPU_R(1.0)/(kappa*yPlus);
        const GpuReal updated = yPlus - function/maximum(derivative, small);
        if (updated <= small)
        {
            break;
        }
        if (fabs(updated - yPlus) <= GPU_R(1.0e-8)*maximum(yPlus, GPU_R(1.0)))
        {
            yPlus = updated;
            break;
        }
        yPlus = updated;
    }
    const GpuReal residual=yPlus-log(maximum(E*yPlus,GPU_R(1)+small))/kappa;
    if(!std::isfinite(yPlus)||!std::isfinite(residual)){
        if(diagnostics)recordFatal(diagnostics,makeRecord(Code::NonFinite,2831,32,0,EntityKind::Face,entity,-1,!std::isfinite(yPlus)?yPlus:residual));
    }else if(fabs(residual)>GPU_R(1e-8)*maximum(yPlus,GPU_R(1)))
        recordWarning(diagnostics,makeRecord(Code::AccuracyResidual,2832,32,0,EntityKind::Face,entity,-1,residual,GPU_R(1e-8)*maximum(yPlus,GPU_R(1)),yPlus,Quantity::Accuracy));
    return yPlus;
}

UGKP_WALL_HD GpuReal spaldingReynoldsRatio
(
    const GpuReal uPlus,
    const GpuReal reynolds,
    const GpuReal logReynolds,
    const GpuReal kappa,
    const GpuReal E
)
{
    const GpuReal z = kappa*uPlus;
    GpuReal remainderOverRe;
    if (z < GPU_R(1.0))
    {
        GpuReal term = z*z*z*z/GPU_R(24.0);
        GpuReal remainder = term;
        for (int order = 5; order <= 12; ++order)
        {
            term *= z/GpuReal(order);
            remainder += term;
        }
        remainderOverRe = remainder/reynolds;
    }
    else
    {
        remainderOverRe = exp(z - logReynolds)
          - (GPU_R(1.0) + z*(GPU_R(1.0)
            + z*(GPU_R(0.5) + z/GPU_R(6.0))))/reynolds;
    }
    return (uPlus/reynolds)*uPlus + (uPlus/E)*remainderOverRe;
}

// The law coordinate y and the patch-normal velocity gradient are distinct
// geometric quantities (nutUSpaldingWallFunction, OpenFOAM Foundation v14).
UGKP_WALL_HD SpaldingWallState spaldingWallStateFromNormalGradient
(
    const GpuReal velocityDifference,
    const GpuReal wallDistance,
    const GpuReal kinematicViscosity,
    const GpuReal velocityNormalGradient,
    const GpuReal kappa = GPU_R(0.41),
    const GpuReal E = GPU_R(9.8),
    ugkp::diagnostics::DeviceContext* diagnostics = nullptr,
    const int entity = -1
)
{
    using namespace ugkp::diagnostics;
    const GpuReal inputs[]={velocityDifference,wallDistance,kinematicViscosity,velocityNormalGradient,kappa,E};
    for(int i=0;i<6;++i)if(!std::isfinite(inputs[i])){
        if(diagnostics)recordFatal(diagnostics,makeRecord(Code::NonFinite,2825,32,0,EntityKind::Face,entity,i,inputs[i]));
        const GpuReal nan=std::numeric_limits<GpuReal>::quiet_NaN();return {nan,nan,nan};
    }
    constexpr GpuReal rootVSmall = GPU_TINY(1.4916681462400413e-154);
    const GpuReal up = maximum(velocityDifference, GPU_R(0.0));
    const GpuReal y = maximum(wallDistance, rootVSmall);
    const GpuReal nu = maximum(kinematicViscosity, rootVSmall);
    const GpuReal magGradU = maximum(velocityNormalGradient, GPU_R(0.0));
    // One bracketed root solve for every nonzero velocity. No Newton attempt
    // followed by a residual-triggered replacement solver.
    GpuReal uTau=GPU_R(0);
    if(up>rootVSmall){
        const GpuReal reynolds=up*y/nu,logReynolds=log(reynolds);
        GpuReal lower=GPU_R(0),upper=minimum(sqrt(reynolds),
            (log(maximum(reynolds,GPU_R(1)))+log(maximum(E,GPU_R(1)))+GPU_R(4))/kappa);
        GpuReal ratio=0;
        if(!std::isfinite(reynolds)||reynolds<=0||!std::isfinite(upper)||upper<=0){
            if(diagnostics)recordFatal(diagnostics,makeRecord(!std::isfinite(reynolds)||!std::isfinite(upper)?Code::NonFinite:Code::InvalidPhysicalState,
                2826,32,0,EntityKind::Face,entity,-1,!std::isfinite(reynolds)||reynolds<=0?reynolds:upper));
            const GpuReal nan=std::numeric_limits<GpuReal>::quiet_NaN();return {nan,nan,nan};
        }
        for(int iteration=0;iteration<64;++iteration){
            const GpuReal middle=lower+GPU_R(.5)*(upper-lower);
            if(middle==lower||middle==upper)break;
            ratio=spaldingReynoldsRatio(middle,reynolds,logReynolds,kappa,E);
            if(!std::isfinite(ratio)){
                if(diagnostics)recordFatal(diagnostics,makeRecord(Code::NonFinite,2828,32,0,EntityKind::Face,entity,iteration,ratio));
                const GpuReal nan=std::numeric_limits<GpuReal>::quiet_NaN();return {nan,nan,nan};
            }
            if(fabs(ratio-GPU_R(1))<=(GPU_R(32)*GPU_REAL_EPSILON)){lower=upper=middle;break;}
            if(ratio>GPU_R(1))upper=middle;else lower=middle;
        }
        const GpuReal terminal=lower+GPU_R(.5)*(upper-lower);
        uTau=up/terminal;
        const GpuReal residual=spaldingReynoldsRatio(terminal,reynolds,logReynolds,kappa,E)-GPU_R(1);
        if(!std::isfinite(residual)){
            if(diagnostics)recordFatal(diagnostics,makeRecord(Code::NonFinite,2834,32,0,EntityKind::Face,entity,-1,residual));
        }
        else if(fabs(residual)>(GPU_R(32)*GPU_REAL_EPSILON))
            recordWarning(diagnostics,makeRecord(Code::AccuracyResidual,2827,32,0,EntityKind::Face,entity,-1,residual,(GPU_R(32)*GPU_REAL_EPSILON),uTau,Quantity::Accuracy));
    }
    uTau = maximum(uTau, GPU_R(0.0));
    const GpuReal nut = maximum
    (
        uTau*uTau/(magGradU + rootVSmall) - nu,
        GPU_R(0.0)
    );
    const GpuReal yPlus=y*uTau/nu;
    const GpuReal outputs[]={uTau,yPlus,nut};
    for(int i=0;i<3;++i)if(!std::isfinite(outputs[i])){
        if(diagnostics)recordFatal(diagnostics,makeRecord(Code::NonFinite,2833,32,0,EntityKind::Face,entity,i,outputs[i]));
    }
    return SpaldingWallState{uTau, yPlus, nut};
}

// Backward-compatible orthogonal specialization for callers with y=1/delta.
UGKP_WALL_HD SpaldingWallState spaldingWallState
(
    const GpuReal velocityDifference,
    const GpuReal wallDistance,
    const GpuReal kinematicViscosity,
    const GpuReal kappa = GPU_R(0.41),
    const GpuReal E = GPU_R(9.8)
)
{
    constexpr GpuReal rootVSmall = GPU_TINY(1.4916681462400413e-154);
    return spaldingWallStateFromNormalGradient
    (
        velocityDifference,
        wallDistance,
        kinematicViscosity,
        maximum(velocityDifference, GPU_R(0.0))/maximum(wallDistance, rootVSmall),
        kappa,
        E
    );
}

UGKP_WALL_HD WallSubgridTransport wallSubgridTransport
(
    const GpuReal density,
    const GpuReal heatCapacity,
    const GpuReal turbulentPrandtl,
    const GpuReal wallNut
)
{
    constexpr GpuReal small = GPU_R(2.2204460492503131e-16);
    const GpuReal muT =
        maximum(density, GPU_R(0.0))*maximum(wallNut, GPU_R(0.0));
    return WallSubgridTransport
    {
        muT,
        maximum(heatCapacity, GPU_R(0.0))*muT
       /maximum(turbulentPrandtl, small)
    };
}

UGKP_WALL_HD OmegaWallFunctionState omegaWallFunctionState
(
    const GpuReal k,
    const GpuReal velocityNormalGradient,
    const GpuReal wallDistance,
    const GpuReal kinematicViscosity,
    const GpuReal beta1 = GPU_R(0.075),
    const GpuReal Cmu = GPU_R(0.09),
    const GpuReal kappa = GPU_R(0.41),
    const GpuReal E = GPU_R(9.8),
    const GpuReal cellProduction = GPU_R(0.0),
    ugkp::diagnostics::DeviceContext* diagnostics = nullptr,
    const int entity = -1
)
{
    constexpr GpuReal small = GPU_R(2.2204460492503131e-16);
    const GpuReal kSafe = maximum(k, GPU_R(0.0));
    const GpuReal y = maximum(wallDistance, small);
    const GpuReal nu = maximum(kinematicViscosity, small);
    const GpuReal CmuSafe = maximum(Cmu, small);
    const GpuReal Cmu25 = sqrt(sqrt(CmuSafe));
    const GpuReal Cmu5 = sqrt(CmuSafe);
    const GpuReal reynoldsY = y*sqrt(kSafe)/nu;
    const GpuReal yPlus = Cmu25*reynoldsY;
    const GpuReal omegaViscous = GPU_R(6.0)*nu/(maximum(beta1, small)*y*y);

    if (yPlus < wallYPlusLaminar(kappa, E, diagnostics, entity))
    {
        return OmegaWallFunctionState
        {
            omegaViscous,
            cellProduction,
            yPlus
        };
    }

    const GpuReal uPlus = log(maximum(E*yPlus, GPU_R(1.0) + small))
      /maximum(kappa, small);
    const GpuReal uStar = Cmu25*sqrt(kSafe);
    const GpuReal omegaLog =
        uStar/(Cmu5*maximum(kappa, small)*y);
    const GpuReal scaledShear =
        uStar*maximum(velocityNormalGradient, GPU_R(0.0))*y
       /maximum(uPlus, small);
    const GpuReal productionLog =
        scaledShear*scaledShear
       /(nu*maximum(kappa, small)*maximum(yPlus, small));
    return OmegaWallFunctionState
    {
        maximum(omegaLog, GPU_R(0.0)),
        maximum(productionLog, GPU_R(0.0)),
        yPlus
    };
}

UGKP_WALL_HD GpuReal jayatillekeSmoothP(const GpuReal Prat)
{
    const GpuReal ratio = maximum(Prat, GPU_R(2.2204460492503131e-16));
    return
        GPU_R(9.24)*(pow(ratio, GPU_R(0.75)) - GPU_R(1.0))
       *(GPU_R(1.0) + GPU_R(0.28)*exp(-GPU_R(0.007)*ratio));
}

// No matching root is an undefined model value, not a zero wall coordinate.
// The optional context lets production callers retain the exact root site.
UGKP_WALL_HD GpuReal jayatillekeThermalYPlus
(
    const GpuReal Prat,
    const GpuReal kappa = GPU_R(0.41),
    const GpuReal E = GPU_R(9.8),
    ugkp::diagnostics::DeviceContext* diagnostics = nullptr,
    const int entity = -1,
    const int maxIterations = 80
)
{
    using namespace ugkp::diagnostics;
    const GpuReal nan=std::numeric_limits<GpuReal>::quiet_NaN();
    const GpuReal inputs[]={Prat,kappa,E};
    for(int i=0;i<3;++i)if(!std::isfinite(inputs[i])||inputs[i]<=0){
        if(diagnostics)recordFatal(diagnostics,makeRecord(!std::isfinite(inputs[i])?Code::NonFinite:
            (inputs[i]==0?Code::ZeroDenominator:Code::InvalidPhysicalState),2820,32,0,EntityKind::Face,entity,i,inputs[i]));
        return nan;
    }
    const GpuReal P=jayatillekeSmoothP(Prat);
    // x=kappa*Prat*y gives x-log(x)=A. Its outer root has x>=1.
    const GpuReal A=log(E)-log(kappa)-log(Prat)+kappa*P;
    if(!std::isfinite(A)||A<GPU_R(1)){
        if(diagnostics)recordFatal(diagnostics,makeRecord(!std::isfinite(A)?Code::NonFinite:Code::InvalidPhysicalState,
            2821,32,0,EntityKind::Face,entity,-1,A,1));
        return nan;
    }
    GpuReal lower=GPU_R(1);
    // For A>=1, A+log(A)+2 brackets the outer root analytically.
    GpuReal upper=maximum(GPU_R(2),A+log(A)+GPU_R(2));
    GpuReal x=upper,residual=0,tolerance=0;
    for(int iteration=0;iteration<maxIterations;++iteration){
        residual=x-log(x)-A;
        tolerance=GPU_R(8)*GPU_REAL_EPSILON*maximum(GPU_R(1),maximum(A,x));
        if(!std::isfinite(residual)){
            if(diagnostics)recordFatal(diagnostics,makeRecord(Code::NonFinite,2822,32,0,EntityKind::Face,entity,iteration,residual));
            return nan;
        }
        if(fabs(residual)<=tolerance)break;
        if(residual>0)upper=x;else lower=x;
        const GpuReal derivative=GPU_R(1)-GPU_R(1)/x;
        const GpuReal candidate=derivative!=0?x-residual/derivative:lower;
        x=std::isfinite(candidate)&&candidate>lower&&candidate<upper?candidate:lower+GPU_R(.5)*(upper-lower);
    }
    residual=x-log(x)-A;
    if(fabs(residual)>tolerance)
        recordWarning(diagnostics,makeRecord(Code::AccuracyResidual,2823,32,0,EntityKind::Face,entity,-1,residual,tolerance,x,Quantity::Accuracy));
    const GpuReal denominator=kappa*Prat;
    // Select the algebraic evaluation by scaling before division, not by retry.
    const GpuReal yPlus=std::isfinite(denominator)&&denominator>0?x/denominator:exp(log(x)-log(kappa)-log(Prat));
    if(!std::isfinite(yPlus)||yPlus<=0){
        if(diagnostics)recordFatal(diagnostics,makeRecord(!std::isfinite(yPlus)?Code::NonFinite:Code::InvalidPhysicalState,
            2824,32,0,EntityKind::Face,entity,-1,yPlus));
        return nan;
    }
    return yPlus;
}

struct JayatillekeThermalTransport
{
    GpuReal conductivity;
    int valid;
    GpuReal heatFlux = GPU_R(0.0);
};

UGKP_WALL_HD JayatillekeThermalTransport sstJayatillekeThermalTransport
(
    const GpuReal rhoWall,
    const GpuReal cp,
    const GpuReal mu,
    const GpuReal Pr,
    const GpuReal Prt,
    const GpuReal Cmu,
    const GpuReal kappa,
    const GpuReal E,
    const GpuReal P,
    const GpuReal yPlusThermal,
    const GpuReal turbulentK,
    const GpuReal y,
    const GpuReal velocityDifference,
    const GpuReal wallSpeed,
    const GpuReal temperatureNormalGradient, const bool trialEligibility=false
)
{
    const GpuReal molecular = mu*cp/Pr;
    const GpuReal uStar = sqrt(sqrt(Cmu))*sqrt(maximum(turbulentK, GPU_R(0.0)));
    const bool inputsFinite=std::isfinite(molecular)&&std::isfinite(uStar)
        &&std::isfinite(temperatureNormalGradient);
    if (trialEligibility && !inputsFinite)
    {
        return {molecular, 0};
    }
    if (uStar == GPU_R(0.0))
    {
        const GpuReal heatFlux = -molecular*temperatureNormalGradient;
        return {molecular, std::isfinite(heatFlux) ? 1 : 0, heatFlux};
    }


    if (trialEligibility && mu == GPU_R(0.0)) return {molecular, 0};
    const GpuReal yPlus = uStar*y*rhoWall/mu;
    const bool viscous = yPlus < yPlusThermal;
    const GpuReal tPlus = viscous ? Pr*yPlus : Prt*(log(E*yPlus)/kappa + P);
    if (trialEligibility && (tPlus == GPU_R(0.0) || !std::isfinite(tPlus)))
    {
        return {molecular, 0};
    }
    const GpuReal uc = viscous ? GPU_R(0.0)
      : uStar/kappa*log(E*yPlusThermal) - wallSpeed;
    const GpuReal C = GPU_R(0.5)*rhoWall*uStar
      *(viscous ? Pr*velocityDifference*velocityDifference
        : Prt*velocityDifference*velocityDifference + (Pr-Prt)*uc*uc);
    const GpuReal conductivity = maximum(molecular, cp*rhoWall*uStar*y/tPlus);
    const GpuReal heatFlux = -conductivity*temperatureNormalGradient + C/tPlus;
    return {conductivity, inputsFinite && mu != GPU_R(0.0) && tPlus != GPU_R(0.0)
        && std::isfinite(tPlus) && std::isfinite(heatFlux) ? 1 : 0, heatFlux};
}

UGKP_WALL_HD JayatillekeWallHeatState jayatillekeWallHeatFluxPrecomputed
(
    const GpuReal density,
    const GpuReal heatCapacity,
    const GpuReal molecularPrandtl,
    const GpuReal turbulentPrandtl,
    const GpuReal kappa,
    const GpuReal E,
    const GpuReal P,
    const GpuReal thermalYPlus,
    const GpuReal uTau,
    const GpuReal yPlus,
    const GpuReal cellTemperature,
    const GpuReal wallTemperature
)
{
    constexpr GpuReal small = GPU_R(2.2204460492503131e-16);
    const GpuReal Pr = maximum(molecularPrandtl, small);
    const GpuReal Prt = maximum(turbulentPrandtl, small);
    const GpuReal yPlusSafe = maximum(yPlus, GPU_R(0.0));
    const bool valid =
        density > GPU_R(0.0)
     && heatCapacity > GPU_R(0.0)
     && uTau > small
     && yPlusSafe > small
     && molecularPrandtl > GPU_R(0.0)
     && turbulentPrandtl > GPU_R(0.0)
     && kappa > GPU_R(0.0) && E > GPU_R(0.0)
     && thermalYPlus > GPU_R(0.0)
     && std::isfinite(density) && std::isfinite(heatCapacity)
     && std::isfinite(uTau) && std::isfinite(yPlus)
     && std::isfinite(thermalYPlus) && std::isfinite(P);
    if (!valid)
    {
        return JayatillekeWallHeatState{std::numeric_limits<GpuReal>::quiet_NaN(), std::numeric_limits<GpuReal>::quiet_NaN(), thermalYPlus, 0};
    }

    const GpuReal temperaturePlus = yPlusSafe < thermalYPlus
      ? Pr*yPlusSafe
      : Prt*
        (
            log(maximum(E*yPlusSafe, GPU_R(1.0) + small))/maximum(kappa, small)
          + P
        );
    if (!(temperaturePlus > GPU_R(0.0)) || !std::isfinite(temperaturePlus))
    {
        return JayatillekeWallHeatState{std::numeric_limits<GpuReal>::quiet_NaN(), temperaturePlus, thermalYPlus, 0};
    }
    const GpuReal heatFlux =
        density*heatCapacity*uTau*(cellTemperature - wallTemperature)/temperaturePlus;
    return JayatillekeWallHeatState
    {
        heatFlux,
        temperaturePlus,
        thermalYPlus,
        std::isfinite(heatFlux) ? 1 : 0
    };
}

UGKP_WALL_HD JayatillekeWallHeatState jayatillekeWallHeatFlux
(
    const GpuReal density,
    const GpuReal heatCapacity,
    const GpuReal molecularPrandtl,
    const GpuReal turbulentPrandtl,
    const GpuReal kappa,
    const GpuReal E,
    const GpuReal uTau,
    const GpuReal yPlus,
    const GpuReal cellTemperature,
    const GpuReal wallTemperature
)
{
    constexpr GpuReal small = GPU_R(2.2204460492503131e-16);
    const GpuReal Pr = maximum(molecularPrandtl, small);
    const GpuReal Prt = maximum(turbulentPrandtl, small);
    const GpuReal PrRatio = Pr/Prt;
    return jayatillekeWallHeatFluxPrecomputed
    (
        density,
        heatCapacity,
        molecularPrandtl,
        turbulentPrandtl,
        kappa,
        E,
        jayatillekeSmoothP(PrRatio),
        jayatillekeThermalYPlus(PrRatio, kappa, E),
        uTau,
        yPlus,
        cellTemperature,
        wallTemperature
    );
}

}                      

#undef UGKP_WALL_HD
