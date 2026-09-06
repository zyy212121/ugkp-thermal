#include "GpuPrecisionTypes.H"
#pragma once

  
                                          
  
                                                                            
                                                                             
                                                                           
  
                      
                                                                            
                                                       
                                                                           
                               
                                                                            
                                                                      
                   
                                                                    
                      
                                                                   
                                                                          
                                                                          
                            
                                                                          
                                                                       
                                                                          
                                                                            
                                                                
                                                                     
                                                                            
  
                                                                        
                                                     
   

#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#define UGKP_RIEMANN_HD __host__ __device__ __forceinline__
#else
#define UGKP_RIEMANN_HD inline
#endif

namespace ugkpriemann
{

enum class Scheme : int
{
    RusanovTadmor = 0,
    HllKurganov = 1,
    HLLE = 2,
    HLLC = 3,
    Roe = 4,
    HLLEM = 5,
    HLLC_ADC = 6,
    SLAU2 = 7,
    SLAU2_2 = 8
};

  
                                                                         
                         
  
                                            
                                             
                                             
                                             
                                             
                                             
                                             
                                             
                                             
  
                                                                       
                                  
   
UGKP_RIEMANN_HD bool schemeFromCreateCode
(
    const int createCode,
    Scheme& scheme
)
{
    switch (createCode)
    {
        case 1:
            scheme = Scheme::RusanovTadmor;
            return true;
        case 2:
            scheme = Scheme::HllKurganov;
            return true;
        case 3:
            scheme = Scheme::HLLE;
            return true;
        case 4:
            scheme = Scheme::HLLC;
            return true;
        case 5:
            scheme = Scheme::Roe;
            return true;
        case 6:
            scheme = Scheme::HLLEM;
            return true;
        case 7:
            scheme = Scheme::HLLC_ADC;
            return true;
        case 8:
            scheme = Scheme::SLAU2;
            return true;
        case 9:
            scheme = Scheme::SLAU2_2;
            return true;
        default:
            scheme = Scheme::RusanovTadmor;
            return false;
    }
}

UGKP_RIEMANN_HD int createCodeFromScheme(const Scheme scheme)
{
    switch (scheme)
    {
        case Scheme::RusanovTadmor:
            return 1;
        case Scheme::HllKurganov:
            return 2;
        case Scheme::HLLE:
            return 3;
        case Scheme::HLLC:
            return 4;
        case Scheme::Roe:
            return 5;
        case Scheme::HLLEM:
            return 6;
        case Scheme::HLLC_ADC:
            return 7;
        case Scheme::SLAU2:
            return 8;
        case Scheme::SLAU2_2:
            return 9;
        default:
            return 0;
    }
}

struct Primitive
{
    GpuReal rho;
    GpuReal ux;
    GpuReal uy;
    GpuReal uz;
    GpuReal p;
};

struct DensityGradient
{
    GpuReal x;
    GpuReal y;
    GpuReal z;
};

struct Conservative
{
    GpuReal q[5];
};

struct FluxResult
{
    GpuReal flux[5];
    GpuReal maxSignalSpeed;
    Scheme evaluatedScheme;
    bool valid;
    bool usedFallback;
};

struct RoeAverage
{
    GpuReal ux;
    GpuReal uy;
    GpuReal uz;
    GpuReal enthalpy;
    GpuReal soundSpeed;
    GpuReal normalVelocity;
    GpuReal density;
    bool valid;
};

UGKP_RIEMANN_HD GpuReal minimum(const GpuReal a, const GpuReal b)
{
    return a < b ? a : b;
}

UGKP_RIEMANN_HD GpuReal maximum(const GpuReal a, const GpuReal b)
{
    return a > b ? a : b;
}

UGKP_RIEMANN_HD GpuReal absolute(const GpuReal a)
{
    return a < GPU_R(0.0) ? -a : a;
}

UGKP_RIEMANN_HD bool finiteScalar(const GpuReal value)
{
    return value == value && value <= GPU_REAL_MAX && value >= -GPU_REAL_MAX;
}

UGKP_RIEMANN_HD bool finiteFive(const GpuReal values[5])
{
    for (int component = 0; component < 5; ++component)
    {
        if (!finiteScalar(values[component]))
        {
            return false;
        }
    }
    return true;
}

UGKP_RIEMANN_HD FluxResult invalidResult(const Scheme scheme)
{
    FluxResult result;
    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] = GPU_R(0.0);
    }
    result.maxSignalSpeed = GPU_R(0.0);
    result.evaluatedScheme = scheme;
    result.valid = false;
    result.usedFallback = false;
    return result;
}

UGKP_RIEMANN_HD bool normalise
(
    const GpuReal normalX,
    const GpuReal normalY,
    const GpuReal normalZ,
    GpuReal& nx,
    GpuReal& ny,
    GpuReal& nz,
    GpuReal& magnitude
)
{
    const GpuReal scale = maximum
    (
        absolute(normalX),
        maximum(absolute(normalY), absolute(normalZ))
    );
    if (!finiteScalar(scale) || scale <= GPU_R(0.0))
    {
        nx = GPU_R(0.0);
        ny = GPU_R(0.0);
        nz = GPU_R(0.0);
        magnitude = GPU_R(0.0);
        return false;
    }

    const GpuReal scaledX = normalX/scale;
    const GpuReal scaledY = normalY/scale;
    const GpuReal scaledZ = normalZ/scale;
    const GpuReal scaledMagnitude = ::sqrt
    (
        scaledX*scaledX + scaledY*scaledY + scaledZ*scaledZ
    );
    if (!finiteScalar(scaledMagnitude) || scaledMagnitude <= GPU_R(0.0))
    {
        nx = GPU_R(0.0);
        ny = GPU_R(0.0);
        nz = GPU_R(0.0);
        magnitude = GPU_R(0.0);
        return false;
    }
    nx = scaledX/scaledMagnitude;
    ny = scaledY/scaledMagnitude;
    nz = scaledZ/scaledMagnitude;
    magnitude = scale*scaledMagnitude;
    return
        finiteScalar(nx)
     && finiteScalar(ny)
     && finiteScalar(nz);
}

UGKP_RIEMANN_HD bool physicalPrimitive
(
    const Primitive& state,
    const GpuReal gamma,
    const GpuReal rhoFloor,
    const GpuReal pressureFloor
)
{
    return
        finiteScalar(gamma)
     && gamma > GPU_R(1.0)
     && finiteScalar(rhoFloor)
     && rhoFloor > GPU_R(0.0)
     && finiteScalar(pressureFloor)
     && pressureFloor > GPU_R(0.0)
     && finiteScalar(state.rho)
     && finiteScalar(state.ux)
     && finiteScalar(state.uy)
     && finiteScalar(state.uz)
     && finiteScalar(state.p)
     && state.rho >= rhoFloor
     && state.p >= pressureFloor;
}

UGKP_RIEMANN_HD Conservative conservative
(
    const Primitive& state,
    const GpuReal gamma
)
{
    Conservative result;
    result.q[0] = state.rho;
    result.q[1] = state.rho*state.ux;
    result.q[2] = state.rho*state.uy;
    result.q[3] = state.rho*state.uz;
    result.q[4] =
        state.p/(gamma - GPU_R(1.0))
      + GPU_R(0.5)*state.rho*
       (
           state.ux*state.ux
         + state.uy*state.uy
         + state.uz*state.uz
       );
    return result;
}

UGKP_RIEMANN_HD GpuReal totalEnthalpy
(
    const Primitive& state,
    const GpuReal gamma
)
{
    const GpuReal velocitySquared =
        state.ux*state.ux + state.uy*state.uy + state.uz*state.uz;
    return
        gamma*state.p/(state.rho*(gamma - GPU_R(1.0)))
      + GPU_R(0.5)*velocitySquared;
}

UGKP_RIEMANN_HD GpuReal normalVelocity
(
    const Primitive& state,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz
)
{
    return state.ux*nx + state.uy*ny + state.uz*nz;
}

UGKP_RIEMANN_HD void projectedEulerFlux
(
    const Primitive& state,
    const Conservative& conserved,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    GpuReal flux[5]
)
{
    const GpuReal un = normalVelocity(state, nx, ny, nz);
    flux[0] = state.rho*un;
    flux[1] = state.rho*state.ux*un + state.p*nx;
    flux[2] = state.rho*state.uy*un + state.p*ny;
    flux[3] = state.rho*state.uz*un + state.p*nz;
    flux[4] = (conserved.q[4] + state.p)*un;
}

UGKP_RIEMANN_HD bool physicalConservative
(
    const Conservative& state,
    const GpuReal gamma,
    const GpuReal rhoFloor,
    const GpuReal pressureFloor
)
{
    if
    (
        !finiteScalar(gamma)
     || gamma <= GPU_R(1.0)
     || !finiteScalar(rhoFloor)
     || rhoFloor <= GPU_R(0.0)
     || !finiteScalar(pressureFloor)
     || pressureFloor <= GPU_R(0.0)
     || !finiteFive(state.q)
     || state.q[0] < rhoFloor
    )
    {
        return false;
    }

    const GpuReal inverseDensity = GPU_R(1.0)/state.q[0];
    const GpuReal kineticEnergy =
        GPU_R(0.5)*
       (
           state.q[1]*state.q[1]
         + state.q[2]*state.q[2]
         + state.q[3]*state.q[3]
       )*inverseDensity;
    const GpuReal pressure = (gamma - GPU_R(1.0))*(state.q[4] - kineticEnergy);
    return finiteScalar(pressure) && pressure >= pressureFloor;
}

UGKP_RIEMANN_HD RoeAverage makeRoeAverage
(
    const Primitive& left,
    const Primitive& right,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal gamma
)
{
    RoeAverage average;
    average.valid = false;
    average.ux = GPU_R(0.0);
    average.uy = GPU_R(0.0);
    average.uz = GPU_R(0.0);
    average.enthalpy = GPU_R(0.0);
    average.soundSpeed = GPU_R(0.0);
    average.normalVelocity = GPU_R(0.0);
    average.density = GPU_R(0.0);

    const GpuReal sqrtLeftDensity = ::sqrt(left.rho);
    const GpuReal sqrtRightDensity = ::sqrt(right.rho);
    const GpuReal denominator = sqrtLeftDensity + sqrtRightDensity;
    if (!finiteScalar(denominator) || denominator <= GPU_REAL_MIN)
    {
        return average;
    }

    average.ux =
        (sqrtLeftDensity*left.ux + sqrtRightDensity*right.ux)/denominator;
    average.uy =
        (sqrtLeftDensity*left.uy + sqrtRightDensity*right.uy)/denominator;
    average.uz =
        (sqrtLeftDensity*left.uz + sqrtRightDensity*right.uz)/denominator;
    average.enthalpy =
       (
           sqrtLeftDensity*totalEnthalpy(left, gamma)
         + sqrtRightDensity*totalEnthalpy(right, gamma)
       )/denominator;
    average.density = sqrtLeftDensity*sqrtRightDensity;

    const GpuReal velocitySquared =
        average.ux*average.ux
      + average.uy*average.uy
      + average.uz*average.uz;
    const GpuReal soundSpeedSquared =
        (gamma - GPU_R(1.0))*(average.enthalpy - GPU_R(0.5)*velocitySquared);
    if (!finiteScalar(soundSpeedSquared) || soundSpeedSquared <= GPU_REAL_MIN)
    {
        return average;
    }

    average.soundSpeed = ::sqrt(soundSpeedSquared);
    average.normalVelocity =
        average.ux*nx + average.uy*ny + average.uz*nz;
    average.valid =
        finiteScalar(average.ux)
     && finiteScalar(average.uy)
     && finiteScalar(average.uz)
     && finiteScalar(average.enthalpy)
     && finiteScalar(average.soundSpeed)
     && finiteScalar(average.normalVelocity)
     && finiteScalar(average.density);
    return average;
}

UGKP_RIEMANN_HD FluxResult rusanovTadmorFluxUnitNormal
(
    const Primitive& left,
    const Primitive& right,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal gamma,
    const bool inheritedFallback
)
{
    FluxResult result = invalidResult(Scheme::RusanovTadmor);
    const Conservative leftConserved = conservative(left, gamma);
    const Conservative rightConserved = conservative(right, gamma);
    GpuReal leftFlux[5];
    GpuReal rightFlux[5];
    projectedEulerFlux(left, leftConserved, nx, ny, nz, leftFlux);
    projectedEulerFlux(right, rightConserved, nx, ny, nz, rightFlux);

    const GpuReal leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const GpuReal rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const GpuReal leftNormalVelocity = normalVelocity(left, nx, ny, nz);
    const GpuReal rightNormalVelocity = normalVelocity(right, nx, ny, nz);
    const GpuReal signalSpeed = maximum
    (
        absolute(leftNormalVelocity) + leftSoundSpeed,
        absolute(rightNormalVelocity) + rightSoundSpeed
    );

    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] =
            GPU_R(0.5)*(leftFlux[component] + rightFlux[component])
          - GPU_R(0.5)*signalSpeed*
            (rightConserved.q[component] - leftConserved.q[component]);
    }
    result.maxSignalSpeed = signalSpeed;
    result.valid =
        finiteScalar(signalSpeed)
     && signalSpeed >= GPU_R(0.0)
     && finiteFive(result.flux);
    result.usedFallback = inheritedFallback;
    return result;
}

UGKP_RIEMANN_HD FluxResult hllKurganovFluxUnitNormal
(
    const Primitive& left,
    const Primitive& right,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal gamma,
    const bool inheritedFallback
)
{
    const Conservative leftConserved = conservative(left, gamma);
    const Conservative rightConserved = conservative(right, gamma);
    GpuReal leftFlux[5];
    GpuReal rightFlux[5];
    projectedEulerFlux(left, leftConserved, nx, ny, nz, leftFlux);
    projectedEulerFlux(right, rightConserved, nx, ny, nz, rightFlux);

    const GpuReal leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const GpuReal rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const GpuReal leftNormalVelocity = normalVelocity(left, nx, ny, nz);
    const GpuReal rightNormalVelocity = normalVelocity(right, nx, ny, nz);
    const GpuReal positiveSpeed = maximum
    (
        GPU_R(0.0),
        maximum
        (
            leftNormalVelocity + leftSoundSpeed,
            rightNormalVelocity + rightSoundSpeed
        )
    );
    const GpuReal negativeSpeed = minimum
    (
        GPU_R(0.0),
        minimum
        (
            leftNormalVelocity - leftSoundSpeed,
            rightNormalVelocity - rightSoundSpeed
        )
    );
    const GpuReal denominator = positiveSpeed - negativeSpeed;
    const GpuReal scale =
        maximum
        (
            GPU_R(1.0),
            maximum(absolute(positiveSpeed), absolute(negativeSpeed))
        );
    if
    (
        !finiteScalar(denominator)
     || denominator <= GPU_R(64.0)*GPU_REAL_EPSILON*scale
    )
    {
        return rusanovTadmorFluxUnitNormal
        (
            left, right, nx, ny, nz, gamma, true
        );
    }

    FluxResult result = invalidResult(Scheme::HllKurganov);
    const GpuReal diffusion =
        positiveSpeed*negativeSpeed/denominator;
    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] =
            positiveSpeed/denominator*leftFlux[component]
          - negativeSpeed/denominator*rightFlux[component]
          + diffusion*
            (rightConserved.q[component] - leftConserved.q[component]);
    }
    result.maxSignalSpeed =
        maximum(absolute(positiveSpeed), absolute(negativeSpeed));
    result.valid =
        finiteScalar(result.maxSignalSpeed)
     && finiteFive(result.flux);
    result.usedFallback = inheritedFallback;
    return result;
}

UGKP_RIEMANN_HD FluxResult hlleFluxUnitNormal
(
    const Primitive& left,
    const Primitive& right,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal gamma,
    const GpuReal rhoFloor,
    const GpuReal pressureFloor,
    const bool inheritedFallback
)
{
    const Conservative leftConserved = conservative(left, gamma);
    const Conservative rightConserved = conservative(right, gamma);
    GpuReal leftFlux[5];
    GpuReal rightFlux[5];
    projectedEulerFlux(left, leftConserved, nx, ny, nz, leftFlux);
    projectedEulerFlux(right, rightConserved, nx, ny, nz, rightFlux);

    const GpuReal leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const GpuReal rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const GpuReal leftNormalVelocity = normalVelocity(left, nx, ny, nz);
    const GpuReal rightNormalVelocity = normalVelocity(right, nx, ny, nz);
    const RoeAverage roe =
        makeRoeAverage(left, right, nx, ny, nz, gamma);

    GpuReal negativeSpeed = minimum
    (
        GPU_R(0.0),
        minimum
        (
            leftNormalVelocity - leftSoundSpeed,
            rightNormalVelocity - rightSoundSpeed
        )
    );
    GpuReal positiveSpeed = maximum
    (
        GPU_R(0.0),
        maximum
        (
            leftNormalVelocity + leftSoundSpeed,
            rightNormalVelocity + rightSoundSpeed
        )
    );
    if (roe.valid)
    {
        negativeSpeed = minimum
        (
            negativeSpeed,
            roe.normalVelocity - roe.soundSpeed
        );
        positiveSpeed = maximum
        (
            positiveSpeed,
            roe.normalVelocity + roe.soundSpeed
        );
    }

    if (negativeSpeed >= GPU_R(0.0))
    {
        FluxResult result = invalidResult(Scheme::HLLE);
        for (int component = 0; component < 5; ++component)
        {
            result.flux[component] = leftFlux[component];
        }
        result.maxSignalSpeed = positiveSpeed;
        result.valid =
            finiteScalar(result.maxSignalSpeed)
         && finiteFive(result.flux);
        result.usedFallback = inheritedFallback;
        return result;
    }
    if (positiveSpeed <= GPU_R(0.0))
    {
        FluxResult result = invalidResult(Scheme::HLLE);
        for (int component = 0; component < 5; ++component)
        {
            result.flux[component] = rightFlux[component];
        }
        result.maxSignalSpeed = absolute(negativeSpeed);
        result.valid =
            finiteScalar(result.maxSignalSpeed)
         && finiteFive(result.flux);
        result.usedFallback = inheritedFallback;
        return result;
    }

    const GpuReal denominator = positiveSpeed - negativeSpeed;
    const GpuReal scale =
        maximum
        (
            GPU_R(1.0),
            maximum(absolute(positiveSpeed), absolute(negativeSpeed))
        );
    if
    (
        !finiteScalar(denominator)
     || denominator <= GPU_R(64.0)*GPU_REAL_EPSILON*scale
    )
    {
        return rusanovTadmorFluxUnitNormal
        (
            left, right, nx, ny, nz, gamma, true
        );
    }

    FluxResult result = invalidResult(Scheme::HLLE);
    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] =
           (
               positiveSpeed*leftFlux[component]
             - negativeSpeed*rightFlux[component]
             + positiveSpeed*negativeSpeed*
               (rightConserved.q[component] - leftConserved.q[component])
           )/denominator;
    }

    Conservative intermediate;
    for (int component = 0; component < 5; ++component)
    {
        intermediate.q[component] =
           (
               positiveSpeed*rightConserved.q[component]
             - negativeSpeed*leftConserved.q[component]
             - (rightFlux[component] - leftFlux[component])
           )/denominator;
    }
    if
    (
        !finiteFive(result.flux)
     || !physicalConservative
        (
            intermediate, gamma, rhoFloor, pressureFloor
        )
    )
    {
        return rusanovTadmorFluxUnitNormal
        (
            left, right, nx, ny, nz, gamma, true
        );
    }

    result.maxSignalSpeed =
        maximum(absolute(positiveSpeed), absolute(negativeSpeed));
    result.valid = finiteScalar(result.maxSignalSpeed);
    result.usedFallback = inheritedFallback;
    return result;
}

UGKP_RIEMANN_HD FluxResult hllemFluxUnitNormal
(
    const Primitive& left,
    const Primitive& right,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal gamma,
    const GpuReal rhoFloor,
    const GpuReal pressureFloor
)
{
    FluxResult result = hlleFluxUnitNormal
    (
        left,
        right,
        nx,
        ny,
        nz,
        gamma,
        rhoFloor,
        pressureFloor,
        false
    );
    if
    (
        !result.valid
     || result.usedFallback
     || result.evaluatedScheme != Scheme::HLLE
    )
    {
        return result;
    }

    const GpuReal leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const GpuReal rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const GpuReal leftNormalVelocity = normalVelocity(left, nx, ny, nz);
    const GpuReal rightNormalVelocity = normalVelocity(right, nx, ny, nz);
    const RoeAverage roe =
        makeRoeAverage(left, right, nx, ny, nz, gamma);
    if (!roe.valid)
    {
        return result;
    }

    GpuReal negativeSpeed = minimum
    (
        GPU_R(0.0),
        minimum
        (
            leftNormalVelocity - leftSoundSpeed,
            rightNormalVelocity - rightSoundSpeed
        )
    );
    GpuReal positiveSpeed = maximum
    (
        GPU_R(0.0),
        maximum
        (
            leftNormalVelocity + leftSoundSpeed,
            rightNormalVelocity + rightSoundSpeed
        )
    );
    negativeSpeed = minimum
    (
        negativeSpeed,
        roe.normalVelocity - roe.soundSpeed
    );
    positiveSpeed = maximum
    (
        positiveSpeed,
        roe.normalVelocity + roe.soundSpeed
    );
    if (negativeSpeed >= GPU_R(0.0) || positiveSpeed <= GPU_R(0.0))
    {
        result.evaluatedScheme = Scheme::HLLEM;
        return result;
    }

    const GpuReal denominator = positiveSpeed - negativeSpeed;
    const GpuReal scale = maximum
    (
        GPU_R(1.0),
        maximum(absolute(positiveSpeed), absolute(negativeSpeed))
    );
    if
    (
        !finiteScalar(denominator)
     || denominator <= GPU_R(64.0)*GPU_REAL_EPSILON*scale
    )
    {
        return result;
    }

    const GpuReal densityJump = right.rho - left.rho;
    const GpuReal pressureJump = right.p - left.p;
    const GpuReal velocityJumpX = right.ux - left.ux;
    const GpuReal velocityJumpY = right.uy - left.uy;
    const GpuReal velocityJumpZ = right.uz - left.uz;
    const GpuReal normalVelocityJump =
        velocityJumpX*nx + velocityJumpY*ny + velocityJumpZ*nz;
    const GpuReal tangentialVelocityJumpX =
        velocityJumpX - normalVelocityJump*nx;
    const GpuReal tangentialVelocityJumpY =
        velocityJumpY - normalVelocityJump*ny;
    const GpuReal tangentialVelocityJumpZ =
        velocityJumpZ - normalVelocityJump*nz;
    const GpuReal soundSpeedSquared = roe.soundSpeed*roe.soundSpeed;
    const GpuReal contactStrength =
        densityJump - pressureJump/soundSpeedSquared;
    const GpuReal roeVelocitySquared =
        roe.ux*roe.ux + roe.uy*roe.uy + roe.uz*roe.uz;

    GpuReal linearlyDegenerateJump[5];
    linearlyDegenerateJump[0] = contactStrength;
    linearlyDegenerateJump[1] =
        contactStrength*roe.ux
      + roe.density*tangentialVelocityJumpX;
    linearlyDegenerateJump[2] =
        contactStrength*roe.uy
      + roe.density*tangentialVelocityJumpY;
    linearlyDegenerateJump[3] =
        contactStrength*roe.uz
      + roe.density*tangentialVelocityJumpZ;
    linearlyDegenerateJump[4] =
        GPU_R(0.5)*contactStrength*roeVelocitySquared
      + roe.density*
       (
           roe.ux*tangentialVelocityJumpX
         + roe.uy*tangentialVelocityJumpY
         + roe.uz*tangentialVelocityJumpZ
       );

                                                                       
                                                                        
                                                                       
                                                                
                                                   
    const GpuReal antidiffusion = minimum
    (
        GPU_R(1.0),
        maximum
        (
            GPU_R(0.0),
            roe.normalVelocity >= GPU_R(0.0)
          ? (positiveSpeed - roe.normalVelocity)/positiveSpeed
          : (roe.normalVelocity - negativeSpeed)/(-negativeSpeed)
        )
    );
    const GpuReal correction =
        -negativeSpeed*positiveSpeed/denominator*antidiffusion;
    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] +=
            correction*linearlyDegenerateJump[component];
    }
    if (!finiteFive(result.flux))
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }
    result.evaluatedScheme = Scheme::HLLEM;
    result.usedFallback = false;
    return result;
}

  
                                                                        
  
                                                                        
                                                                             
                                                                     
                                                                           
                                                                            
   
UGKP_RIEMANN_HD FluxResult slau2FluxUnitNormalImpl
(
    const Primitive& left,
    const Primitive& right,
    const DensityGradient& leftDensityGradient,
    const DensityGradient& rightDensityGradient,
    const bool densityGradientAlignedDamping,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal gamma,
    const GpuReal rhoFloor,
    const GpuReal pressureFloor,
    const Scheme evaluatedScheme
)
{
    const GpuReal leftNormalVelocity =
        normalVelocity(left, nx, ny, nz);
    const GpuReal rightNormalVelocity =
        normalVelocity(right, nx, ny, nz);
    const GpuReal leftVelocitySquared =
        left.ux*left.ux + left.uy*left.uy + left.uz*left.uz;
    const GpuReal rightVelocitySquared =
        right.ux*right.ux + right.uy*right.uy + right.uz*right.uz;
    const GpuReal leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const GpuReal rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const GpuReal interfaceSoundSpeed =
        GPU_R(0.5)*(leftSoundSpeed + rightSoundSpeed);
    const GpuReal soundScale = maximum
    (
        GPU_R(1.0),
        maximum(leftSoundSpeed, rightSoundSpeed)
    );
    if
    (
        !finiteScalar(interfaceSoundSpeed)
     || interfaceSoundSpeed <= GPU_R(64.0)*GPU_REAL_EPSILON*soundScale
    )
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }

    const GpuReal leftMach = leftNormalVelocity/interfaceSoundSpeed;
    const GpuReal rightMach = rightNormalVelocity/interfaceSoundSpeed;
    const GpuReal velocityMagnitude = ::sqrt
    (
        GPU_R(0.5)*(leftVelocitySquared + rightVelocitySquared)
    );
    GpuReal dampingMach = velocityMagnitude/interfaceSoundSpeed;
    if (densityGradientAlignedDamping)
    {
        const GpuReal leftGradientMagnitude = ::sqrt
        (
            leftDensityGradient.x*leftDensityGradient.x
          + leftDensityGradient.y*leftDensityGradient.y
          + leftDensityGradient.z*leftDensityGradient.z
        );
        const GpuReal rightGradientMagnitude = ::sqrt
        (
            rightDensityGradient.x*rightDensityGradient.x
          + rightDensityGradient.y*rightDensityGradient.y
          + rightDensityGradient.z*rightDensityGradient.z
        );
        const GpuReal gradientScale = maximum
        (
            GPU_R(1.0),
            maximum(leftGradientMagnitude, rightGradientMagnitude)
        );
        const GpuReal gradientFloor = GPU_R(64.0)*GPU_REAL_EPSILON*gradientScale;
        const GpuReal leftGradientMach =
            leftGradientMagnitude > gradientFloor
          ? (
                left.ux*leftDensityGradient.x
              + left.uy*leftDensityGradient.y
              + left.uz*leftDensityGradient.z
            )/(leftGradientMagnitude*interfaceSoundSpeed)
          : velocityMagnitude/interfaceSoundSpeed;
        const GpuReal rightGradientMach =
            rightGradientMagnitude > gradientFloor
          ? (
                right.ux*rightDensityGradient.x
              + right.uy*rightDensityGradient.y
              + right.uz*rightDensityGradient.z
            )/(rightGradientMagnitude*interfaceSoundSpeed)
          : velocityMagnitude/interfaceSoundSpeed;
        dampingMach = ::sqrt
        (
            GPU_R(0.5)*
            (
                leftGradientMach*leftGradientMach
              + rightGradientMach*rightGradientMach
            )
        );
    }
    const GpuReal limitedMach = minimum(GPU_R(1.0), dampingMach);
    const GpuReal chi = (GPU_R(1.0) - limitedMach)*(GPU_R(1.0) - limitedMach);
    const GpuReal densitySwitch =
       -maximum(minimum(leftMach, GPU_R(0.0)), -GPU_R(1.0))
       *minimum(maximum(rightMach, GPU_R(0.0)), GPU_R(1.0));

    const GpuReal densitySum = left.rho + right.rho;
    if
    (
        !finiteScalar(densitySum)
     || densitySum < maximum(GPU_R(2.0)*rhoFloor, GPU_REAL_MIN)
    )
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }
    const GpuReal densityWeightedNormalSpeed =
       (
           left.rho*absolute(leftNormalVelocity)
         + right.rho*absolute(rightNormalVelocity)
       )/densitySum;
    const GpuReal leftNormalSpeed =
        (GPU_R(1.0) - densitySwitch)*densityWeightedNormalSpeed
      + densitySwitch*absolute(leftNormalVelocity);
    const GpuReal rightNormalSpeed =
        (GPU_R(1.0) - densitySwitch)*densityWeightedNormalSpeed
      + densitySwitch*absolute(rightNormalVelocity);
    const GpuReal massFlux = GPU_R(0.5)*
    (
        left.rho*(leftNormalVelocity + leftNormalSpeed)
      + right.rho*(rightNormalVelocity - rightNormalSpeed)
      - chi/interfaceSoundSpeed*(right.p - left.p)
    );

    GpuReal leftPressureWeight = GPU_R(0.0);
    if (absolute(leftMach) < GPU_R(1.0))
    {
        const GpuReal shifted = leftMach + GPU_R(1.0);
        leftPressureWeight =
            GPU_R(0.25)*(GPU_R(2.0) - leftMach)*shifted*shifted;
    }
    else if (leftMach >= GPU_R(0.0))
    {
        leftPressureWeight = GPU_R(1.0);
    }

    GpuReal rightPressureWeight = GPU_R(0.0);
    if (absolute(rightMach) < GPU_R(1.0))
    {
        const GpuReal shifted = rightMach - GPU_R(1.0);
        rightPressureWeight =
            GPU_R(0.25)*(GPU_R(2.0) + rightMach)*shifted*shifted;
    }
    else if (rightMach < GPU_R(0.0))
    {
        rightPressureWeight = GPU_R(1.0);
    }

    const GpuReal interfacePressure =
        GPU_R(0.5)*(left.p + right.p)
      + GPU_R(0.5)*(leftPressureWeight - rightPressureWeight)
       *(left.p - right.p)
      + velocityMagnitude*
       (leftPressureWeight + rightPressureWeight - GPU_R(1.0))
       *interfaceSoundSpeed*GPU_R(0.5)*densitySum;

    const Primitive& upwind = massFlux >= GPU_R(0.0) ? left : right;
    const GpuReal upwindVelocitySquared =
        massFlux >= GPU_R(0.0) ? leftVelocitySquared : rightVelocitySquared;
    const GpuReal upwindEnthalpy =
        gamma/(gamma - GPU_R(1.0))*upwind.p/upwind.rho
      + GPU_R(0.5)*upwindVelocitySquared;

    FluxResult result = invalidResult(evaluatedScheme);
    result.flux[0] = massFlux;
    result.flux[1] = massFlux*upwind.ux + interfacePressure*nx;
    result.flux[2] = massFlux*upwind.uy + interfacePressure*ny;
    result.flux[3] = massFlux*upwind.uz + interfacePressure*nz;
    result.flux[4] = massFlux*upwindEnthalpy;
    result.maxSignalSpeed = maximum
    (
        absolute(leftNormalVelocity) + leftSoundSpeed,
        absolute(rightNormalVelocity) + rightSoundSpeed
    );
    result.valid =
        finiteScalar(interfacePressure)
     && interfacePressure >= pressureFloor
     && finiteScalar(result.maxSignalSpeed)
     && finiteFive(result.flux);
    result.usedFallback = false;
    if (!result.valid)
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }
    return result;
}

UGKP_RIEMANN_HD FluxResult slau2FluxUnitNormal
(
    const Primitive& left,
    const Primitive& right,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal gamma,
    const GpuReal rhoFloor,
    const GpuReal pressureFloor
)
{
    const DensityGradient unusedGradient{GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)};
    return slau2FluxUnitNormalImpl
    (
        left,
        right,
        unusedGradient,
        unusedGradient,
        false,
        nx,
        ny,
        nz,
        gamma,
        rhoFloor,
        pressureFloor,
        Scheme::SLAU2
    );
}

UGKP_RIEMANN_HD FluxResult slau22FluxUnitNormal
(
    const Primitive& left,
    const Primitive& right,
    const DensityGradient& leftDensityGradient,
    const DensityGradient& rightDensityGradient,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal gamma,
    const GpuReal rhoFloor,
    const GpuReal pressureFloor
)
{
    return slau2FluxUnitNormalImpl
    (
        left,
        right,
        leftDensityGradient,
        rightDensityGradient,
        true,
        nx,
        ny,
        nz,
        gamma,
        rhoFloor,
        pressureFloor,
        Scheme::SLAU2_2
    );
}

UGKP_RIEMANN_HD bool makeHllcStarState
(
    const Primitive& side,
    const Conservative& sideConserved,
    const GpuReal sideSpeed,
    const GpuReal middleSpeed,
    const GpuReal starPressure,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    Conservative& starState
)
{
    const GpuReal sideNormalVelocity = normalVelocity(side, nx, ny, nz);
    const GpuReal denominator = sideSpeed - middleSpeed;
    const GpuReal scale = maximum
    (
        GPU_R(1.0),
        maximum(absolute(sideSpeed), absolute(middleSpeed))
    );
    if
    (
        !finiteScalar(denominator)
     || absolute(denominator) <= GPU_R(64.0)*GPU_REAL_EPSILON*scale
    )
    {
        return false;
    }

    const GpuReal densityRatio =
        (sideSpeed - sideNormalVelocity)/denominator;
    starState.q[0] = densityRatio*side.rho;
    starState.q[1] =
        densityRatio*side.rho*side.ux
      + (starPressure - side.p)*nx/denominator;
    starState.q[2] =
        densityRatio*side.rho*side.uy
      + (starPressure - side.p)*ny/denominator;
    starState.q[3] =
        densityRatio*side.rho*side.uz
      + (starPressure - side.p)*nz/denominator;
    starState.q[4] =
        densityRatio*sideConserved.q[4]
      - (side.p*sideNormalVelocity - starPressure*middleSpeed)/denominator;
    return finiteFive(starState.q);
}

UGKP_RIEMANN_HD FluxResult hllcFluxUnitNormal
(
    const Primitive& left,
    const Primitive& right,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal gamma,
    const GpuReal rhoFloor,
    const GpuReal pressureFloor
)
{
    const Conservative leftConserved = conservative(left, gamma);
    const Conservative rightConserved = conservative(right, gamma);
    GpuReal leftFlux[5];
    GpuReal rightFlux[5];
    projectedEulerFlux(left, leftConserved, nx, ny, nz, leftFlux);
    projectedEulerFlux(right, rightConserved, nx, ny, nz, rightFlux);

    const GpuReal leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const GpuReal rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const GpuReal leftNormalVelocity = normalVelocity(left, nx, ny, nz);
    const GpuReal rightNormalVelocity = normalVelocity(right, nx, ny, nz);
    const RoeAverage roe =
        makeRoeAverage(left, right, nx, ny, nz, gamma);
    if (!roe.valid)
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }

                                                              
    const GpuReal leftSpeed = minimum
    (
        roe.normalVelocity - roe.soundSpeed,
        leftNormalVelocity - leftSoundSpeed
    );
    const GpuReal rightSpeed = maximum
    (
        roe.normalVelocity + roe.soundSpeed,
        rightNormalVelocity + rightSoundSpeed
    );
    if (leftSpeed >= GPU_R(0.0))
    {
        FluxResult result = invalidResult(Scheme::HLLC);
        for (int component = 0; component < 5; ++component)
        {
            result.flux[component] = leftFlux[component];
        }
        result.maxSignalSpeed =
            maximum(absolute(leftSpeed), absolute(rightSpeed));
        result.valid =
            finiteScalar(result.maxSignalSpeed)
         && finiteFive(result.flux);
        return result;
    }
    if (rightSpeed <= GPU_R(0.0))
    {
        FluxResult result = invalidResult(Scheme::HLLC);
        for (int component = 0; component < 5; ++component)
        {
            result.flux[component] = rightFlux[component];
        }
        result.maxSignalSpeed =
            maximum(absolute(leftSpeed), absolute(rightSpeed));
        result.valid =
            finiteScalar(result.maxSignalSpeed)
         && finiteFive(result.flux);
        return result;
    }

    const GpuReal middleDenominator =
        right.rho*(rightSpeed - rightNormalVelocity)
      - left.rho*(leftSpeed - leftNormalVelocity);
    const GpuReal middleScale = maximum
    (
        GPU_R(1.0),
        maximum
        (
            absolute(right.rho*(rightSpeed - rightNormalVelocity)),
            absolute(left.rho*(leftSpeed - leftNormalVelocity))
        )
    );
    if
    (
        !finiteScalar(middleDenominator)
     || absolute(middleDenominator)
        <= GPU_R(64.0)*GPU_REAL_EPSILON*middleScale
    )
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }

    const GpuReal middleSpeed =
       (
           left.p - right.p
         - left.rho*leftNormalVelocity*
           (leftSpeed - leftNormalVelocity)
         + right.rho*rightNormalVelocity*
           (rightSpeed - rightNormalVelocity)
       )/middleDenominator;
    const GpuReal starPressure =
        right.rho*
        (rightNormalVelocity - rightSpeed)*
        (rightNormalVelocity - middleSpeed)
      + right.p;
    if
    (
        !finiteScalar(middleSpeed)
     || !finiteScalar(starPressure)
     || starPressure < pressureFloor
    )
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }

    Conservative leftStar;
    Conservative rightStar;
    if
    (
        !makeHllcStarState
        (
            left,
            leftConserved,
            leftSpeed,
            middleSpeed,
            starPressure,
            nx,
            ny,
            nz,
            leftStar
        )
     || !makeHllcStarState
        (
            right,
            rightConserved,
            rightSpeed,
            middleSpeed,
            starPressure,
            nx,
            ny,
            nz,
            rightStar
        )
     || !physicalConservative
        (
            leftStar, gamma, rhoFloor, pressureFloor
        )
     || !physicalConservative
        (
            rightStar, gamma, rhoFloor, pressureFloor
        )
    )
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }

    FluxResult result = invalidResult(Scheme::HLLC);
    if (leftSpeed >= GPU_R(0.0))
    {
        for (int component = 0; component < 5; ++component)
        {
            result.flux[component] = leftFlux[component];
        }
    }
    else if (rightSpeed <= GPU_R(0.0))
    {
        for (int component = 0; component < 5; ++component)
        {
            result.flux[component] = rightFlux[component];
        }
    }
    else
    {
        const Conservative& selectedStar =
            middleSpeed >= GPU_R(0.0) ? leftStar : rightStar;
        result.flux[0] = middleSpeed*selectedStar.q[0];
        result.flux[1] =
            middleSpeed*selectedStar.q[1] + starPressure*nx;
        result.flux[2] =
            middleSpeed*selectedStar.q[2] + starPressure*ny;
        result.flux[3] =
            middleSpeed*selectedStar.q[3] + starPressure*nz;
        result.flux[4] =
            middleSpeed*(selectedStar.q[4] + starPressure);
    }
    result.maxSignalSpeed = maximum
    (
        maximum(absolute(leftSpeed), absolute(rightSpeed)),
        absolute(middleSpeed)
    );
    result.valid =
        finiteScalar(result.maxSignalSpeed)
     && finiteFive(result.flux);
    result.usedFallback = false;
    if (!result.valid)
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }
    return result;
}

  
                                                              
  
                                                                      
                                                                       
                                                                            
                                                                          
                                                                         
                        
  
                                                                        
                                                                       
   
UGKP_RIEMANN_HD FluxResult hllcAdcFluxUnitNormal
(
    const Primitive& left,
    const Primitive& right,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal gamma,
    const GpuReal rhoFloor,
    const GpuReal pressureFloor,
    const GpuReal suppliedOmega
)
{
    const FluxResult hllc = hllcFluxUnitNormal
    (
        left,
        right,
        nx,
        ny,
        nz,
        gamma,
        rhoFloor,
        pressureFloor
    );
    if
    (
        !hllc.valid
     || hllc.usedFallback
     || hllc.evaluatedScheme != Scheme::HLLC
    )
    {
        return hllc;
    }

    const FluxResult hll = hlleFluxUnitNormal
    (
        left,
        right,
        nx,
        ny,
        nz,
        gamma,
        rhoFloor,
        pressureFloor,
        false
    );
    if
    (
        !hll.valid
     || hll.usedFallback
     || hll.evaluatedScheme != Scheme::HLLE
    )
    {
        return hll;
    }

    const GpuReal omega = minimum
    (
        GPU_R(1.0),
        maximum(GPU_R(0.0), finiteScalar(suppliedOmega) ? suppliedOmega : GPU_R(0.0))
    );
    FluxResult result = hllc;
    result.evaluatedScheme = Scheme::HLLC_ADC;
    result.maxSignalSpeed =
        maximum(hllc.maxSignalSpeed, hll.maxSignalSpeed);

    const GpuReal oneMinusOmega = GPU_R(1.0) - omega;
    result.flux[0] =
        hll.flux[0] + omega*(hllc.flux[0] - hll.flux[0]);

    const GpuReal normalMomentumCorrection =
       (hllc.flux[1] - hll.flux[1])*nx
     + (hllc.flux[2] - hll.flux[2])*ny
     + (hllc.flux[3] - hll.flux[3])*nz;
    result.flux[1] =
        hllc.flux[1] - oneMinusOmega*normalMomentumCorrection*nx;
    result.flux[2] =
        hllc.flux[2] - oneMinusOmega*normalMomentumCorrection*ny;
    result.flux[3] =
        hllc.flux[3] - oneMinusOmega*normalMomentumCorrection*nz;
                                                                             
    result.flux[4] = hllc.flux[4];

    result.valid =
        finiteScalar(result.maxSignalSpeed)
     && finiteFive(result.flux);
    result.usedFallback = false;
    if (!result.valid)
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }
    return result;
}

UGKP_RIEMANN_HD FluxResult roeFluxUnitNormal
(
    const Primitive& left,
    const Primitive& right,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal gamma,
    const GpuReal rhoFloor,
    const GpuReal pressureFloor
)
{
    const RoeAverage roe =
        makeRoeAverage(left, right, nx, ny, nz, gamma);
    if (!roe.valid)
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }

    const Conservative leftConserved = conservative(left, gamma);
    const Conservative rightConserved = conservative(right, gamma);
    GpuReal leftFlux[5];
    GpuReal rightFlux[5];
    projectedEulerFlux(left, leftConserved, nx, ny, nz, leftFlux);
    projectedEulerFlux(right, rightConserved, nx, ny, nz, rightFlux);
    const GpuReal leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const GpuReal rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const GpuReal leftNormalVelocity = normalVelocity(left, nx, ny, nz);
    const GpuReal rightNormalVelocity = normalVelocity(right, nx, ny, nz);

    const GpuReal soundSpeedSquared = roe.soundSpeed*roe.soundSpeed;
    const GpuReal densityJump = right.rho - left.rho;
    const GpuReal pressureJump = right.p - left.p;
    const GpuReal velocityJumpX = right.ux - left.ux;
    const GpuReal velocityJumpY = right.uy - left.uy;
    const GpuReal velocityJumpZ = right.uz - left.uz;
    const GpuReal normalVelocityJump =
        velocityJumpX*nx + velocityJumpY*ny + velocityJumpZ*nz;
    const GpuReal tangentialVelocityJumpX =
        velocityJumpX - normalVelocityJump*nx;
    const GpuReal tangentialVelocityJumpY =
        velocityJumpY - normalVelocityJump*ny;
    const GpuReal tangentialVelocityJumpZ =
        velocityJumpZ - normalVelocityJump*nz;

    const GpuReal acousticMinusStrength =
       (
           pressureJump
         - roe.density*roe.soundSpeed*normalVelocityJump
       )/(GPU_R(2.0)*soundSpeedSquared);
    const GpuReal acousticPlusStrength =
       (
           pressureJump
         + roe.density*roe.soundSpeed*normalVelocityJump
       )/(GPU_R(2.0)*soundSpeedSquared);
    const GpuReal contactStrength =
        densityJump - pressureJump/soundSpeedSquared;

    const GpuReal leftAcousticMinus =
        leftNormalVelocity - leftSoundSpeed;
    const GpuReal rightAcousticMinus =
        rightNormalVelocity - rightSoundSpeed;
    const GpuReal roeAcousticMinus =
        roe.normalVelocity - roe.soundSpeed;
    const GpuReal minusSpread =
        rightAcousticMinus - leftAcousticMinus;
    GpuReal lambdaMinus = absolute(roeAcousticMinus);
    if
    (
        leftAcousticMinus < GPU_R(0.0)
     && rightAcousticMinus > GPU_R(0.0)
     && minusSpread > GPU_REAL_MIN
     && lambdaMinus < minusSpread
    )
    {
        lambdaMinus =
            GPU_R(0.5)*
           (
               roeAcousticMinus*roeAcousticMinus/minusSpread
             + minusSpread
           );
    }
                                                                           
                                      
    const GpuReal lambdaContact = absolute(roe.normalVelocity);
    const GpuReal leftAcousticPlus =
        leftNormalVelocity + leftSoundSpeed;
    const GpuReal rightAcousticPlus =
        rightNormalVelocity + rightSoundSpeed;
    const GpuReal roeAcousticPlus =
        roe.normalVelocity + roe.soundSpeed;
    const GpuReal plusSpread =
        rightAcousticPlus - leftAcousticPlus;
    GpuReal lambdaPlus = absolute(roeAcousticPlus);
    if
    (
        leftAcousticPlus < GPU_R(0.0)
     && rightAcousticPlus > GPU_R(0.0)
     && plusSpread > GPU_REAL_MIN
     && lambdaPlus < plusSpread
    )
    {
        lambdaPlus =
            GPU_R(0.5)*
           (
               roeAcousticPlus*roeAcousticPlus/plusSpread
             + plusSpread
           );
    }

                                                                        
                                                                     
    Conservative leftAcousticState = leftConserved;
    Conservative rightAcousticState = rightConserved;
    const GpuReal roeVelocitySquared =
        roe.ux*roe.ux + roe.uy*roe.uy + roe.uz*roe.uz;
    const GpuReal minusMomentumX = roe.ux - roe.soundSpeed*nx;
    const GpuReal minusMomentumY = roe.uy - roe.soundSpeed*ny;
    const GpuReal minusMomentumZ = roe.uz - roe.soundSpeed*nz;
    const GpuReal minusEnergy =
        roe.enthalpy - roe.soundSpeed*roe.normalVelocity;
    const GpuReal plusMomentumX = roe.ux + roe.soundSpeed*nx;
    const GpuReal plusMomentumY = roe.uy + roe.soundSpeed*ny;
    const GpuReal plusMomentumZ = roe.uz + roe.soundSpeed*nz;
    const GpuReal plusEnergy =
        roe.enthalpy + roe.soundSpeed*roe.normalVelocity;

    leftAcousticState.q[0] += acousticMinusStrength;
    leftAcousticState.q[1] += acousticMinusStrength*minusMomentumX;
    leftAcousticState.q[2] += acousticMinusStrength*minusMomentumY;
    leftAcousticState.q[3] += acousticMinusStrength*minusMomentumZ;
    leftAcousticState.q[4] += acousticMinusStrength*minusEnergy;

    rightAcousticState.q[0] -= acousticPlusStrength;
    rightAcousticState.q[1] -= acousticPlusStrength*plusMomentumX;
    rightAcousticState.q[2] -= acousticPlusStrength*plusMomentumY;
    rightAcousticState.q[3] -= acousticPlusStrength*plusMomentumZ;
    rightAcousticState.q[4] -= acousticPlusStrength*plusEnergy;
    if
    (
        !physicalConservative
        (
            leftAcousticState, gamma, rhoFloor, pressureFloor
        )
     || !physicalConservative
        (
            rightAcousticState, gamma, rhoFloor, pressureFloor
        )
    )
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }

    const GpuReal tangentialEnergyJump =
        roe.density*
       (
           roe.ux*tangentialVelocityJumpX
         + roe.uy*tangentialVelocityJumpY
         + roe.uz*tangentialVelocityJumpZ
       );
    GpuReal dissipation[5];
    dissipation[0] =
        lambdaMinus*acousticMinusStrength
      + lambdaContact*contactStrength
      + lambdaPlus*acousticPlusStrength;
    dissipation[1] =
        lambdaMinus*acousticMinusStrength*minusMomentumX
      + lambdaContact*contactStrength*roe.ux
      + lambdaContact*roe.density*tangentialVelocityJumpX
      + lambdaPlus*acousticPlusStrength*plusMomentumX;
    dissipation[2] =
        lambdaMinus*acousticMinusStrength*minusMomentumY
      + lambdaContact*contactStrength*roe.uy
      + lambdaContact*roe.density*tangentialVelocityJumpY
      + lambdaPlus*acousticPlusStrength*plusMomentumY;
    dissipation[3] =
        lambdaMinus*acousticMinusStrength*minusMomentumZ
      + lambdaContact*contactStrength*roe.uz
      + lambdaContact*roe.density*tangentialVelocityJumpZ
      + lambdaPlus*acousticPlusStrength*plusMomentumZ;
    dissipation[4] =
        lambdaMinus*acousticMinusStrength*minusEnergy
      + lambdaContact*contactStrength*GPU_R(0.5)*roeVelocitySquared
      + lambdaContact*tangentialEnergyJump
      + lambdaPlus*acousticPlusStrength*plusEnergy;

    FluxResult result = invalidResult(Scheme::Roe);
    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] =
            GPU_R(0.5)*(leftFlux[component] + rightFlux[component])
          - GPU_R(0.5)*dissipation[component];
    }
    result.maxSignalSpeed =
        absolute(roe.normalVelocity) + roe.soundSpeed;
    result.valid =
        finiteScalar(result.maxSignalSpeed)
     && finiteFive(result.flux);
    result.usedFallback = false;
    if (!result.valid)
    {
        return hlleFluxUnitNormal
        (
            left,
            right,
            nx,
            ny,
            nz,
            gamma,
            rhoFloor,
            pressureFloor,
            true
        );
    }
    return result;
}

  
                                                                           
                                                                         
                                                      
   
UGKP_RIEMANN_HD FluxResult fluxUnitArea
(
    const Primitive& left,
    const Primitive& right,
    const GpuReal normalX,
    const GpuReal normalY,
    const GpuReal normalZ,
    const GpuReal gamma,
    const Scheme scheme,
    const GpuReal rhoFloor = GPU_R(1.0e-14),
    const GpuReal pressureFloor = GPU_R(1.0e-12),
    const GpuReal hllcAdcOmega = GPU_R(1.0)
)
{
    if
    (
        !physicalPrimitive(left, gamma, rhoFloor, pressureFloor)
     || !physicalPrimitive(right, gamma, rhoFloor, pressureFloor)
    )
    {
        return invalidResult(scheme);
    }

    GpuReal nx;
    GpuReal ny;
    GpuReal nz;
    GpuReal magnitude;
    if
    (
        !normalise
        (
            normalX, normalY, normalZ, nx, ny, nz, magnitude
        )
    )
    {
        return invalidResult(scheme);
    }

    switch (scheme)
    {
        case Scheme::RusanovTadmor:
            return rusanovTadmorFluxUnitNormal
            (
                left, right, nx, ny, nz, gamma, false
            );
        case Scheme::HllKurganov:
            return hllKurganovFluxUnitNormal
            (
                left, right, nx, ny, nz, gamma, false
            );
        case Scheme::HLLE:
            return hlleFluxUnitNormal
            (
                left,
                right,
                nx,
                ny,
                nz,
                gamma,
                rhoFloor,
                pressureFloor,
                false
            );
        case Scheme::HLLC:
            return hllcFluxUnitNormal
            (
                left,
                right,
                nx,
                ny,
                nz,
                gamma,
                rhoFloor,
                pressureFloor
            );
        case Scheme::Roe:
            return roeFluxUnitNormal
            (
                left,
                right,
                nx,
                ny,
                nz,
                gamma,
                rhoFloor,
                pressureFloor
            );
        case Scheme::HLLEM:
            return hllemFluxUnitNormal
            (
                left,
                right,
                nx,
                ny,
                nz,
                gamma,
                rhoFloor,
                pressureFloor
            );
        case Scheme::HLLC_ADC:
            return hllcAdcFluxUnitNormal
            (
                left,
                right,
                nx,
                ny,
                nz,
                gamma,
                rhoFloor,
                pressureFloor,
                hllcAdcOmega
            );
        case Scheme::SLAU2:
            return slau2FluxUnitNormal
            (
                left,
                right,
                nx,
                ny,
                nz,
                gamma,
                rhoFloor,
                pressureFloor
            );
        case Scheme::SLAU2_2:
            return invalidResult(Scheme::SLAU2_2);
        default:
            return rusanovTadmorFluxUnitNormal
            (
                left, right, nx, ny, nz, gamma, true
            );
    }
}

UGKP_RIEMANN_HD FluxResult fluxAreaVector
(
    const Primitive& left,
    const Primitive& right,
    const GpuReal areaX,
    const GpuReal areaY,
    const GpuReal areaZ,
    const GpuReal gamma,
    const Scheme scheme,
    const GpuReal rhoFloor = GPU_R(1.0e-14),
    const GpuReal pressureFloor = GPU_R(1.0e-12),
    const GpuReal hllcAdcOmega = GPU_R(1.0)
)
{
    GpuReal nx;
    GpuReal ny;
    GpuReal nz;
    GpuReal area;
    if (!normalise(areaX, areaY, areaZ, nx, ny, nz, area))
    {
        return invalidResult(scheme);
    }
    if (!finiteScalar(area))
    {
        return invalidResult(scheme);
    }

    FluxResult result = fluxUnitArea
    (
        left,
        right,
        nx,
        ny,
        nz,
        gamma,
        scheme,
        rhoFloor,
        pressureFloor,
        hllcAdcOmega
    );
    if (!result.valid)
    {
        return result;
    }
    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] *= area;
    }
    if (!finiteFive(result.flux))
    {
        result.valid = false;
    }
    return result;
}

inline const char* schemeName(const Scheme scheme)
{
    switch (scheme)
    {
        case Scheme::RusanovTadmor:
            return "RusanovTadmor";
        case Scheme::HllKurganov:
            return "HllKurganov";
        case Scheme::HLLE:
            return "HLLE";
        case Scheme::HLLC:
            return "HLLC";
        case Scheme::Roe:
            return "Roe";
        case Scheme::HLLEM:
            return "HLLEM";
        case Scheme::HLLC_ADC:
            return "HLLC_ADC";
        case Scheme::SLAU2:
            return "SLAU2";
        case Scheme::SLAU2_2:
            return "SLAU2_2";
        default:
            return "Unknown";
    }
}

}                         

#undef UGKP_RIEMANN_HD
