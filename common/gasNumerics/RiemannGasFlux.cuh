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
    double rho;
    double ux;
    double uy;
    double uz;
    double p;
};

struct DensityGradient
{
    double x;
    double y;
    double z;
};

struct Conservative
{
    double q[5];
};

struct FluxResult
{
    double flux[5];
    double maxSignalSpeed;
    Scheme evaluatedScheme;
    bool valid;
    bool usedFallback;
};

struct RoeAverage
{
    double ux;
    double uy;
    double uz;
    double enthalpy;
    double soundSpeed;
    double normalVelocity;
    double density;
    bool valid;
};

UGKP_RIEMANN_HD double minimum(const double a, const double b)
{
    return a < b ? a : b;
}

UGKP_RIEMANN_HD double maximum(const double a, const double b)
{
    return a > b ? a : b;
}

UGKP_RIEMANN_HD double absolute(const double a)
{
    return a < 0.0 ? -a : a;
}

UGKP_RIEMANN_HD bool finiteScalar(const double value)
{
    return value == value && value <= DBL_MAX && value >= -DBL_MAX;
}

UGKP_RIEMANN_HD bool finiteFive(const double values[5])
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
        result.flux[component] = 0.0;
    }
    result.maxSignalSpeed = 0.0;
    result.evaluatedScheme = scheme;
    result.valid = false;
    result.usedFallback = false;
    return result;
}

UGKP_RIEMANN_HD bool normalise
(
    const double normalX,
    const double normalY,
    const double normalZ,
    double& nx,
    double& ny,
    double& nz,
    double& magnitude
)
{
    const double scale = maximum
    (
        absolute(normalX),
        maximum(absolute(normalY), absolute(normalZ))
    );
    if (!finiteScalar(scale) || scale <= 0.0)
    {
        nx = 0.0;
        ny = 0.0;
        nz = 0.0;
        magnitude = 0.0;
        return false;
    }

    const double scaledX = normalX/scale;
    const double scaledY = normalY/scale;
    const double scaledZ = normalZ/scale;
    const double scaledMagnitude = ::sqrt
    (
        scaledX*scaledX + scaledY*scaledY + scaledZ*scaledZ
    );
    if (!finiteScalar(scaledMagnitude) || scaledMagnitude <= 0.0)
    {
        nx = 0.0;
        ny = 0.0;
        nz = 0.0;
        magnitude = 0.0;
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
    const double gamma,
    const double rhoFloor,
    const double pressureFloor
)
{
    return
        finiteScalar(gamma)
     && gamma > 1.0
     && finiteScalar(rhoFloor)
     && rhoFloor > 0.0
     && finiteScalar(pressureFloor)
     && pressureFloor > 0.0
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
    const double gamma
)
{
    Conservative result;
    result.q[0] = state.rho;
    result.q[1] = state.rho*state.ux;
    result.q[2] = state.rho*state.uy;
    result.q[3] = state.rho*state.uz;
    result.q[4] =
        state.p/(gamma - 1.0)
      + 0.5*state.rho*
       (
           state.ux*state.ux
         + state.uy*state.uy
         + state.uz*state.uz
       );
    return result;
}

UGKP_RIEMANN_HD double totalEnthalpy
(
    const Primitive& state,
    const double gamma
)
{
    const double velocitySquared =
        state.ux*state.ux + state.uy*state.uy + state.uz*state.uz;
    return
        gamma*state.p/(state.rho*(gamma - 1.0))
      + 0.5*velocitySquared;
}

UGKP_RIEMANN_HD double normalVelocity
(
    const Primitive& state,
    const double nx,
    const double ny,
    const double nz
)
{
    return state.ux*nx + state.uy*ny + state.uz*nz;
}

UGKP_RIEMANN_HD void projectedEulerFlux
(
    const Primitive& state,
    const Conservative& conserved,
    const double nx,
    const double ny,
    const double nz,
    double flux[5]
)
{
    const double un = normalVelocity(state, nx, ny, nz);
    flux[0] = state.rho*un;
    flux[1] = state.rho*state.ux*un + state.p*nx;
    flux[2] = state.rho*state.uy*un + state.p*ny;
    flux[3] = state.rho*state.uz*un + state.p*nz;
    flux[4] = (conserved.q[4] + state.p)*un;
}

UGKP_RIEMANN_HD bool physicalConservative
(
    const Conservative& state,
    const double gamma,
    const double rhoFloor,
    const double pressureFloor
)
{
    if
    (
        !finiteScalar(gamma)
     || gamma <= 1.0
     || !finiteScalar(rhoFloor)
     || rhoFloor <= 0.0
     || !finiteScalar(pressureFloor)
     || pressureFloor <= 0.0
     || !finiteFive(state.q)
     || state.q[0] < rhoFloor
    )
    {
        return false;
    }

    const double inverseDensity = 1.0/state.q[0];
    const double kineticEnergy =
        0.5*
       (
           state.q[1]*state.q[1]
         + state.q[2]*state.q[2]
         + state.q[3]*state.q[3]
       )*inverseDensity;
    const double pressure = (gamma - 1.0)*(state.q[4] - kineticEnergy);
    return finiteScalar(pressure) && pressure >= pressureFloor;
}

UGKP_RIEMANN_HD RoeAverage makeRoeAverage
(
    const Primitive& left,
    const Primitive& right,
    const double nx,
    const double ny,
    const double nz,
    const double gamma
)
{
    RoeAverage average;
    average.valid = false;
    average.ux = 0.0;
    average.uy = 0.0;
    average.uz = 0.0;
    average.enthalpy = 0.0;
    average.soundSpeed = 0.0;
    average.normalVelocity = 0.0;
    average.density = 0.0;

    const double sqrtLeftDensity = ::sqrt(left.rho);
    const double sqrtRightDensity = ::sqrt(right.rho);
    const double denominator = sqrtLeftDensity + sqrtRightDensity;
    if (!finiteScalar(denominator) || denominator <= DBL_MIN)
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

    const double velocitySquared =
        average.ux*average.ux
      + average.uy*average.uy
      + average.uz*average.uz;
    const double soundSpeedSquared =
        (gamma - 1.0)*(average.enthalpy - 0.5*velocitySquared);
    if (!finiteScalar(soundSpeedSquared) || soundSpeedSquared <= DBL_MIN)
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
    const double nx,
    const double ny,
    const double nz,
    const double gamma,
    const bool inheritedFallback
)
{
    FluxResult result = invalidResult(Scheme::RusanovTadmor);
    const Conservative leftConserved = conservative(left, gamma);
    const Conservative rightConserved = conservative(right, gamma);
    double leftFlux[5];
    double rightFlux[5];
    projectedEulerFlux(left, leftConserved, nx, ny, nz, leftFlux);
    projectedEulerFlux(right, rightConserved, nx, ny, nz, rightFlux);

    const double leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const double rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const double leftNormalVelocity = normalVelocity(left, nx, ny, nz);
    const double rightNormalVelocity = normalVelocity(right, nx, ny, nz);
    const double signalSpeed = maximum
    (
        absolute(leftNormalVelocity) + leftSoundSpeed,
        absolute(rightNormalVelocity) + rightSoundSpeed
    );

    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] =
            0.5*(leftFlux[component] + rightFlux[component])
          - 0.5*signalSpeed*
            (rightConserved.q[component] - leftConserved.q[component]);
    }
    result.maxSignalSpeed = signalSpeed;
    result.valid =
        finiteScalar(signalSpeed)
     && signalSpeed >= 0.0
     && finiteFive(result.flux);
    result.usedFallback = inheritedFallback;
    return result;
}

UGKP_RIEMANN_HD FluxResult hllKurganovFluxUnitNormal
(
    const Primitive& left,
    const Primitive& right,
    const double nx,
    const double ny,
    const double nz,
    const double gamma,
    const bool inheritedFallback
)
{
    const Conservative leftConserved = conservative(left, gamma);
    const Conservative rightConserved = conservative(right, gamma);
    double leftFlux[5];
    double rightFlux[5];
    projectedEulerFlux(left, leftConserved, nx, ny, nz, leftFlux);
    projectedEulerFlux(right, rightConserved, nx, ny, nz, rightFlux);

    const double leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const double rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const double leftNormalVelocity = normalVelocity(left, nx, ny, nz);
    const double rightNormalVelocity = normalVelocity(right, nx, ny, nz);
    const double positiveSpeed = maximum
    (
        0.0,
        maximum
        (
            leftNormalVelocity + leftSoundSpeed,
            rightNormalVelocity + rightSoundSpeed
        )
    );
    const double negativeSpeed = minimum
    (
        0.0,
        minimum
        (
            leftNormalVelocity - leftSoundSpeed,
            rightNormalVelocity - rightSoundSpeed
        )
    );
    const double denominator = positiveSpeed - negativeSpeed;
    const double scale =
        maximum
        (
            1.0,
            maximum(absolute(positiveSpeed), absolute(negativeSpeed))
        );
    if
    (
        !finiteScalar(denominator)
     || denominator <= 64.0*DBL_EPSILON*scale
    )
    {
        return rusanovTadmorFluxUnitNormal
        (
            left, right, nx, ny, nz, gamma, true
        );
    }

    FluxResult result = invalidResult(Scheme::HllKurganov);
    const double diffusion =
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
    const double nx,
    const double ny,
    const double nz,
    const double gamma,
    const double rhoFloor,
    const double pressureFloor,
    const bool inheritedFallback
)
{
    const Conservative leftConserved = conservative(left, gamma);
    const Conservative rightConserved = conservative(right, gamma);
    double leftFlux[5];
    double rightFlux[5];
    projectedEulerFlux(left, leftConserved, nx, ny, nz, leftFlux);
    projectedEulerFlux(right, rightConserved, nx, ny, nz, rightFlux);

    const double leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const double rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const double leftNormalVelocity = normalVelocity(left, nx, ny, nz);
    const double rightNormalVelocity = normalVelocity(right, nx, ny, nz);
    const RoeAverage roe =
        makeRoeAverage(left, right, nx, ny, nz, gamma);

    double negativeSpeed = minimum
    (
        0.0,
        minimum
        (
            leftNormalVelocity - leftSoundSpeed,
            rightNormalVelocity - rightSoundSpeed
        )
    );
    double positiveSpeed = maximum
    (
        0.0,
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

    if (negativeSpeed >= 0.0)
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
    if (positiveSpeed <= 0.0)
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

    const double denominator = positiveSpeed - negativeSpeed;
    const double scale =
        maximum
        (
            1.0,
            maximum(absolute(positiveSpeed), absolute(negativeSpeed))
        );
    if
    (
        !finiteScalar(denominator)
     || denominator <= 64.0*DBL_EPSILON*scale
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
    const double nx,
    const double ny,
    const double nz,
    const double gamma,
    const double rhoFloor,
    const double pressureFloor
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

    const double leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const double rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const double leftNormalVelocity = normalVelocity(left, nx, ny, nz);
    const double rightNormalVelocity = normalVelocity(right, nx, ny, nz);
    const RoeAverage roe =
        makeRoeAverage(left, right, nx, ny, nz, gamma);
    if (!roe.valid)
    {
        return result;
    }

    double negativeSpeed = minimum
    (
        0.0,
        minimum
        (
            leftNormalVelocity - leftSoundSpeed,
            rightNormalVelocity - rightSoundSpeed
        )
    );
    double positiveSpeed = maximum
    (
        0.0,
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
    if (negativeSpeed >= 0.0 || positiveSpeed <= 0.0)
    {
        result.evaluatedScheme = Scheme::HLLEM;
        return result;
    }

    const double denominator = positiveSpeed - negativeSpeed;
    const double scale = maximum
    (
        1.0,
        maximum(absolute(positiveSpeed), absolute(negativeSpeed))
    );
    if
    (
        !finiteScalar(denominator)
     || denominator <= 64.0*DBL_EPSILON*scale
    )
    {
        return result;
    }

    const double densityJump = right.rho - left.rho;
    const double pressureJump = right.p - left.p;
    const double velocityJumpX = right.ux - left.ux;
    const double velocityJumpY = right.uy - left.uy;
    const double velocityJumpZ = right.uz - left.uz;
    const double normalVelocityJump =
        velocityJumpX*nx + velocityJumpY*ny + velocityJumpZ*nz;
    const double tangentialVelocityJumpX =
        velocityJumpX - normalVelocityJump*nx;
    const double tangentialVelocityJumpY =
        velocityJumpY - normalVelocityJump*ny;
    const double tangentialVelocityJumpZ =
        velocityJumpZ - normalVelocityJump*nz;
    const double soundSpeedSquared = roe.soundSpeed*roe.soundSpeed;
    const double contactStrength =
        densityJump - pressureJump/soundSpeedSquared;
    const double roeVelocitySquared =
        roe.ux*roe.ux + roe.uy*roe.uy + roe.uz*roe.uz;

    double linearlyDegenerateJump[5];
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
        0.5*contactStrength*roeVelocitySquared
      + roe.density*
       (
           roe.ux*tangentialVelocityJumpX
         + roe.uy*tangentialVelocityJumpY
         + roe.uz*tangentialVelocityJumpZ
       );

                                                                       
                                                                        
                                                                       
                                                                
                                                   
    const double antidiffusion = minimum
    (
        1.0,
        maximum
        (
            0.0,
            roe.normalVelocity >= 0.0
          ? (positiveSpeed - roe.normalVelocity)/positiveSpeed
          : (roe.normalVelocity - negativeSpeed)/(-negativeSpeed)
        )
    );
    const double correction =
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
    const double nx,
    const double ny,
    const double nz,
    const double gamma,
    const double rhoFloor,
    const double pressureFloor,
    const Scheme evaluatedScheme
)
{
    const double leftNormalVelocity =
        normalVelocity(left, nx, ny, nz);
    const double rightNormalVelocity =
        normalVelocity(right, nx, ny, nz);
    const double leftVelocitySquared =
        left.ux*left.ux + left.uy*left.uy + left.uz*left.uz;
    const double rightVelocitySquared =
        right.ux*right.ux + right.uy*right.uy + right.uz*right.uz;
    const double leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const double rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const double interfaceSoundSpeed =
        0.5*(leftSoundSpeed + rightSoundSpeed);
    const double soundScale = maximum
    (
        1.0,
        maximum(leftSoundSpeed, rightSoundSpeed)
    );
    if
    (
        !finiteScalar(interfaceSoundSpeed)
     || interfaceSoundSpeed <= 64.0*DBL_EPSILON*soundScale
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

    const double leftMach = leftNormalVelocity/interfaceSoundSpeed;
    const double rightMach = rightNormalVelocity/interfaceSoundSpeed;
    const double velocityMagnitude = ::sqrt
    (
        0.5*(leftVelocitySquared + rightVelocitySquared)
    );
    double dampingMach = velocityMagnitude/interfaceSoundSpeed;
    if (densityGradientAlignedDamping)
    {
        const double leftGradientMagnitude = ::sqrt
        (
            leftDensityGradient.x*leftDensityGradient.x
          + leftDensityGradient.y*leftDensityGradient.y
          + leftDensityGradient.z*leftDensityGradient.z
        );
        const double rightGradientMagnitude = ::sqrt
        (
            rightDensityGradient.x*rightDensityGradient.x
          + rightDensityGradient.y*rightDensityGradient.y
          + rightDensityGradient.z*rightDensityGradient.z
        );
        const double gradientScale = maximum
        (
            1.0,
            maximum(leftGradientMagnitude, rightGradientMagnitude)
        );
        const double gradientFloor = 64.0*DBL_EPSILON*gradientScale;
        const double leftGradientMach =
            leftGradientMagnitude > gradientFloor
          ? (
                left.ux*leftDensityGradient.x
              + left.uy*leftDensityGradient.y
              + left.uz*leftDensityGradient.z
            )/(leftGradientMagnitude*interfaceSoundSpeed)
          : velocityMagnitude/interfaceSoundSpeed;
        const double rightGradientMach =
            rightGradientMagnitude > gradientFloor
          ? (
                right.ux*rightDensityGradient.x
              + right.uy*rightDensityGradient.y
              + right.uz*rightDensityGradient.z
            )/(rightGradientMagnitude*interfaceSoundSpeed)
          : velocityMagnitude/interfaceSoundSpeed;
        dampingMach = ::sqrt
        (
            0.5*
            (
                leftGradientMach*leftGradientMach
              + rightGradientMach*rightGradientMach
            )
        );
    }
    const double limitedMach = minimum(1.0, dampingMach);
    const double chi = (1.0 - limitedMach)*(1.0 - limitedMach);
    const double densitySwitch =
       -maximum(minimum(leftMach, 0.0), -1.0)
       *minimum(maximum(rightMach, 0.0), 1.0);

    const double densitySum = left.rho + right.rho;
    if
    (
        !finiteScalar(densitySum)
     || densitySum < maximum(2.0*rhoFloor, DBL_MIN)
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
    const double densityWeightedNormalSpeed =
       (
           left.rho*absolute(leftNormalVelocity)
         + right.rho*absolute(rightNormalVelocity)
       )/densitySum;
    const double leftNormalSpeed =
        (1.0 - densitySwitch)*densityWeightedNormalSpeed
      + densitySwitch*absolute(leftNormalVelocity);
    const double rightNormalSpeed =
        (1.0 - densitySwitch)*densityWeightedNormalSpeed
      + densitySwitch*absolute(rightNormalVelocity);
    const double massFlux = 0.5*
    (
        left.rho*(leftNormalVelocity + leftNormalSpeed)
      + right.rho*(rightNormalVelocity - rightNormalSpeed)
      - chi/interfaceSoundSpeed*(right.p - left.p)
    );

    double leftPressureWeight = 0.0;
    if (absolute(leftMach) < 1.0)
    {
        const double shifted = leftMach + 1.0;
        leftPressureWeight =
            0.25*(2.0 - leftMach)*shifted*shifted;
    }
    else if (leftMach >= 0.0)
    {
        leftPressureWeight = 1.0;
    }

    double rightPressureWeight = 0.0;
    if (absolute(rightMach) < 1.0)
    {
        const double shifted = rightMach - 1.0;
        rightPressureWeight =
            0.25*(2.0 + rightMach)*shifted*shifted;
    }
    else if (rightMach < 0.0)
    {
        rightPressureWeight = 1.0;
    }

    const double interfacePressure =
        0.5*(left.p + right.p)
      + 0.5*(leftPressureWeight - rightPressureWeight)
       *(left.p - right.p)
      + velocityMagnitude*
       (leftPressureWeight + rightPressureWeight - 1.0)
       *interfaceSoundSpeed*0.5*densitySum;

    const Primitive& upwind = massFlux >= 0.0 ? left : right;
    const double upwindVelocitySquared =
        massFlux >= 0.0 ? leftVelocitySquared : rightVelocitySquared;
    const double upwindEnthalpy =
        gamma/(gamma - 1.0)*upwind.p/upwind.rho
      + 0.5*upwindVelocitySquared;

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
    const double nx,
    const double ny,
    const double nz,
    const double gamma,
    const double rhoFloor,
    const double pressureFloor
)
{
    const DensityGradient unusedGradient{0.0, 0.0, 0.0};
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
    const double nx,
    const double ny,
    const double nz,
    const double gamma,
    const double rhoFloor,
    const double pressureFloor
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
    const double sideSpeed,
    const double middleSpeed,
    const double starPressure,
    const double nx,
    const double ny,
    const double nz,
    Conservative& starState
)
{
    const double sideNormalVelocity = normalVelocity(side, nx, ny, nz);
    const double denominator = sideSpeed - middleSpeed;
    const double scale = maximum
    (
        1.0,
        maximum(absolute(sideSpeed), absolute(middleSpeed))
    );
    if
    (
        !finiteScalar(denominator)
     || absolute(denominator) <= 64.0*DBL_EPSILON*scale
    )
    {
        return false;
    }

    const double densityRatio =
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
    const double nx,
    const double ny,
    const double nz,
    const double gamma,
    const double rhoFloor,
    const double pressureFloor
)
{
    const Conservative leftConserved = conservative(left, gamma);
    const Conservative rightConserved = conservative(right, gamma);
    double leftFlux[5];
    double rightFlux[5];
    projectedEulerFlux(left, leftConserved, nx, ny, nz, leftFlux);
    projectedEulerFlux(right, rightConserved, nx, ny, nz, rightFlux);

    const double leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const double rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const double leftNormalVelocity = normalVelocity(left, nx, ny, nz);
    const double rightNormalVelocity = normalVelocity(right, nx, ny, nz);
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

                                                              
    const double leftSpeed = minimum
    (
        roe.normalVelocity - roe.soundSpeed,
        leftNormalVelocity - leftSoundSpeed
    );
    const double rightSpeed = maximum
    (
        roe.normalVelocity + roe.soundSpeed,
        rightNormalVelocity + rightSoundSpeed
    );
    if (leftSpeed >= 0.0)
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
    if (rightSpeed <= 0.0)
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

    const double middleDenominator =
        right.rho*(rightSpeed - rightNormalVelocity)
      - left.rho*(leftSpeed - leftNormalVelocity);
    const double middleScale = maximum
    (
        1.0,
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
        <= 64.0*DBL_EPSILON*middleScale
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

    const double middleSpeed =
       (
           left.p - right.p
         - left.rho*leftNormalVelocity*
           (leftSpeed - leftNormalVelocity)
         + right.rho*rightNormalVelocity*
           (rightSpeed - rightNormalVelocity)
       )/middleDenominator;
    const double starPressure =
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
    if (leftSpeed >= 0.0)
    {
        for (int component = 0; component < 5; ++component)
        {
            result.flux[component] = leftFlux[component];
        }
    }
    else if (rightSpeed <= 0.0)
    {
        for (int component = 0; component < 5; ++component)
        {
            result.flux[component] = rightFlux[component];
        }
    }
    else
    {
        const Conservative& selectedStar =
            middleSpeed >= 0.0 ? leftStar : rightStar;
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
    const double nx,
    const double ny,
    const double nz,
    const double gamma,
    const double rhoFloor,
    const double pressureFloor,
    const double suppliedOmega
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

    const double omega = minimum
    (
        1.0,
        maximum(0.0, finiteScalar(suppliedOmega) ? suppliedOmega : 0.0)
    );
    FluxResult result = hllc;
    result.evaluatedScheme = Scheme::HLLC_ADC;
    result.maxSignalSpeed =
        maximum(hllc.maxSignalSpeed, hll.maxSignalSpeed);

    const double oneMinusOmega = 1.0 - omega;
    result.flux[0] =
        hll.flux[0] + omega*(hllc.flux[0] - hll.flux[0]);

    const double normalMomentumCorrection =
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
    const double nx,
    const double ny,
    const double nz,
    const double gamma,
    const double rhoFloor,
    const double pressureFloor
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
    double leftFlux[5];
    double rightFlux[5];
    projectedEulerFlux(left, leftConserved, nx, ny, nz, leftFlux);
    projectedEulerFlux(right, rightConserved, nx, ny, nz, rightFlux);
    const double leftSoundSpeed = ::sqrt(gamma*left.p/left.rho);
    const double rightSoundSpeed = ::sqrt(gamma*right.p/right.rho);
    const double leftNormalVelocity = normalVelocity(left, nx, ny, nz);
    const double rightNormalVelocity = normalVelocity(right, nx, ny, nz);

    const double soundSpeedSquared = roe.soundSpeed*roe.soundSpeed;
    const double densityJump = right.rho - left.rho;
    const double pressureJump = right.p - left.p;
    const double velocityJumpX = right.ux - left.ux;
    const double velocityJumpY = right.uy - left.uy;
    const double velocityJumpZ = right.uz - left.uz;
    const double normalVelocityJump =
        velocityJumpX*nx + velocityJumpY*ny + velocityJumpZ*nz;
    const double tangentialVelocityJumpX =
        velocityJumpX - normalVelocityJump*nx;
    const double tangentialVelocityJumpY =
        velocityJumpY - normalVelocityJump*ny;
    const double tangentialVelocityJumpZ =
        velocityJumpZ - normalVelocityJump*nz;

    const double acousticMinusStrength =
       (
           pressureJump
         - roe.density*roe.soundSpeed*normalVelocityJump
       )/(2.0*soundSpeedSquared);
    const double acousticPlusStrength =
       (
           pressureJump
         + roe.density*roe.soundSpeed*normalVelocityJump
       )/(2.0*soundSpeedSquared);
    const double contactStrength =
        densityJump - pressureJump/soundSpeedSquared;

    const double leftAcousticMinus =
        leftNormalVelocity - leftSoundSpeed;
    const double rightAcousticMinus =
        rightNormalVelocity - rightSoundSpeed;
    const double roeAcousticMinus =
        roe.normalVelocity - roe.soundSpeed;
    const double minusSpread =
        rightAcousticMinus - leftAcousticMinus;
    double lambdaMinus = absolute(roeAcousticMinus);
    if
    (
        leftAcousticMinus < 0.0
     && rightAcousticMinus > 0.0
     && minusSpread > DBL_MIN
     && lambdaMinus < minusSpread
    )
    {
        lambdaMinus =
            0.5*
           (
               roeAcousticMinus*roeAcousticMinus/minusSpread
             + minusSpread
           );
    }
                                                                           
                                      
    const double lambdaContact = absolute(roe.normalVelocity);
    const double leftAcousticPlus =
        leftNormalVelocity + leftSoundSpeed;
    const double rightAcousticPlus =
        rightNormalVelocity + rightSoundSpeed;
    const double roeAcousticPlus =
        roe.normalVelocity + roe.soundSpeed;
    const double plusSpread =
        rightAcousticPlus - leftAcousticPlus;
    double lambdaPlus = absolute(roeAcousticPlus);
    if
    (
        leftAcousticPlus < 0.0
     && rightAcousticPlus > 0.0
     && plusSpread > DBL_MIN
     && lambdaPlus < plusSpread
    )
    {
        lambdaPlus =
            0.5*
           (
               roeAcousticPlus*roeAcousticPlus/plusSpread
             + plusSpread
           );
    }

                                                                        
                                                                     
    Conservative leftAcousticState = leftConserved;
    Conservative rightAcousticState = rightConserved;
    const double roeVelocitySquared =
        roe.ux*roe.ux + roe.uy*roe.uy + roe.uz*roe.uz;
    const double minusMomentumX = roe.ux - roe.soundSpeed*nx;
    const double minusMomentumY = roe.uy - roe.soundSpeed*ny;
    const double minusMomentumZ = roe.uz - roe.soundSpeed*nz;
    const double minusEnergy =
        roe.enthalpy - roe.soundSpeed*roe.normalVelocity;
    const double plusMomentumX = roe.ux + roe.soundSpeed*nx;
    const double plusMomentumY = roe.uy + roe.soundSpeed*ny;
    const double plusMomentumZ = roe.uz + roe.soundSpeed*nz;
    const double plusEnergy =
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

    const double tangentialEnergyJump =
        roe.density*
       (
           roe.ux*tangentialVelocityJumpX
         + roe.uy*tangentialVelocityJumpY
         + roe.uz*tangentialVelocityJumpZ
       );
    double dissipation[5];
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
      + lambdaContact*contactStrength*0.5*roeVelocitySquared
      + lambdaContact*tangentialEnergyJump
      + lambdaPlus*acousticPlusStrength*plusEnergy;

    FluxResult result = invalidResult(Scheme::Roe);
    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] =
            0.5*(leftFlux[component] + rightFlux[component])
          - 0.5*dissipation[component];
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
    const double normalX,
    const double normalY,
    const double normalZ,
    const double gamma,
    const Scheme scheme,
    const double rhoFloor = 1.0e-14,
    const double pressureFloor = 1.0e-12,
    const double hllcAdcOmega = 1.0
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

    double nx;
    double ny;
    double nz;
    double magnitude;
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
    const double areaX,
    const double areaY,
    const double areaZ,
    const double gamma,
    const Scheme scheme,
    const double rhoFloor = 1.0e-14,
    const double pressureFloor = 1.0e-12,
    const double hllcAdcOmega = 1.0
)
{
    double nx;
    double ny;
    double nz;
    double area;
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
