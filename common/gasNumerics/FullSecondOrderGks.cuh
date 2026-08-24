#ifndef UGKP_FULL_SECOND_ORDER_GKS_CUH
#define UGKP_FULL_SECOND_ORDER_GKS_CUH

#define UGKP_FULL_SECOND_ORDER_GKS 1

#include <cmath>

#if defined(__CUDACC__)
#define UGKP_GKS_HD __host__ __device__ inline
#define UGKP_GKS_ENTRY __host__ __device__ __noinline__
#else
#define UGKP_GKS_HD inline
#define UGKP_GKS_ENTRY inline
#endif

namespace ugkpgks
{

static constexpr double small = 2.22044604925031308085e-16;
static constexpr double pi = 3.141592653589793238462643383279502884;

struct LocalPrimitive
{
    double rho;
    double u;
    double v;
    double w;
    double p;
    double T;
};

struct Monomial
{
    int u;
    int v;
    int w;
    int xi2;
    double coefficient;
};

struct MaxwellMomentCache
{
    double rho;
    double u[7];
    double v[7];
    double w[7];
    double xi2[4];
};

struct TimeCoefficients
{
    double equilibrium;
    double freeTransport;
    double timeExponential;
    double equilibriumTime;
};

struct FullGksResult
{
    double flux[5];
    double molecularViscousFlux[5];
    double molecularHeatFlux;
    LocalPrimitive interfaceState;
    bool valid;
};

UGKP_GKS_HD double absolute(const double x)
{
    return x < 0.0 ? -x : x;
}

UGKP_GKS_HD double maximum(const double a, const double b)
{
    return a > b ? a : b;
}

UGKP_GKS_HD bool finiteValue(const double x)
{
    return ::isfinite(x) != 0;
}

UGKP_GKS_HD void fullGaussianMoments
(
    const double mean,
    const double variance,
    double moments[7]
)
{
    moments[0] = 1.0;
    moments[1] = mean;
    for (int order = 2; order <= 6; ++order)
    {
        moments[order] =
            mean*moments[order - 1]
          + (order - 1)*variance*moments[order - 2];
    }
}

UGKP_GKS_HD void halfGaussianMoments
(
    const double mean,
    const double variance,
    const int halfSign,
    double moments[7]
)
{
    const double sigma = ::sqrt(maximum(variance, small));
    const double beta = mean/(::sqrt(2.0)*sigma);
    const double boundary =
        sigma/::sqrt(2.0*pi)*::exp(-beta*beta);

    if (halfSign > 0)
    {
        moments[0] = 0.5*::erfc(-beta);
        moments[1] = mean*moments[0] + boundary;
    }
    else
    {
        moments[0] = 0.5*::erfc(beta);
        moments[1] = mean*moments[0] - boundary;
    }

    for (int order = 2; order <= 6; ++order)
    {
        moments[order] =
            mean*moments[order - 1]
          + (order - 1)*variance*moments[order - 2];
    }
}

UGKP_GKS_HD MaxwellMomentCache makeMomentCache
(
    const LocalPrimitive& state,
    const double internalDegrees,
    const int halfSign
)
{
    MaxwellMomentCache cache;
    cache.rho = state.rho;
    const double variance = maximum(state.p/state.rho, small);

    if (halfSign == 0)
    {
        fullGaussianMoments(state.u, variance, cache.u);
    }
    else
    {
        halfGaussianMoments(state.u, variance, halfSign, cache.u);
    }
    fullGaussianMoments(state.v, variance, cache.v);
    fullGaussianMoments(state.w, variance, cache.w);

    cache.xi2[0] = 1.0;
    cache.xi2[1] = internalDegrees*variance;
    cache.xi2[2] =
        internalDegrees*(internalDegrees + 2.0)*variance*variance;
    cache.xi2[3] =
        internalDegrees*(internalDegrees + 2.0)*(internalDegrees + 4.0)
       *variance*variance*variance;
    return cache;
}

UGKP_GKS_HD double monomialMoment
(
    const MaxwellMomentCache& cache,
    const int u,
    const int v,
    const int w,
    const int xi2
)
{
    if
    (
        u < 0 || u > 6 || v < 0 || v > 6 || w < 0 || w > 6
     || xi2 < 0 || xi2 > 3
    )
    {
        return 0.0;
    }
    return cache.rho*cache.u[u]*cache.v[v]*cache.w[w]*cache.xi2[xi2];
}

UGKP_GKS_HD int basisTerms(const int basis, Monomial terms[4])
{
    if (basis >= 0 && basis <= 3)
    {
        terms[0].u = basis == 1 ? 1 : 0;
        terms[0].v = basis == 2 ? 1 : 0;
        terms[0].w = basis == 3 ? 1 : 0;
        terms[0].xi2 = 0;
        terms[0].coefficient = 1.0;
        return 1;
    }

    terms[0] = Monomial{2, 0, 0, 0, 0.5};
    terms[1] = Monomial{0, 2, 0, 0, 0.5};
    terms[2] = Monomial{0, 0, 2, 0, 0.5};
    terms[3] = Monomial{0, 0, 0, 1, 0.5};
    return 4;
}

UGKP_GKS_HD double basisProductMoment
(
    const MaxwellMomentCache& cache,
    const int leftBasis,
    const int rightBasis,
    const int extraU,
    const int extraV,
    const int extraW
)
{
    if
    (
        leftBasis < 0 || leftBasis > 4
     || rightBasis < 0 || rightBasis > 4
    )
    {
        return 0.0;
    }

    const bool leftEnergy = leftBasis == 4;
    const bool rightEnergy = rightBasis == 4;
    const int baseU =
        extraU + (leftBasis == 1 ? 1 : 0) + (rightBasis == 1 ? 1 : 0);
    const int baseV =
        extraV + (leftBasis == 2 ? 1 : 0) + (rightBasis == 2 ? 1 : 0);
    const int baseW =
        extraW + (leftBasis == 3 ? 1 : 0) + (rightBasis == 3 ? 1 : 0);

    if (!leftEnergy && !rightEnergy)
    {
        return monomialMoment(cache, baseU, baseV, baseW, 0);
    }

    if (leftEnergy != rightEnergy)
    {
        double value = 0.0;
        value += 0.5*monomialMoment(cache, baseU + 2, baseV, baseW, 0);
        value += 0.5*monomialMoment(cache, baseU, baseV + 2, baseW, 0);
        value += 0.5*monomialMoment(cache, baseU, baseV, baseW + 2, 0);
        value += 0.5*monomialMoment(cache, baseU, baseV, baseW, 1);
        return value;
    }

    double value = 0.0;
    value += 0.25*monomialMoment(cache, baseU + 4, baseV, baseW, 0);
    value += 0.25*monomialMoment(cache, baseU + 2, baseV + 2, baseW, 0);
    value += 0.25*monomialMoment(cache, baseU + 2, baseV, baseW + 2, 0);
    value += 0.25*monomialMoment(cache, baseU + 2, baseV, baseW, 1);
    value += 0.25*monomialMoment(cache, baseU + 2, baseV + 2, baseW, 0);
    value += 0.25*monomialMoment(cache, baseU, baseV + 4, baseW, 0);
    value += 0.25*monomialMoment(cache, baseU, baseV + 2, baseW + 2, 0);
    value += 0.25*monomialMoment(cache, baseU, baseV + 2, baseW, 1);
    value += 0.25*monomialMoment(cache, baseU + 2, baseV, baseW + 2, 0);
    value += 0.25*monomialMoment(cache, baseU, baseV + 2, baseW + 2, 0);
    value += 0.25*monomialMoment(cache, baseU, baseV, baseW + 4, 0);
    value += 0.25*monomialMoment(cache, baseU, baseV, baseW + 2, 1);
    value += 0.25*monomialMoment(cache, baseU + 2, baseV, baseW, 1);
    value += 0.25*monomialMoment(cache, baseU, baseV + 2, baseW, 1);
    value += 0.25*monomialMoment(cache, baseU, baseV, baseW + 2, 1);
    value += 0.25*monomialMoment(cache, baseU, baseV, baseW, 2);
    return value;
}

template<int U, int V, int W, int Xi2>
UGKP_GKS_HD double fixedMonomialMoment(const MaxwellMomentCache& cache)
{
    static_assert(U >= 0 && U <= 6, "fixed U moment is out of range");
    static_assert(V >= 0 && V <= 6, "fixed V moment is out of range");
    static_assert(W >= 0 && W <= 6, "fixed W moment is out of range");
    static_assert(Xi2 >= 0 && Xi2 <= 3, "fixed internal moment is out of range");
    return cache.rho*cache.u[U]*cache.v[V]*cache.w[W]*cache.xi2[Xi2];
}

template<int LeftBasis, int RightBasis, int ExtraU, int ExtraV, int ExtraW>
UGKP_GKS_HD double fixedBasisProductMoment(const MaxwellMomentCache& cache)
{
    static_assert(LeftBasis >= 0 && LeftBasis <= 4, "bad left basis");
    static_assert(RightBasis >= 0 && RightBasis <= 4, "bad right basis");
    constexpr bool leftEnergy = LeftBasis == 4;
    constexpr bool rightEnergy = RightBasis == 4;
    constexpr int baseU = ExtraU
      + (LeftBasis == 1 ? 1 : 0) + (RightBasis == 1 ? 1 : 0);
    constexpr int baseV = ExtraV
      + (LeftBasis == 2 ? 1 : 0) + (RightBasis == 2 ? 1 : 0);
    constexpr int baseW = ExtraW
      + (LeftBasis == 3 ? 1 : 0) + (RightBasis == 3 ? 1 : 0);

    if constexpr (!leftEnergy && !rightEnergy)
    {
        return fixedMonomialMoment<baseU, baseV, baseW, 0>(cache);
    }
    else if constexpr (leftEnergy != rightEnergy)
    {
        double value = 0.0;
        value += 0.5*fixedMonomialMoment<baseU + 2, baseV, baseW, 0>(cache);
        value += 0.5*fixedMonomialMoment<baseU, baseV + 2, baseW, 0>(cache);
        value += 0.5*fixedMonomialMoment<baseU, baseV, baseW + 2, 0>(cache);
        value += 0.5*fixedMonomialMoment<baseU, baseV, baseW, 1>(cache);
        return value;
    }
    else
    {
        double value = 0.0;
        value += 0.25*fixedMonomialMoment<baseU + 4, baseV, baseW, 0>(cache);
        value += 0.25*fixedMonomialMoment<baseU + 2, baseV + 2, baseW, 0>(cache);
        value += 0.25*fixedMonomialMoment<baseU + 2, baseV, baseW + 2, 0>(cache);
        value += 0.25*fixedMonomialMoment<baseU + 2, baseV, baseW, 1>(cache);
        value += 0.25*fixedMonomialMoment<baseU + 2, baseV + 2, baseW, 0>(cache);
        value += 0.25*fixedMonomialMoment<baseU, baseV + 4, baseW, 0>(cache);
        value += 0.25*fixedMonomialMoment<baseU, baseV + 2, baseW + 2, 0>(cache);
        value += 0.25*fixedMonomialMoment<baseU, baseV + 2, baseW, 1>(cache);
        value += 0.25*fixedMonomialMoment<baseU + 2, baseV, baseW + 2, 0>(cache);
        value += 0.25*fixedMonomialMoment<baseU, baseV + 2, baseW + 2, 0>(cache);
        value += 0.25*fixedMonomialMoment<baseU, baseV, baseW + 4, 0>(cache);
        value += 0.25*fixedMonomialMoment<baseU, baseV, baseW + 2, 1>(cache);
        value += 0.25*fixedMonomialMoment<baseU + 2, baseV, baseW, 1>(cache);
        value += 0.25*fixedMonomialMoment<baseU, baseV + 2, baseW, 1>(cache);
        value += 0.25*fixedMonomialMoment<baseU, baseV, baseW + 2, 1>(cache);
        value += 0.25*fixedMonomialMoment<baseU, baseV, baseW, 2>(cache);
        return value;
    }
}

template<int Basis, int ExtraU, int ExtraV, int ExtraW>
UGKP_GKS_HD void appendFixedCoefficientMoment
(
    const MaxwellMomentCache& cache,
    const double coefficients[5],
    double& value
)
{
    value += coefficients[0]
      *fixedBasisProductMoment<Basis, 0, ExtraU, ExtraV, ExtraW>(cache);
    value += coefficients[1]
      *fixedBasisProductMoment<Basis, 1, ExtraU, ExtraV, ExtraW>(cache);
    value += coefficients[2]
      *fixedBasisProductMoment<Basis, 2, ExtraU, ExtraV, ExtraW>(cache);
    value += coefficients[3]
      *fixedBasisProductMoment<Basis, 3, ExtraU, ExtraV, ExtraW>(cache);
    value += coefficients[4]
      *fixedBasisProductMoment<Basis, 4, ExtraU, ExtraV, ExtraW>(cache);
}

template<int Basis, int ExtraU, int ExtraV, int ExtraW>
UGKP_GKS_HD double fixedCoefficientComponent
(
    const MaxwellMomentCache& cache,
    const double coefficients[5]
)
{
    double value = 0.0;
    appendFixedCoefficientMoment<Basis, ExtraU, ExtraV, ExtraW>
    (
        cache, coefficients, value
    );
    return value;
}

template<int ExtraU, int ExtraV, int ExtraW>
UGKP_GKS_HD void fixedCoefficientMoment
(
    const MaxwellMomentCache& cache,
    const double coefficients[5],
    double values[5]
)
{
    values[0] = fixedCoefficientComponent<0, ExtraU, ExtraV, ExtraW>
      (cache, coefficients);
    values[1] = fixedCoefficientComponent<1, ExtraU, ExtraV, ExtraW>
      (cache, coefficients);
    values[2] = fixedCoefficientComponent<2, ExtraU, ExtraV, ExtraW>
      (cache, coefficients);
    values[3] = fixedCoefficientComponent<3, ExtraU, ExtraV, ExtraW>
      (cache, coefficients);
    values[4] = fixedCoefficientComponent<4, ExtraU, ExtraV, ExtraW>
      (cache, coefficients);
}

template<int Basis, bool IncludeFluxVelocity>
UGKP_GKS_HD double fixedSpatialCoefficientComponent
(
    const MaxwellMomentCache& cache,
    const double coefficients[3][5]
)
{
    double value = 0.0;
    appendFixedCoefficientMoment
      <Basis, IncludeFluxVelocity ? 2 : 1, 0, 0>
      (cache, coefficients[0], value);
    appendFixedCoefficientMoment
      <Basis, IncludeFluxVelocity ? 1 : 0, 1, 0>
      (cache, coefficients[1], value);
    appendFixedCoefficientMoment
      <Basis, IncludeFluxVelocity ? 1 : 0, 0, 1>
      (cache, coefficients[2], value);
    return value;
}

template<bool IncludeFluxVelocity>
UGKP_GKS_HD void fixedSpatialCoefficientMoment
(
    const MaxwellMomentCache& cache,
    const double coefficients[3][5],
    double values[5]
)
{
    values[0] = fixedSpatialCoefficientComponent<0, IncludeFluxVelocity>
      (cache, coefficients);
    values[1] = fixedSpatialCoefficientComponent<1, IncludeFluxVelocity>
      (cache, coefficients);
    values[2] = fixedSpatialCoefficientComponent<2, IncludeFluxVelocity>
      (cache, coefficients);
    values[3] = fixedSpatialCoefficientComponent<3, IncludeFluxVelocity>
      (cache, coefficients);
    values[4] = fixedSpatialCoefficientComponent<4, IncludeFluxVelocity>
      (cache, coefficients);
}

UGKP_GKS_HD void momentMatrix
(
    const MaxwellMomentCache& cache,
    double matrix[5][5]
)
{
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            matrix[i][j] = basisProductMoment(cache, i, j, 0, 0, 0);
        }
    }
}

UGKP_GKS_HD bool solveFiveByFive
(
    const double matrix[5][5],
    const double rhs[5],
    double solution[5]
)
{
    double augmented[5][6];
    double scale = 0.0;
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            augmented[i][j] = matrix[i][j];
            scale = maximum(scale, absolute(matrix[i][j]));
        }
        augmented[i][5] = rhs[i];
    }
    const double pivotFloor = maximum(scale*1.0e-14, 1.0e-300);

    for (int column = 0; column < 5; ++column)
    {
        int pivot = column;
        double pivotMagnitude = absolute(augmented[column][column]);
        for (int row = column + 1; row < 5; ++row)
        {
            const double candidate = absolute(augmented[row][column]);
            if (candidate > pivotMagnitude)
            {
                pivot = row;
                pivotMagnitude = candidate;
            }
        }
        if (!finiteValue(pivotMagnitude) || pivotMagnitude <= pivotFloor)
        {
            return false;
        }
        if (pivot != column)
        {
            for (int j = column; j < 6; ++j)
            {
                const double temporary = augmented[column][j];
                augmented[column][j] = augmented[pivot][j];
                augmented[pivot][j] = temporary;
            }
        }

        const double diagonal = augmented[column][column];
        for (int j = column; j < 6; ++j)
        {
            augmented[column][j] /= diagonal;
        }
        for (int row = 0; row < 5; ++row)
        {
            if (row == column)
            {
                continue;
            }
            const double factor = augmented[row][column];
            for (int j = column; j < 6; ++j)
            {
                augmented[row][j] -= factor*augmented[column][j];
            }
        }
    }

    for (int i = 0; i < 5; ++i)
    {
        solution[i] = augmented[i][5];
        if (!finiteValue(solution[i]))
        {
            return false;
        }
    }
    return true;
}

UGKP_GKS_HD bool solveMomentCoefficients
(
    const MaxwellMomentCache& cache,
    const double conservativeDerivative[5],
    double coefficients[5]
)
{
    double matrix[5][5];
    momentMatrix(cache, matrix);

                                                                          
                                                                         
                                                                          
                                                                    
    double scale[5];
    double equilibrated[5][5];
    double rhs[5];
    for (int i = 0; i < 5; ++i)
    {
        const double diagonal = absolute(matrix[i][i]);
        if (!finiteValue(diagonal) || diagonal <= 1.0e-300)
        {
            return false;
        }
        scale[i] = ::sqrt(diagonal);
        rhs[i] = conservativeDerivative[i]/scale[i];
    }
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            equilibrated[i][j] = matrix[i][j]/(scale[i]*scale[j]);
        }
    }

    double scaledSolution[5];
    if (!solveFiveByFive(equilibrated, rhs, scaledSolution))
    {
        return false;
    }
    for (int i = 0; i < 5; ++i)
    {
        coefficients[i] = scaledSolution[i]/scale[i];
        if (!finiteValue(coefficients[i]))
        {
            return false;
        }
    }
    return true;
}

UGKP_GKS_HD void conservativeMoment
(
    const MaxwellMomentCache& cache,
    double values[5]
)
{
    for (int i = 0; i < 5; ++i)
    {
        values[i] = basisProductMoment(cache, i, 0, 0, 0, 0);
    }
}

UGKP_GKS_HD void coefficientMoment
(
    const MaxwellMomentCache& cache,
    const double coefficients[5],
    const int extraU,
    const int extraV,
    const int extraW,
    double values[5]
)
{
    for (int i = 0; i < 5; ++i)
    {
        double value = 0.0;
        for (int j = 0; j < 5; ++j)
        {
            value += coefficients[j]
              *basisProductMoment
               (cache, i, j, extraU, extraV, extraW);
        }
        values[i] = value;
    }
}

UGKP_GKS_HD void baseFluxMoment
(
    const MaxwellMomentCache& cache,
    double values[5]
)
{
    for (int i = 0; i < 5; ++i)
    {
        values[i] = basisProductMoment(cache, i, 0, 1, 0, 0);
    }
}

UGKP_GKS_HD void spatialCoefficientMoment
(
    const MaxwellMomentCache& cache,
    const double coefficients[3][5],
    const bool includeFluxVelocity,
    double values[5]
)
{
    for (int i = 0; i < 5; ++i)
    {
        double value = 0.0;
        for (int direction = 0; direction < 3; ++direction)
        {
            const int extraU = (includeFluxVelocity ? 1 : 0)
              + (direction == 0 ? 1 : 0);
            const int extraV = direction == 1 ? 1 : 0;
            const int extraW = direction == 2 ? 1 : 0;
            for (int j = 0; j < 5; ++j)
            {
                value += coefficients[direction][j]
                  *basisProductMoment
                   (cache, i, j, extraU, extraV, extraW);
            }
        }
        values[i] = value;
    }
}

UGKP_GKS_HD bool solveTemporalCoefficient
(
    const MaxwellMomentCache& cache,
    const double spatialCoefficients[3][5],
    double temporalCoefficients[5]
)
{
    double spatialMoment[5];
    fixedSpatialCoefficientMoment<false>
    (
        cache, spatialCoefficients, spatialMoment
    );
    for (int i = 0; i < 5; ++i)
    {
        spatialMoment[i] = -spatialMoment[i];
    }
    return solveMomentCoefficients
    (
        cache, spatialMoment, temporalCoefficients
    );
}

UGKP_GKS_HD bool primitiveFromConservative
(
    const double conservative[5],
    const double gamma,
    const double gasConstant,
    LocalPrimitive& state
)
{
    if
    (
        !finiteValue(conservative[0]) || conservative[0] <= 0.0
     || !finiteValue(gamma) || gamma <= 1.0
     || !finiteValue(gasConstant) || gasConstant <= 0.0
    )
    {
        return false;
    }
    state.rho = conservative[0];
    state.u = conservative[1]/state.rho;
    state.v = conservative[2]/state.rho;
    state.w = conservative[3]/state.rho;
    const double kinetic = 0.5*state.rho*
      (state.u*state.u + state.v*state.v + state.w*state.w);
    const double internalEnergy = conservative[4] - kinetic;
    state.p = (gamma - 1.0)*internalEnergy;
    state.T = state.p/(state.rho*gasConstant);
    return
        finiteValue(state.u) && finiteValue(state.v)
     && finiteValue(state.w) && finiteValue(state.p)
     && finiteValue(state.T) && state.p > 0.0 && state.T > 0.0;
}

UGKP_GKS_HD bool analyticMomentCoefficients
(
    const LocalPrimitive& state,
    const double gamma,
    const double conservativeDerivative[5],
    double coefficients[5]
)
{
    if
    (
        state.rho <= 0.0 || state.p <= 0.0
     || gamma <= 1.0 || gamma > 5.0/3.0 + 1.0e-12
    )
    {
        return false;
    }

    const double inverseRho = 1.0/state.rho;
    const double theta = state.p*inverseRho;
    const double inverseTheta = 1.0/theta;
    const double densityDerivative = conservativeDerivative[0];
    const double velocityDerivative[3]
    {
        (conservativeDerivative[1] - state.u*densityDerivative)*inverseRho,
        (conservativeDerivative[2] - state.v*densityDerivative)*inverseRho,
        (conservativeDerivative[3] - state.w*densityDerivative)*inverseRho
    };
    const double speedSquared =
        state.u*state.u + state.v*state.v + state.w*state.w;
    const double velocityWork =
        state.u*velocityDerivative[0]
      + state.v*velocityDerivative[1]
      + state.w*velocityDerivative[2];
    const double pressureDerivative = (gamma - 1.0)*
      (
          conservativeDerivative[4]
        - 0.5*speedSquared*densityDerivative
        - state.rho*velocityWork
      );
    const double thetaDerivative = inverseRho*
      (pressureDerivative - theta*densityDerivative);
    const double inverseThetaSquared = inverseTheta*inverseTheta;
    const double internalDegrees = 2.0/(gamma - 1.0) - 3.0;
    const double totalDegrees = internalDegrees + 3.0;

    coefficients[4] = thetaDerivative*inverseThetaSquared;
    coefficients[1] = velocityDerivative[0]*inverseTheta
      - state.u*thetaDerivative*inverseThetaSquared;
    coefficients[2] = velocityDerivative[1]*inverseTheta
      - state.v*thetaDerivative*inverseThetaSquared;
    coefficients[3] = velocityDerivative[2]*inverseTheta
      - state.w*thetaDerivative*inverseThetaSquared;
    coefficients[0] = densityDerivative*inverseRho
      - 0.5*totalDegrees*thetaDerivative*inverseTheta
      - velocityWork*inverseTheta
      + 0.5*speedSquared*thetaDerivative*inverseThetaSquared;

    for (int i = 0; i < 5; ++i)
    {
        if (!finiteValue(coefficients[i]))
        {
            return false;
        }
    }
    return true;
}

UGKP_GKS_HD bool analyticTemporalCoefficient
(
    const LocalPrimitive& state,
    const MaxwellMomentCache& cache,
    const double gamma,
    const double spatialCoefficients[3][5],
    double temporalCoefficients[5]
)
{
    double temporalDerivative[5];
    fixedSpatialCoefficientMoment<false>
    (
        cache, spatialCoefficients, temporalDerivative
    );
    for (int i = 0; i < 5; ++i)
    {
        temporalDerivative[i] = -temporalDerivative[i];
    }
    return analyticMomentCoefficients
    (
        state, gamma, temporalDerivative, temporalCoefficients
    );
}

UGKP_GKS_HD TimeCoefficients timeCoefficients
(
    const double dt,
    const double numericalTau
)
{
    TimeCoefficients coefficients;
    coefficients.equilibriumTime = 0.5*dt;
    if (dt <= 0.0)
    {
        coefficients.equilibrium = 0.0;
        coefficients.freeTransport = 0.0;
        coefficients.timeExponential = 0.0;
        return coefficients;
    }
    if (numericalTau <= 1.0e-300)
    {
        coefficients.equilibrium = 1.0;
        coefficients.freeTransport = 0.0;
        coefficients.timeExponential = 0.0;
        return coefficients;
    }

    const double ratio = dt/numericalTau;
    double freeFraction;
    double exponential;
    double timeExponential;
    if (ratio < 1.0e-4)
    {
        const double r2 = ratio*ratio;
        const double r3 = r2*ratio;
        const double r4 = r3*ratio;
        freeFraction = 1.0 - 0.5*ratio + r2/6.0 - r3/24.0 + r4/120.0;
        exponential = 1.0 - ratio + 0.5*r2 - r3/6.0 + r4/24.0;
        timeExponential = dt*
          (0.5 - ratio/3.0 + r2/8.0 - r3/30.0 + r4/144.0);
    }
    else if (ratio > 80.0)
    {
        exponential = 0.0;
        freeFraction = 1.0/ratio;
        timeExponential = numericalTau*freeFraction;
    }
    else
    {
        exponential = ::exp(-ratio);
        freeFraction = -::expm1(-ratio)/ratio;
        timeExponential = numericalTau*(freeFraction - exponential);
    }
    coefficients.freeTransport = freeFraction;
    coefficients.equilibrium = 1.0 - freeFraction;
    coefficients.timeExponential = timeExponential;
    return coefficients;
}

UGKP_GKS_ENTRY FullGksResult fullSecondOrderFlux
(
    const LocalPrimitive& left,
    const LocalPrimitive& right,
    const double leftConservativeGradients[3][5],
    const double rightConservativeGradients[3][5],
    const double gamma,
    const double gasConstant,
    const double molecularViscosity,
    const double prandtl,
    const double dt,
    const double numericalDissipationTau
)
{
    FullGksResult result;
    result.valid = false;
    result.molecularHeatFlux = 0.0;
    for (int i = 0; i < 5; ++i)
    {
        result.flux[i] = 0.0;
        result.molecularViscousFlux[i] = 0.0;
    }

    if
    (
        left.rho <= 0.0 || right.rho <= 0.0
     || left.p <= 0.0 || right.p <= 0.0
     || gamma <= 1.0 || gamma > 5.0/3.0 + 1.0e-12
     || gasConstant <= 0.0 || molecularViscosity < 0.0
     || prandtl <= 0.0 || dt <= 0.0 || numericalDissipationTau < 0.0
    )
    {
        return result;
    }

    const double internalDegrees = 2.0/(gamma - 1.0) - 3.0;
    const MaxwellMomentCache leftFull =
        makeMomentCache(left, internalDegrees, 0);
    const MaxwellMomentCache rightFull =
        makeMomentCache(right, internalDegrees, 0);
    const MaxwellMomentCache leftPositive =
        makeMomentCache(left, internalDegrees, 1);
    const MaxwellMomentCache rightNegative =
        makeMomentCache(right, internalDegrees, -1);

    double leftSpatial[3][5];
    double rightSpatial[3][5];
    for (int direction = 0; direction < 3; ++direction)
    {
        if
        (
            !analyticMomentCoefficients
             (
                 left,
                 gamma,
                 leftConservativeGradients[direction],
                 leftSpatial[direction]
             )
         || !analyticMomentCoefficients
             (
                 right,
                 gamma,
                 rightConservativeGradients[direction],
                 rightSpatial[direction]
             )
        )
        {
            return result;
        }
    }

    double leftTemporal[5];
    double rightTemporal[5];
    if
    (
        !analyticTemporalCoefficient
         (left, leftFull, gamma, leftSpatial, leftTemporal)
     || !analyticTemporalCoefficient
         (right, rightFull, gamma, rightSpatial, rightTemporal)
    )
    {
        return result;
    }

    double leftHalfState[5];
    double rightHalfState[5];
    conservativeMoment(leftPositive, leftHalfState);
    conservativeMoment(rightNegative, rightHalfState);
    double interfaceConservative[5];
    for (int i = 0; i < 5; ++i)
    {
        interfaceConservative[i] = leftHalfState[i] + rightHalfState[i];
    }
    if
    (
        !primitiveFromConservative
         (
             interfaceConservative,
             gamma,
             gasConstant,
             result.interfaceState
         )
    )
    {
        return result;
    }

    const MaxwellMomentCache interfaceFull =
        makeMomentCache(result.interfaceState, internalDegrees, 0);

    double interfaceSpatial[3][5];
    for (int direction = 0; direction < 3; ++direction)
    {
        double leftDerivative[5];
        double rightDerivative[5];
        fixedCoefficientMoment<0, 0, 0>
        (
            leftPositive, leftSpatial[direction], leftDerivative
        );
        fixedCoefficientMoment<0, 0, 0>
        (
            rightNegative, rightSpatial[direction], rightDerivative
        );
        double derivative[5];
        for (int i = 0; i < 5; ++i)
        {
            derivative[i] = leftDerivative[i] + rightDerivative[i];
        }
        if
        (
            !analyticMomentCoefficients
             (
                 result.interfaceState,
                 gamma,
                 derivative,
                 interfaceSpatial[direction]
             )
        )
        {
            return result;
        }
    }

    double interfaceTemporal[5];
    if
    (
        !analyticTemporalCoefficient
         (
             result.interfaceState,
             interfaceFull,
             gamma,
             interfaceSpatial,
             interfaceTemporal
         )
    )
    {
        return result;
    }

    double equilibriumFlux[5];
    double freeLeft[5];
    double freeRight[5];
    baseFluxMoment(interfaceFull, equilibriumFlux);
    baseFluxMoment(leftPositive, freeLeft);
    baseFluxMoment(rightNegative, freeRight);

    double equilibriumTemporalFlux[5];
    double leftTemporalFlux[5];
    double rightTemporalFlux[5];
    fixedCoefficientMoment<1, 0, 0>
    (
        interfaceFull, interfaceTemporal, equilibriumTemporalFlux
    );
    fixedCoefficientMoment<1, 0, 0>
    (
        leftPositive, leftTemporal, leftTemporalFlux
    );
    fixedCoefficientMoment<1, 0, 0>
    (
        rightNegative, rightTemporal, rightTemporalFlux
    );

    double equilibriumSpatialFlux[5];
    double leftSpatialFlux[5];
    double rightSpatialFlux[5];
    fixedSpatialCoefficientMoment<true>
    (
        interfaceFull, interfaceSpatial, equilibriumSpatialFlux
    );
    fixedSpatialCoefficientMoment<true>
    (
        leftPositive, leftSpatial, leftSpatialFlux
    );
    fixedSpatialCoefficientMoment<true>
    (
        rightNegative, rightSpatial, rightSpatialFlux
    );

    const double physicalTau = molecularViscosity
       /maximum(result.interfaceState.p, 1.0e-300);
                                                                       
                                                                         
                                                                             
                                                  
    const TimeCoefficients coefficients = timeCoefficients
    (
        dt, physicalTau + numericalDissipationTau
    );

    for (int i = 0; i < 5; ++i)
    {
        const double freeFlux = freeLeft[i] + freeRight[i];
        const double freeTemporalFlux =
            leftTemporalFlux[i] + rightTemporalFlux[i];
        const double freeSpatialFlux =
            leftSpatialFlux[i] + rightSpatialFlux[i];

        result.flux[i] =
            coefficients.equilibrium*equilibriumFlux[i]
          + coefficients.freeTransport*freeFlux
          + coefficients.equilibriumTime*equilibriumTemporalFlux[i]
          - physicalTau*coefficients.equilibrium
           *(equilibriumSpatialFlux[i] + equilibriumTemporalFlux[i])
          - physicalTau*coefficients.freeTransport
           *(freeSpatialFlux + freeTemporalFlux)
          + coefficients.timeExponential
           *(equilibriumSpatialFlux[i] - freeSpatialFlux);

        result.molecularViscousFlux[i] = -physicalTau*
          (equilibriumSpatialFlux[i] + equilibriumTemporalFlux[i]);
        if (!finiteValue(result.flux[i]))
        {
            return result;
        }
    }

    result.molecularHeatFlux =
        result.molecularViscousFlux[4]
      - result.interfaceState.u*result.molecularViscousFlux[1]
      - result.interfaceState.v*result.molecularViscousFlux[2]
      - result.interfaceState.w*result.molecularViscousFlux[3];
    result.flux[4] +=
        (1.0/prandtl - 1.0)*result.molecularHeatFlux;
    result.valid = finiteValue(result.flux[4]);
    return result;
}

struct SideFluxAccumulation
{
    double interfaceConservative[5];
    double interfaceDerivative[3][5];
    double freeFlux[5];
    double freeTemporalFlux[5];
    double freeSpatialFlux[5];
};

UGKP_GKS_HD void clearGksSideAccumulation
(
    SideFluxAccumulation& accumulation
)
{
    for (int component = 0; component < 5; ++component)
    {
        accumulation.interfaceConservative[component] = 0.0;
        accumulation.freeFlux[component] = 0.0;
        accumulation.freeTemporalFlux[component] = 0.0;
        accumulation.freeSpatialFlux[component] = 0.0;
        for (int direction = 0; direction < 3; ++direction)
        {
            accumulation.interfaceDerivative[direction][component] = 0.0;
        }
    }
}

UGKP_GKS_ENTRY bool accumulateGksSide
(
    const LocalPrimitive& state,
    const double conservativeGradients[3][5],
    const double gamma,
    const int halfSign,
    SideFluxAccumulation& accumulation
)
{
    const double internalDegrees = 2.0/(gamma - 1.0) - 3.0;
    double spatial[3][5];
    double temporal[5];
    {
        const MaxwellMomentCache full =
            makeMomentCache(state, internalDegrees, 0);
        for (int direction = 0; direction < 3; ++direction)
        {
            if
            (
                !analyticMomentCoefficients
                 (
                     state,
                     gamma,
                     conservativeGradients[direction],
                     spatial[direction]
                 )
            )
            {
                return false;
            }
        }
        if
        (
            !analyticTemporalCoefficient
             (state, full, gamma, spatial, temporal)
        )
        {
            return false;
        }
    }

    const MaxwellMomentCache half =
        makeMomentCache(state, internalDegrees, halfSign);
    double values[5];
    conservativeMoment(half, values);
    for (int component = 0; component < 5; ++component)
    {
        accumulation.interfaceConservative[component] += values[component];
    }

    for (int direction = 0; direction < 3; ++direction)
    {
        fixedCoefficientMoment<0, 0, 0>
        (
            half, spatial[direction], values
        );
        for (int component = 0; component < 5; ++component)
        {
            accumulation.interfaceDerivative[direction][component]
              += values[component];
        }
    }

    baseFluxMoment(half, values);
    for (int component = 0; component < 5; ++component)
    {
        accumulation.freeFlux[component] += values[component];
    }
    fixedCoefficientMoment<1, 0, 0>(half, temporal, values);
    for (int component = 0; component < 5; ++component)
    {
        accumulation.freeTemporalFlux[component] += values[component];
    }
    fixedSpatialCoefficientMoment<true>(half, spatial, values);
    for (int component = 0; component < 5; ++component)
    {
        accumulation.freeSpatialFlux[component] += values[component];
    }
    return true;
}

UGKP_GKS_ENTRY FullGksResult finalizeGksFlux
(
    const SideFluxAccumulation& accumulation,
    const double gamma,
    const double gasConstant,
    const double molecularViscosity,
    const double prandtl,
    const double dt,
    const double numericalDissipationTau
)
{
    FullGksResult result;
    result.valid = false;
    result.molecularHeatFlux = 0.0;
    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] = 0.0;
        result.molecularViscousFlux[component] = 0.0;
    }
    if
    (
        gamma <= 1.0 || gamma > 5.0/3.0 + 1.0e-12
     || gasConstant <= 0.0 || molecularViscosity < 0.0
     || prandtl <= 0.0 || dt <= 0.0 || numericalDissipationTau < 0.0
    )
    {
        return result;
    }
    if
    (
        !primitiveFromConservative
         (
             accumulation.interfaceConservative,
             gamma,
             gasConstant,
             result.interfaceState
         )
    )
    {
        return result;
    }

    const double internalDegrees = 2.0/(gamma - 1.0) - 3.0;
    const MaxwellMomentCache interfaceFull =
        makeMomentCache(result.interfaceState, internalDegrees, 0);
    double interfaceSpatial[3][5];
    for (int direction = 0; direction < 3; ++direction)
    {
        if
        (
            !analyticMomentCoefficients
             (
                 result.interfaceState,
                 gamma,
                 accumulation.interfaceDerivative[direction],
                 interfaceSpatial[direction]
             )
        )
        {
            return result;
        }
    }
    double interfaceTemporal[5];
    if
    (
        !analyticTemporalCoefficient
         (
             result.interfaceState,
             interfaceFull,
             gamma,
             interfaceSpatial,
             interfaceTemporal
         )
    )
    {
        return result;
    }

    double equilibriumFlux[5];
    double equilibriumTemporalFlux[5];
    double equilibriumSpatialFlux[5];
    baseFluxMoment(interfaceFull, equilibriumFlux);
    fixedCoefficientMoment<1, 0, 0>
    (
        interfaceFull, interfaceTemporal, equilibriumTemporalFlux
    );
    fixedSpatialCoefficientMoment<true>
    (
        interfaceFull, interfaceSpatial, equilibriumSpatialFlux
    );

    const double physicalTau = molecularViscosity
       /maximum(result.interfaceState.p, 1.0e-300);
    const TimeCoefficients coefficients = timeCoefficients
    (
        dt, physicalTau + numericalDissipationTau
    );
    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] =
            coefficients.equilibrium*equilibriumFlux[component]
          + coefficients.freeTransport*accumulation.freeFlux[component]
          + coefficients.equilibriumTime*equilibriumTemporalFlux[component]
          - physicalTau*coefficients.equilibrium
           *(
               equilibriumSpatialFlux[component]
             + equilibriumTemporalFlux[component]
            )
          - physicalTau*coefficients.freeTransport
           *(
               accumulation.freeSpatialFlux[component]
             + accumulation.freeTemporalFlux[component]
            )
          + coefficients.timeExponential
           *(
               equilibriumSpatialFlux[component]
             - accumulation.freeSpatialFlux[component]
            );
        result.molecularViscousFlux[component] = -physicalTau*
          (
              equilibriumSpatialFlux[component]
            + equilibriumTemporalFlux[component]
          );
        if (!finiteValue(result.flux[component]))
        {
            return result;
        }
    }
    result.molecularHeatFlux =
        result.molecularViscousFlux[4]
      - result.interfaceState.u*result.molecularViscousFlux[1]
      - result.interfaceState.v*result.molecularViscousFlux[2]
      - result.interfaceState.w*result.molecularViscousFlux[3];
    result.flux[4] +=
        (1.0/prandtl - 1.0)*result.molecularHeatFlux;
    result.valid = finiteValue(result.flux[4]);
    return result;
}

UGKP_GKS_ENTRY FullGksResult fullSecondOrderFluxStaged
(
    const LocalPrimitive& left,
    const LocalPrimitive& right,
    const double leftConservativeGradients[3][5],
    const double rightConservativeGradients[3][5],
    const double gamma,
    const double gasConstant,
    const double molecularViscosity,
    const double prandtl,
    const double dt,
    const double numericalDissipationTau
)
{
    FullGksResult result;
    result.valid = false;
    result.molecularHeatFlux = 0.0;
    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] = 0.0;
        result.molecularViscousFlux[component] = 0.0;
    }
    if
    (
        left.rho <= 0.0 || right.rho <= 0.0
     || left.p <= 0.0 || right.p <= 0.0
     || gamma <= 1.0 || gamma > 5.0/3.0 + 1.0e-12
     || gasConstant <= 0.0 || molecularViscosity < 0.0
     || prandtl <= 0.0 || dt <= 0.0 || numericalDissipationTau < 0.0
    )
    {
        return result;
    }

    SideFluxAccumulation accumulation;
    for (int component = 0; component < 5; ++component)
    {
        accumulation.interfaceConservative[component] = 0.0;
        accumulation.freeFlux[component] = 0.0;
        accumulation.freeTemporalFlux[component] = 0.0;
        accumulation.freeSpatialFlux[component] = 0.0;
        for (int direction = 0; direction < 3; ++direction)
        {
            accumulation.interfaceDerivative[direction][component] = 0.0;
        }
    }
    if
    (
        !accumulateGksSide
         (
             left,
             leftConservativeGradients,
             gamma,
             1,
             accumulation
         )
     || !accumulateGksSide
         (
             right,
             rightConservativeGradients,
             gamma,
             -1,
             accumulation
         )
    )
    {
        return result;
    }

    if
    (
        !primitiveFromConservative
         (
             accumulation.interfaceConservative,
             gamma,
             gasConstant,
             result.interfaceState
         )
    )
    {
        return result;
    }

    const double internalDegrees = 2.0/(gamma - 1.0) - 3.0;
    const MaxwellMomentCache interfaceFull =
        makeMomentCache(result.interfaceState, internalDegrees, 0);
    double interfaceSpatial[3][5];
    for (int direction = 0; direction < 3; ++direction)
    {
        if
        (
            !analyticMomentCoefficients
             (
                 result.interfaceState,
                 gamma,
                 accumulation.interfaceDerivative[direction],
                 interfaceSpatial[direction]
             )
        )
        {
            return result;
        }
    }
    double interfaceTemporal[5];
    if
    (
        !analyticTemporalCoefficient
         (
             result.interfaceState,
             interfaceFull,
             gamma,
             interfaceSpatial,
             interfaceTemporal
         )
    )
    {
        return result;
    }

    double equilibriumFlux[5];
    double equilibriumTemporalFlux[5];
    double equilibriumSpatialFlux[5];
    baseFluxMoment(interfaceFull, equilibriumFlux);
    fixedCoefficientMoment<1, 0, 0>
    (
        interfaceFull, interfaceTemporal, equilibriumTemporalFlux
    );
    fixedSpatialCoefficientMoment<true>
    (
        interfaceFull, interfaceSpatial, equilibriumSpatialFlux
    );

    const double physicalTau = molecularViscosity
       /maximum(result.interfaceState.p, 1.0e-300);
    const TimeCoefficients coefficients = timeCoefficients
    (
        dt, physicalTau + numericalDissipationTau
    );
    for (int component = 0; component < 5; ++component)
    {
        result.flux[component] =
            coefficients.equilibrium*equilibriumFlux[component]
          + coefficients.freeTransport*accumulation.freeFlux[component]
          + coefficients.equilibriumTime*equilibriumTemporalFlux[component]
          - physicalTau*coefficients.equilibrium
           *(
               equilibriumSpatialFlux[component]
             + equilibriumTemporalFlux[component]
            )
          - physicalTau*coefficients.freeTransport
           *(
               accumulation.freeSpatialFlux[component]
             + accumulation.freeTemporalFlux[component]
            )
          + coefficients.timeExponential
           *(
               equilibriumSpatialFlux[component]
             - accumulation.freeSpatialFlux[component]
            );
        result.molecularViscousFlux[component] = -physicalTau*
          (
              equilibriumSpatialFlux[component]
            + equilibriumTemporalFlux[component]
          );
        if (!finiteValue(result.flux[component]))
        {
            return result;
        }
    }

    result.molecularHeatFlux =
        result.molecularViscousFlux[4]
      - result.interfaceState.u*result.molecularViscousFlux[1]
      - result.interfaceState.v*result.molecularViscousFlux[2]
      - result.interfaceState.w*result.molecularViscousFlux[3];
    result.flux[4] +=
        (1.0/prandtl - 1.0)*result.molecularHeatFlux;
    result.valid = finiteValue(result.flux[4]);
    return result;
}

}                     

#undef UGKP_GKS_HD
#undef UGKP_GKS_ENTRY

#endif
