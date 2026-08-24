#ifndef UGKWP_GPU_DRAG_MODELS_CUH
#define UGKWP_GPU_DRAG_MODELS_CUH

namespace ugkwpGpuDrag
{

struct DragInput
{
    double gasDensity;
    double gasVolumeFraction;
    double gasViscosity;
    double solidDensity;
    double diameter;
    double relativeSpeed;
};

struct SchillerNaumannDrag
{
    __device__ static double inverseRelaxationTime(const DragInput& in)
    {
        const double mu = fmax(in.gasViscosity, 1.0e-30);
        const double re =
            fmax(in.gasDensity, 0.0)*fmax(in.diameter, 1.0e-30)
           *fmax(in.relativeSpeed, 0.0)/mu;
        if (re <= 1.0e-30 || in.relativeSpeed <= 1.0e-30)
        {
            return 0.0;
        }
        const double cd =
            re < 1000.0
          ? 24.0*(1.0 + 0.15*pow(re, 0.687))/re
          : 0.44;
        return
            0.75*cd*fmax(in.gasDensity, 0.0)
           *fmax(in.relativeSpeed, 0.0)
           /(fmax(in.solidDensity, 1.0e-30)
            *fmax(in.diameter, 1.0e-30));
    }
};

struct GidaspowErgunWenYuDrag
{
    double residualRe;

    __device__ double inverseRelaxationTime(const DragInput& in) const
    {
        const double alpha = fmin(fmax(in.gasVolumeFraction, 1.0e-12), 1.0);
        const double mu = fmax(in.gasViscosity, 1.0e-30);
        const double diameter = fmax(in.diameter, 1.0e-30);
        const double re =
            fmax(in.gasDensity, 0.0)*diameter
           *fmax(in.relativeSpeed, 0.0)/mu;
        const double alphaRe = alpha*re;
        const double cdReWenYu =
            alphaRe < 1000.0
          ? 24.0*(1.0 + 0.15*pow(fmax(alphaRe, 0.0), 0.687))
          : 0.44*fmax(alphaRe, residualRe);
        const double cdRe =
            alpha >= 0.8
          ? cdReWenYu*pow(alpha, -2.65)
          : (4.0/3.0)
           *(150.0*(1.0 - alpha)/alpha + 1.75*re);
        return
            0.75*cdRe*mu
           /(fmax(in.solidDensity, 1.0e-30)*diameter*diameter);
    }
};

}                          

#endif
