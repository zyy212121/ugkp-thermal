#ifndef CHMT_CORE_NUMERICALCONTROLS_H
#define CHMT_CORE_NUMERICALCONTROLS_H
namespace chmt {
// Startup-only numerical settings. These are not physical checkpoint identity
// and are deliberately absent from the historical explicit checkpoint archive.
template<class Number> struct SurfaceNumericalControls {
    int compositionMaximumIterations=60;
    Number compositionRelaxation=Number(.5),compositionAbsoluteTolerance=Number(1e-12);
    int polynomialBisectionIterations=80,isolatedRootIterations=80;
    int refinementIterations=8,structuredNewtonIterations=8,poleSearchIterations=160;
    int thermalMaximumIterations=100,initialSubdivisions=32,maximumSubdivisions=512,subdivisionGrowth=4;
};
struct MeshNumericalControls {
    double absoluteVolumeTolerance=1e-14,relativeVolumeTolerance=1e-10,rankTolerance=1e-11;
    int maximumIterations=24,maximumLineSearch=24,harmonicMaximumIterations=20000;
    double harmonicRelativeTolerance=1e-12;
};
template<class Number> struct NumericalControls {
    int thermoMaximumIterations=100,reconstructionMaximumIterations=32;
    SurfaceNumericalControls<Number> surface{};
    MeshNumericalControls meshMotion{};
};
}
#endif
