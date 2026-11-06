#ifndef UGKP_GPU_TRAJECTORY_POLYNOMIAL_MATH_H
#define UGKP_GPU_TRAJECTORY_POLYNOMIAL_MATH_H
#include "ExtendedGeometryMath.H"
#ifdef __CUDACC__
#define UGKP_TRAJECTORY_HD __host__ __device__
#else
#define UGKP_TRAJECTORY_HD
#endif
namespace ugkp { namespace gpu { namespace geometry {
namespace trajectoryDetail {
template<class Extended> struct Arithmetic {
    UGKP_TRAJECTORY_HD static bool less(Extended a,Extended b){return a<b;}
    UGKP_TRAJECTORY_HD static bool zero(Extended a){return a==Extended(0);}
    UGKP_TRAJECTORY_HD static Extended squareRoot(Extended a){return std::sqrt(a);}
    UGKP_TRAJECTORY_HD static double value(Extended a){return static_cast<double>(a);}
};
template<> struct Arithmetic<extended::Wide> {
    UGKP_TRAJECTORY_HD static bool less(extended::Wide a,extended::Wide b){return extended::less(a,b);}
    UGKP_TRAJECTORY_HD static bool zero(extended::Wide a){return a.high==0;}
    UGKP_TRAJECTORY_HD static extended::Wide squareRoot(extended::Wide a){return extended::squareRoot(a);}
    UGKP_TRAJECTORY_HD static double value(extended::Wide a){return extended::value(a);}
};
}
// The cubic is represented by its actual endpoint Bernstein controls. Keep
// stable quadratic derivative roots and De Casteljau evaluation; do not infer
// the endpoint by subtracting nearly cancelling displacement powers. CUDA uses
// scaled FP64 pairs for the extended intermediate range of the native oracle.
template<class Extended>
UGKP_TRAJECTORY_HD inline void trajectoryPolynomialBounds(const double* control,double& minimum,double& maximum){
    using A=trajectoryDetail::Arithmetic<Extended>;
    minimum=control[0]<control[3]?control[0]:control[3];
    maximum=control[0]>control[3]?control[0]:control[3];
    const Extended a=Extended(3)*(-Extended(control[0])+Extended(3)*Extended(control[1])
        -Extended(3)*Extended(control[2])+Extended(control[3]));
    const Extended b=Extended(6)*(Extended(control[0])-Extended(2)*Extended(control[1])+Extended(control[2]));
    const Extended c=Extended(3)*(Extended(control[1])-Extended(control[0]));
    Extended roots[2]={Extended(-1),Extended(-1)};
    if(A::zero(a)){
        if(!A::zero(b))roots[0]=-c/b;
    }else{
        const Extended discriminant=b*b-Extended(4)*a*c;
        if(!A::less(discriminant,Extended(0))){
            const Extended root=A::squareRoot(discriminant);
            const Extended q=Extended(-.5)*(b+(!A::less(b,Extended(0))?root:-root));
            if(!A::zero(q)){roots[0]=q/a;roots[1]=c/q;}
            else roots[0]=-b/(Extended(2)*a);
        }
    }
    for(Extended t:roots)if(A::less(Extended(0),t)&&A::less(t,Extended(1))){
        Extended value[4]={Extended(control[0]),Extended(control[1]),Extended(control[2]),Extended(control[3])};
        for(int degree=3;degree>0;--degree)
            for(int k=0;k<degree;++k)value[k]=(Extended(1)-t)*value[k]+t*value[k+1];
        const double evaluated=A::value(value[0]);
        minimum=minimum<evaluated?minimum:evaluated;
        maximum=maximum>evaluated?maximum:evaluated;
    }
}
}}}
#undef UGKP_TRAJECTORY_HD
#endif
