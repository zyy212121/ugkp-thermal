#include "ParticleDiameterDistribution.H"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
int main() {
    using ugkwp::rosinRammlerDiameter;
    std::mt19937_64 rng(98123);
    std::uniform_real_distribution<double> uniform(0,1);
    int samples=0;
    for(double n:{.5,1.,2.,8.,32.})for(double lo:{1e-8,5e-5,2e-4}) {
        const double scale=1e-4,hi=3e-4;
        const long double a=std::pow((long double)lo/scale,n);
        const long double span=std::pow((long double)hi/scale,n)-a;
        for(int i=0;i<10000;++i) {
            const double u=uniform(rng);
            const double d=rosinRammlerDiameter(scale,n,lo,hi,u);
            assert(std::isfinite(d)&&d>=lo&&d<=hi);
            const long double q=std::pow((long double)d/scale,n)-a;
            const long double cdf=-std::expm1(-q)/(-std::expm1(-span));
            // Extremely far-tail double diameters may have a resolved CDF step.
            const double tolerance=1e-9+std::min(1.,double(a*n)*1e-14);
            assert(std::abs(cdf-u)<tolerance);
            ++samples;
        }
        assert(rosinRammlerDiameter(scale,n,lo,hi,0.)==lo);
        assert(rosinRammlerDiameter(scale,n,lo,hi,1.)==hi);
    }
    assert(rosinRammlerDiameter(1e-4,2.,1e-4,1e-4,.5)==1e-4);
    // Conditional tail normalization and tiny-q limiting distribution.
    for(double n:{1.,2.,8.})for(double scale:{1e-20,1e20}) {
        const double d=rosinRammlerDiameter(scale,n,1e-4,2e-4,.5);
        assert(std::isfinite(d)&&d>=1e-4&&d<=2e-4);
    }
    std::cout<<"PASS RR inverse-CDF "<<samples<<" samples, endpoints, monodisperse, extreme scales\n";
}
