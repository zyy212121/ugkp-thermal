#include "common/gpu/AnchoredRadiationLinearization.H"
#include <cassert>
#include <limits>
#include <iostream>
#ifdef TEST_FP32
using Real=float;
#else
using Real=double;
#endif
bool near(Real a,Real b){return std::abs(a-b)<=Real(64)*std::numeric_limits<Real>::epsilon()*std::max(Real(1),std::abs(b));}
int main(){using namespace ugkp::gpu;
 const Real B0=Real(100),dt=Real(.1),A=Real(2),b=Real(3),fourPi=Real(12.566370614359172953850573533118);
 for(Real trial:{Real(-5),Real(0),Real(4)}){
  auto l=linearizeAnchoredRadiation(trial,B0+b*trial,B0,b,dt,A);assert(l.valid&&l.nonnegativeSlope&&near(l.b,b));
  for(Real G:{Real(80),Real(100),Real(130)}){
   Real q=l.predictedQ(G);assert(near(q,dt*A*(G-(B0+b*q))));
   auto r=materialRadiationResidual(q,B0+b*q,A,Real(1),G,dt,Real(1e-10));assert(r.valid&&r.normalized<Real(2e-5));
   // Uniform angular source integrates back to f*B0 +(1-f)*G.
   Real integral=0;for(int n=0;n<8;++n)integral+=(fourPi/Real(8))*(l.linearizedEmission+l.thermalFeedback*G/fourPi);
   assert(near(integral,l.f*B0+l.thermalFeedback*G));
  }
 }
 auto first=linearizeAnchoredRadiation(Real(4),Real(112),B0,b,dt,A);
 auto changed=linearizeAnchoredRadiation(Real(4),Real(112),B0,b,dt,Real(4));assert(changed.f!=first.f);
 auto residual=materialRadiationResidual(first.predictedQ(Real(130)),B0+b*first.predictedQ(Real(130)),Real(4),Real(1),Real(130),dt,Real(1));assert(residual.valid&&residual.residual!=Real(0));
 auto negative=linearizeAnchoredRadiation(Real(1),Real(99),B0,Real(-1),dt,A);assert(negative.valid&&!negative.nonnegativeSlope&&negative.b==Real(-1)&&negative.f>Real(1));
 auto bad=linearizeAnchoredRadiation(Real(1),Real(90),B0,Real(-10),dt,A);assert(!bad.valid);
 auto nan=linearizeAnchoredRadiation(Real(0),B0,B0,std::numeric_limits<Real>::quiet_NaN(),dt,A);assert(!nan.valid);
 auto exact=materialRadiationResidual(Real(0),B0,A,Real(1),B0,dt,Real(0));assert(exact.valid&&exact.normalized==Real(0));
 // No iterate-difference enters the true material equation residual.
 auto stalled=materialRadiationResidual(Real(0),B0,A,Real(1),Real(130),dt,Real(1));assert(stalled.valid&&stalled.normalized==Real(1));
 std::cout<<"LINEARIZATION_PASS bits="<<sizeof(Real)*8<<" elimination bidirectional zero_derivative kappa_refresh signed_slope isotropic_integral actual_residual invalid_input\n";
}
