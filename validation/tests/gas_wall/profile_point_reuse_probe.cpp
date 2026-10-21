#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/reconstruction/ConservativeProfile.H"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace ugkwp::gaswall;using namespace ugkwp::gaswall::reconstruction;
struct OracleProfile:Profile{OracleProfile(const Profile&p):Profile(p){}
  UGKWP_GAS_HD physical::Primitive<double, 2>
  physicalPrimitive(double y, int side = 0) const {
    auto p = point(y, side), g = derivative(y, side);
    const auto &s = segments[segment(y, side)];
    physical::Primitive<double, 2> v;
    v.rho = p.rho;
    v.T = p.T;
    v.k = p.k;
    v.omega = p.omega;
    for (int j = 0; j < 2; ++j)
      v.Y[j] = p.Y[j];
    auto tangent = [&](const double *gradient, int direction) {
      double dot = 0;
      for (int j = 0; j < 3; ++j)
        dot += gradient[j] * in.normal[j];
      return gradient[direction] - dot * in.normal[direction];
    };
    for (int a = 0; a < 3; ++a) {
      v.U[a] = p.U[a];
      v.gradT[a] = g.T * in.normal[a];
      v.gradK[a] = g.k * in.normal[a];
      v.gradOmega[a] = g.omega * in.normal[a];
      for (int j = 0; j < 2; ++j)
        v.gradY[j][a] = g.Y[j] * in.normal[a];
      if (s.hasMacroGradients) {
        v.gradT[a] += tangent(s.macroGradients.T, a);
        v.gradK[a] += tangent(s.macroGradients.k, a);
        v.gradOmega[a] += tangent(s.macroGradients.omega, a);
        for (int j = 0; j < 2; ++j)
          v.gradY[j][a] += tangent(s.macroGradients.Y[j], a);
      }
      for (int b = 0; b < 3; ++b) {
        v.gradU[a][b] = g.U[a] * in.normal[b];
        if (s.hasMacroGradients)
          v.gradU[a][b] += tangent(s.macroGradients.U[a], b);
        else
          for (int lc = 0; lc < 3; ++lc)
            for (int ld = 0; ld < 2; ++ld)
              v.gradU[a][b] += (lc == 2 ? in.normal[a] : ctx.tangent[lc][a]) *
                               in.sstTangentialGradient[lc][ld] *
                               ctx.tangent[ld][b];
      }
    }
    return v;
  }
};
template<int M>void check(bool coherent,bool endpoint,int affine){CapturedSstWall m;auto cfg=m.config;cfg.nodes=M;cfg.coherentDiscretization=coherent;WallWorkspace<double,2,M>w{};for(int i=0;i<M;++i){double t=double(i)/(M-1);w.y[i]=m.input.matchingDistance*t*t;w.state[i][0]=std::log1p(20*t);w.state[i][1]=-.2*t;w.state[i][2]=25*t;w.state[i][3]=.8-.1*t;w.state[i][4]=.01+std::sin(2*t);w.state[i][5]=.01+t;}
 Profile p(m.input,cfg,w);for(int i=0;i<2;++i){Segment seg{};seg.lo=i?.4*w.y[M-1]:0;seg.hi=i?w.y[M-1]:.4*w.y[M-1];seg.algebraicOmega=i==0;seg.aRho=.05;seg.aT=-.02;seg.aK=.04;seg.aOmega=.03;seg.aY=.01;seg.aU[0]=-.1;seg.hasMacroGradients=i==1;seg.macroGradients.T[0]=3;seg.macroGradients.U[0][2]=2;p.segments.push_back(seg);if(endpoint){auto left=p.base(seg.lo),right=p.base(seg.hi);left.T*=.99;right.rho*=1.01;p.setEndpoints(i,left,right);}}
 p.affineSpeciesMode=affine;p.affineInterface=.4;p.affineSpeciesA=.1;p.affineSpeciesB=-.2;OracleProfile oracle(p);
 for(int side:{-1,0,1})for(double y:{0.,.1*w.y[1],.63*w.y[1],w.y[1],.4*w.y[M-1],.9*w.y[M-1],w.y[M-1]}){const auto&seg=p.segments[p.segment(y,side)];auto base=p.base(y);auto a=p.pointIn(y,seg),b=p.pointIn(y,seg,&base);assert(std::memcmp(&a,&b,sizeof a)==0);auto c=p.derivative(y,side),d=p.derivative(y,side,&base);assert(std::memcmp(&c,&d,sizeof c)==0);auto e=oracle.physicalPrimitive(y,side),f=p.physicalPrimitive(y,side);assert(std::memcmp(&e,&f,sizeof e)==0);}
}
int main(){for(bool coherent:{false,true})for(bool endpoint:{false,true})for(int affine:{0,1,2,3}){check<24>(coherent,endpoint,affine);check<48>(coherent,endpoint,affine);check<128>(coherent,endpoint,affine);}std::cout<<"48 profile paths with21 queries each match uncached primitive bits\n";}
