#ifndef UGKWP_MECHANISM_VALIDATION_H
#define UGKWP_MECHANISM_VALIDATION_H
#include "ChemistryTypes.H"
#include "../gasTransport/MixtureThermo.H"
namespace ugkwp {
namespace gasChemistryDetail {
template<class R> UGKWP_GAS_HD inline bool validArrhenius(const GasArrheniusData<R>& a) noexcept {
    return gasFinite(a.preExponential) && a.preExponential>=R(0)
        && gasFinite(a.temperatureExponent) && gasFinite(a.activationEnergy);
}
UGKWP_GAS_HD inline bool validRange(int offset,int count,int total) noexcept {
    return offset>=0 && count>=0 && count<=total && offset<=total-count;
}
template<class R> UGKWP_GAS_HD inline bool sameTerms(const GasStoichTerm<R>* a,int ao,int an,const GasStoichTerm<R>* b,int bo,int bn) noexcept {
 if(an!=bn)return false;
 for(int i=0;i<an;++i){bool found=false;for(int j=0;j<bn;++j)if(a[ao+i].species==b[bo+j].species && a[ao+i].coefficient==b[bo+j].coefficient)found=true;if(!found)return false;}return true;
}
template<class R,int N> UGKWP_GAS_HD inline bool matchingChannels(const GasReactionData<R>& a,const GasReactionData<R>& b,const GasMechanismView<R,N>& m) noexcept {
 if(a.type!=b.type)return false;
 bool same=sameTerms(m.reactants,a.reactantOffset,a.reactantCount,m.reactants,b.reactantOffset,b.reactantCount)
     && sameTerms(m.products,a.productOffset,a.productCount,m.products,b.productOffset,b.productCount);
 bool reverse=(a.reversible || b.reversible) && sameTerms(m.reactants,a.reactantOffset,a.reactantCount,m.products,b.productOffset,b.productCount)
     && sameTerms(m.products,a.productOffset,a.productCount,m.reactants,b.reactantOffset,b.reactantCount);
 if(!same && !reverse)return false;
 if(a.type==GasReactionType::Elementary)return true;
 for(int s=0;s<N;++s){R ae=a.defaultEfficiency,be=b.defaultEfficiency;
   for(int i=0;i<a.efficiencyCount;++i)if(m.efficiencies[a.efficiencyOffset+i].species==s)ae=m.efficiencies[a.efficiencyOffset+i].efficiency;
   for(int i=0;i<b.efficiencyCount;++i)if(m.efficiencies[b.efficiencyOffset+i].species==s)be=m.efficiencies[b.efficiencyOffset+i].efficiency;
   if(ae>R(0) && be>R(0))return true;
 }return false;
}
template<class R,int N> UGKWP_GAS_HD inline bool orthogonalBasis
(const GasMechanismView<R,N>& m,R* q) noexcept {
    const int rank=m.independentRank;
    if(rank<0 || rank>N || (rank && !m.stoichiometricBasis)) return false;
    const R tol=R(64)*std::numeric_limits<R>::epsilon();
    for(int j=0;j<rank;++j) {
        R original=0;
        for(int s=0;s<N;++s) { R v=m.stoichiometricBasis[s*rank+j]; if(!gasFinite(v))return false;q[s*N+j]=v;original+=v*v; }
        for(int pass=0;pass<2;++pass) for(int k=0;k<j;++k) {
            R dot=0;for(int s=0;s<N;++s)dot+=q[s*N+j]*q[s*N+k];
            for(int s=0;s<N;++s)q[s*N+j]-=dot*q[s*N+k];
        }
        R norm=0;for(int s=0;s<N;++s)norm+=q[s*N+j]*q[s*N+j];
        if(!gasFinite(norm) || !(norm>tol*tol*original))return false;
        norm=::sqrt(norm);for(int s=0;s<N;++s)q[s*N+j]/=norm;
    }
    return true;
}
}
// Host model-load gate, also allocation-free/device-callable for independent
// consumers. Checks every channel, standard state, conservative span and rank.
template<class R,int N> UGKWP_GAS_HD inline ChemistryStatus validateGasMechanismView
(const SpeciesThermoView<R,N>& t,const GasMechanismView<R,N>& m) noexcept {
    using namespace gasChemistryDetail;using gasThermoDetail::absValue;
    ChemistryStatus st; st.code=ChemistryCode::InvalidModel;
    if(validateSpeciesThermoView(t)!=ThermoStatus::Success || m.reactionCount<0
      || m.reactantTermCount<0 || m.productTermCount<0 || m.efficiencyCount<0
      || (m.reactionCount && !m.reactions) || (m.reactantTermCount && !m.reactants)
      || (m.productTermCount && !m.products) || (m.efficiencyCount && !m.efficiencies)
      || !gasFinite(m.referencePressure) || !(m.referencePressure>R(0))
      || t.speciesOrderHash!=m.speciesOrderHash) return st;
    R q[N*N]={};if(!orthogonalBasis(m,q))return st;
    const R tol=R(512)*std::numeric_limits<R>::epsilon();
    // Basis itself may not invent additional nonconservative motion.
    for(int j=0;j<m.independentRank;++j) {
        R mass=0,scale=0;
        for(int s=0;s<N;++s){const R a=q[s*N+j]*t.species[s].molarMass;mass+=a;scale+=absValue(a);}
        if(absValue(mass)>tol*scale)return st;
        for(int e=0;e<t.elementCount;++e){R v=0,z=0;for(int s=0;s<N;++s){R a=q[s*N+j]*t.elementComposition[s*t.elementCount+e];v+=a;z+=absValue(a);}if(absValue(v)>tol*z)return st;}
    }
    for(int r=0;r<m.reactionCount;++r) {
        st.reaction=r;const GasReactionData<R>& a=m.reactions[r];
        if(!validRange(a.reactantOffset,a.reactantCount,m.reactantTermCount)
          || !validRange(a.productOffset,a.productCount,m.productTermCount)
          || !validRange(a.efficiencyOffset,a.efficiencyCount,m.efficiencyCount)
          || a.reactantCount==0 || a.productCount==0 || !validArrhenius(a.highRate)
          || !gasFinite(a.defaultEfficiency) || a.defaultEfficiency<R(0))return st;
        if(a.type!=GasReactionType::Elementary && a.type!=GasReactionType::ThirdBody
          && a.type!=GasReactionType::Lindemann && a.type!=GasReactionType::Troe)return st;
        if(a.type==GasReactionType::Lindemann || a.type==GasReactionType::Troe)
          if(!validArrhenius(a.lowRate))return st;
        if(a.type==GasReactionType::Troe && (!gasFinite(a.troe.alpha)
          || a.troe.alpha<R(0) || a.troe.alpha>R(1) || !gasFinite(a.troe.T1)
          || !gasFinite(a.troe.T3) || !(a.troe.T1>R(0)) || !(a.troe.T3>R(0))
          || (a.troe.hasT2 && (!gasFinite(a.troe.T2) || a.troe.T2<R(0)))))return st;
        R nu[N]={};
        for(int side=0;side<2;++side) {
          const auto* terms=side?m.products:m.reactants;int off=side?a.productOffset:a.reactantOffset;int count=side?a.productCount:a.reactantCount;
          for(int k=0;k<count;++k){const auto& v=terms[off+k];if(v.species<0 || v.species>=N || !gasFinite(v.coefficient) || !(v.coefficient>R(0)))return st;
            for(int p=0;p<k;++p)if(terms[off+p].species==v.species)return st;
            nu[v.species]+=(side?R(1):R(-1))*v.coefficient;
            if(a.reversible){const auto& sp=t.species[v.species];if(sp.referencePressure!=m.referencePressure || (sp.model==SpeciesThermoModel::LinearCp && !sp.hasEntropyReference))return st;}
          }
        }
        for(int k=0;k<a.efficiencyCount;++k){const auto& e=m.efficiencies[a.efficiencyOffset+k];if(e.species<0 || e.species>=N || !gasFinite(e.efficiency) || e.efficiency<R(0))return st;for(int p=0;p<k;++p)if(m.efficiencies[a.efficiencyOffset+p].species==e.species)return st;}
        R mass=0,scale=0;for(int s=0;s<N;++s){R v=nu[s]*t.species[s].molarMass;mass+=v;scale+=absValue(v);}
        if(absValue(mass)>tol*scale)return st;
        for(int e=0;e<t.elementCount;++e){R v=0,z=0;for(int s=0;s<N;++s){R a=nu[s]*t.elementComposition[s*t.elementCount+e];v+=a;z+=absValue(a);}if(absValue(v)>tol*z)return st;}
        R residual[N],norm=0;for(int s=0;s<N;++s){residual[s]=nu[s];norm+=nu[s]*nu[s];}
        if(!(norm>R(0)))return st;
        for(int j=0;j<m.independentRank;++j){R v=0;for(int s=0;s<N;++s)v+=nu[s]*q[s*N+j];for(int s=0;s<N;++s)residual[s]-=v*q[s*N+j];}
        R error=0;for(int s=0;s<N;++s)error+=residual[s]*residual[s];
        if(error>tol*tol*norm)return st;
    }
    // Only compare after all ranges are validated. Declared duplicate channels
    // remain separate; absent flags and orphan flags are both malformed.
    for(int r=0;r<m.reactionCount;++r){st.reaction=r;bool match=false;
      for(int j=0;j<m.reactionCount;++j)if(j!=r && matchingChannels(m.reactions[r],m.reactions[j],m)){
        match=true;if(!m.reactions[r].duplicate || !m.reactions[j].duplicate)return st;}
      if(m.reactions[r].duplicate && !match)return st;
    }
    st.code=ChemistryCode::Success;st.reaction=-1;return st;
}
}
#endif
