#ifndef UGKWP_MECHANISM_VALIDATION_H
#define UGKWP_MECHANISM_VALIDATION_H
#include "ChemistryTypes.H"
#include "../gasTransport/MixtureThermo.H"
namespace ugkwp {
namespace gasChemistryDetail {
UGKWP_GAS_HD inline bool validRange(int offset,int count,int total) noexcept {
    return count==0 || (offset>=0 && count>0 && count<=total && offset<=total-count);
}
template<class R,int N,class TableReal> UGKWP_GAS_HD inline bool orthogonalBasis
(const GasMechanismView<TableReal,N>& m,R* q) noexcept {
    const int rank=m.independentRank;
    if(rank<0 || rank>N || (rank && !m.stoichiometricBasis)) return false;
    for(int j=0;j<rank;++j) {
        for(int s=0;s<N;++s) { R v=m.stoichiometricBasis[s*rank+j]; if(!gasFinite(v))return false;q[s*N+j]=v; }
        for(int pass=0;pass<2;++pass) for(int k=0;k<j;++k) {
            R dot=0;for(int s=0;s<N;++s)dot+=q[s*N+j]*q[s*N+k];
            for(int s=0;s<N;++s)q[s*N+j]-=dot*q[s*N+k];
        }
        R norm=0;for(int s=0;s<N;++s)norm+=q[s*N+j]*q[s*N+j];
        if(!gasFinite(norm) || norm==R(0)) {
            R scale=0;for(int s=0;s<N;++s)scale=gasThermoDetail::maxValue(scale,gasThermoDetail::absValue(q[s*N+j]));
            if(!gasFinite(scale) || scale==R(0))return false;
            norm=0;for(int s=0;s<N;++s){const R value=q[s*N+j]/scale;norm+=value*value;}
            norm=::sqrt(norm);
            for(int s=0;s<N;++s)q[s*N+j]=(q[s*N+j]/scale)/norm;
        } else {norm=::sqrt(norm);for(int s=0;s<N;++s)q[s*N+j]/=norm;}
    }
    return true;
}
}
namespace gasChemistryDetail {
// The checks accompany the reaction being consumed, rather than rescanning a
// whole mechanism before each scalar/AD evaluation.
template<class R,int N> UGKWP_GAS_HD inline bool reactionInputAvailable
(const GasMechanismView<R,N>& m,int r) noexcept {
    if(r<0 || r>=m.reactionCount || !m.reactions)return false;
    const auto& a=m.reactions[r];
    if(!validRange(a.reactantOffset,a.reactantCount,m.reactantTermCount)
      || !validRange(a.productOffset,a.productCount,m.productTermCount)
      || (a.reactantCount && !m.reactants) || (a.productCount && !m.products)
      || (a.type!=GasReactionType::Elementary && a.type!=GasReactionType::ThirdBody
          && a.type!=GasReactionType::Lindemann && a.type!=GasReactionType::Troe))return false;
    for(int side=0;side<2;++side) {
        const auto* terms=side?m.products:m.reactants;
        const int offset=side?a.productOffset:a.reactantOffset;
        const int count=side?a.productCount:a.reactantCount;
        for(int j=0;j<count;++j)if(terms[offset+j].species<0 || terms[offset+j].species>=N)return false;
    }
    return a.type==GasReactionType::Elementary
        || (validRange(a.efficiencyOffset,a.efficiencyCount,m.efficiencyCount)
            && (!a.efficiencyCount || m.efficiencies));
}
}
// Explicit inspection remains available to loaders; no conservation, identity,
// duplicate-declaration or basis proof is needed to read a reaction.
template<class R,int N> UGKWP_GAS_HD inline ChemistryStatus validateGasMechanismView
(const SpeciesThermoView<R,N>& t,const GasMechanismView<R,N>& m) noexcept {
    (void)t;ChemistryStatus st;
    if(m.reactionCount<0 || (m.reactionCount && !m.reactions)) {st.code=ChemistryCode::InvalidModel;return st;}
    for(int r=0;r<m.reactionCount;++r)if(!gasChemistryDetail::reactionInputAvailable(m,r)) {
        st.code=ChemistryCode::InvalidModel;st.reaction=r;return st;
    }
    return st;
}
}
#endif
