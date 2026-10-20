// Frozen test-only B reference: 821bd78c2cdd3ba58a1f11f3336eecbb7aec4b21
// Original function SHA256: 052a4bb5b742c732dc0aeba01550b9539aff6d7b4cd2aecf387fc8afd9161ac0
#pragma once
namespace ugkwp { namespace gaswall { namespace detail {
template<class R,int N,int M,class TableReal> UGKWP_GAS_HD inline bool assembleJacobianNodeReference(
 const WallInput<R,N,TableReal>& in,const WallModelConfig<R>& c,const LayerContext<R,N>& ctx,
 WallWorkspace<R,N,M>& w,int j){
    constexpr int B=N+4;PerturbedLayerState<R,N> perturbed(w.state,j);

        // A local simplex tangent basis remains feasible even when the globally
        // dependent species and an independent species are both exactly zero.
        R sum=0;for(int k=3;k<N+2;++k)sum+=perturbed.row[k];
        R donorMass=R(1)-sum;int donor=-1;
        for(int k=3;k<N+2;++k)if(perturbed.row[k]>donorMass){donorMass=perturbed.row[k];donor=k;}
        for(int pass=0;pass<B;++pass){const int col=donor>=3?(pass==3?donor:(pass==donor?3:pass)):pass;
            const bool species=col>=3&&col<N+2,transfer=species&&donor>=3&&col!=donor;
            const R old=perturbed.row[col],oldDonor=transfer?perturbed.row[donor]:R(0);
            const R rootEpsilon=::sqrt(std::numeric_limits<R>::epsilon());
            R variableScale=col==2?R(100):R(1);
            if(c.enableSst&&col==N+2)variableScale=::sqrt(max(R(1),in.matching.k))*rootEpsilon;
            if(c.enableSst&&col==N+3)variableScale=rootEpsilon/::sqrt(in.matching.omega);
            R h=rootEpsilon*max(abs(old),variableScale);
            if(species){h=min(h,donorMass/R(4));if(col==donor)h=-h;}
            bool valid=false;R rows[3][B]{};const int start=j>0?j-1:0,end=j+1<c.nodes?j+1:c.nodes-1;
            for(int attempt=0;attempt<2&&!valid;++attempt){perturbed.row[col]=old+h;
                if(transfer)perturbed.row[donor]=oldDonor-h;
                valid=true;
                for(int i=start;i<=end;++i)if(!layerRow(in,c,ctx,perturbed,w.y,i,rows[i-start])){valid=false;break;}
                if(!valid){perturbed.row[col]=old;if(transfer)perturbed.row[donor]=oldDonor;h=species?h/R(2):-h;}}
            const R actualStep=perturbed.row[col]-old;
            const R donorStep=transfer?oldDonor-perturbed.row[donor]:R(0);
            perturbed.row[col]=old;if(transfer)perturbed.row[donor]=oldDonor;
            if(!valid||actualStep==0)return false;
            for(int i=start;i<=end;++i)for(int row=0;row<B;++row){const R forward=(rows[i-start][row]-w.residual[i][row])/actualStep;
                R derivative=forward;
                // Direction e_col-e_donor: recover the original-coordinate
                // column using the donor column evaluated first above.
                R* block=j<i?w.lower[i][row]:(j==i?w.diagonal[i][row]:w.upper[i][row]);
                if(transfer)derivative+=(donorStep/actualStep)*block[donor];
                block[col]=derivative;}
        }
        return true;
}
} } }
