// Uncached differential oracle for the abe202ad Jacobian policy.
// Earlier 821bd78c policy predates zero-k probe repair and thermo extrapolation.
// Keep the current probe/failure policy here; this oracle tests cache lifetime,
// immutable state, call ordering and partial block writes, not SST model physics.
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
        // EquilibriumOde's two unused SST rows are exact identity equations.
        // They do not affect transport; avoid two full thermochemistry/flux
        // probes per node while preserving the existing block solve/order.
        const bool algebraicSst=c.model==BoundaryLayerModel::EquilibriumOde&&!c.enableSst;
        if(algebraicSst){w.diagonal[j][N+2][N+2]=R(1);w.diagonal[j][N+3][N+3]=R(1);}
        const int columns=algebraicSst?N+2:B;
        for(int pass=0;pass<columns;++pass){const int col=donor>=3?(pass==3?donor:(pass==donor?3:pass)):pass;
            const bool species=col>=3&&col<N+2,transfer=species&&donor>=3&&col!=donor;
            const R old=perturbed.row[col],oldDonor=transfer?perturbed.row[donor]:R(0);
            const R rootEpsilon=::sqrt(std::numeric_limits<R>::epsilon());
            R variableScale=col==2?R(100):R(1);
            // Match the k residual scale so an exactly zero boundary k still
            // has a representable probe of its linear sqrt(k) constraint.
            if(c.enableSst&&col==N+2)variableScale=::sqrt(max(R(1),in.matching.k))*rootEpsilon;
            if(c.enableSst&&col==N+3)variableScale=rootEpsilon/::sqrt(in.matching.omega);
            R h=rootEpsilon*max(abs(old),variableScale);
            if(species){h=min(h,donorMass/R(4));if(col==donor)h=-h;}
            bool valid=false;R rows[3][B]{};const int start=j>0?j-1:0,end=j+1<c.nodes?j+1:c.nodes-1;
            auto derivativeInput=in;derivativeInput.diagnostics=nullptr;
            derivativeInput.model.thermo.diagnostics=nullptr;
            for(int attempt=0;attempt<2&&!valid;++attempt){
                perturbed.row[col]=old+h;
                if(transfer)perturbed.row[donor]=oldDonor-h;
                if(!ctx.trialStorageComplete
                    &&!layerTrialStorageAvailable(in,in.temperature+perturbed.row[2],j))return false;
                valid=true;
                for(int i=start;i<=end;++i)
                    if(!layerRow(derivativeInput,c,ctx,perturbed,w.y,i,rows[i-start])){valid=false;break;}
                if(!valid){
                    perturbed.row[col]=old;if(transfer)perturbed.row[donor]=oldDonor;
                    // The opposite direction keeps a valid interior base point;
                    // simplex transfers instead reduce their feasible step.
                    h=species?h/R(2):-h;
                }
            }
            const R actualStep=perturbed.row[col]-old;
            const R donorStep=transfer?oldDonor-perturbed.row[donor]:R(0);
            perturbed.row[col]=old;if(transfer)perturbed.row[donor]=oldDonor;
            if(!valid)return false;
            if(actualStep==0)return wallFatal(in,ugkp::diagnostics::Code::ZeroDenominator,2418,actualStep,0,col);
            // A nonzero sqrt(k) can have an entirely unresolvable relative
            // difference when neighboring fluxes dominate this column. Rebuild
            // only a zero-signal column with a symmetric physical-scale probe;
            // a true sqrt(k)=0 derivative remains exactly zero by symmetry.
            bool zeroSignal=c.enableSst&&col==N+2&&old!=R(0);
            // A coupled omega/energy row can change by roundoff while every
            // k equation remains unresolved. Only the k rows can certify this
            // probe; otherwise their zero column makes the Newton block singular.
            if(zeroSignal)for(int i=start;i<=end;++i)
                if(rows[i-start][N+2]!=w.residual[i][N+2])zeroSignal=false;
            R opposite[3][B]{};R denominator=actualStep;
            if(zeroSignal){
                // The existing k residual normalization uses max(1,k_match).
                // cbrt(epsilon) balances central O(h^2) truncation and the
                // O(epsilon/h) subtraction error, without a fitted threshold.
                h=::cbrt(std::numeric_limits<R>::epsilon())
                    *max(abs(old),::sqrt(max(R(1),in.matching.k)));
                perturbed.row[col]=old+h;const R plusStep=perturbed.row[col]-old;
                bool resolved=true;
                for(int i=start;i<=end;++i)if(!layerRow(derivativeInput,c,ctx,perturbed,w.y,i,rows[i-start])){resolved=false;break;}
                perturbed.row[col]=old-h;denominator=plusStep-(perturbed.row[col]-old);

                if(resolved)for(int i=start;i<=end;++i)if(!layerRow(derivativeInput,c,ctx,perturbed,w.y,i,opposite[i-start])){resolved=false;break;}
                perturbed.row[col]=old;
                if(!resolved)return false;
            }
            for(int i=start;i<=end;++i)for(int row=0;row<B;++row){const R forward=(rows[i-start][row]-(zeroSignal?opposite[i-start][row]:w.residual[i][row]))/denominator;
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
