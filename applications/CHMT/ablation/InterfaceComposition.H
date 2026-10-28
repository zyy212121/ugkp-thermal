#ifndef CHMT_ABLATION_INTERFACECOMPOSITION_H
#define CHMT_ABLATION_INTERFACECOMPOSITION_H
// Included inside namespace chmt, after compositionResidual.
// On the composition simplex the nonlinear equations are equivalent to the
// real eigenvectors of a diagonal plus rank-two matrix. Use real-root and
// critical-point isolation, then certify the original equations, not P alone.
namespace composition_detail {

    struct Poly {
        int degree=0;
        Real c[Ns+1]{};
    };

    CHMT_HD inline Poly product(const Real* left,const Real* right,int skip1=-1,int skip2=-1){
        Poly p;
        p.c[0]=1;
        for(int i=0;i<Ns;++i)if(i!=skip1&&i!=skip2){
            Poly r;
            r.degree=p.degree+1;
            for(int k=0;k<=p.degree;++k){
                r.c[k]+=(r.degree-k)/Real(r.degree)*left[i]*p.c[k];
                r.c[k+1]+=(k+1)/Real(r.degree)*right[i]*p.c[k];
            }
            p=r;
        }
        return p;
    }

    CHMT_HD inline Poly elevate(Poly p){
        while(p.degree<Ns){
            Poly r;
            r.degree=p.degree+1;
            r.c[0]=p.c[0];
            r.c[r.degree]=p.c[p.degree];
            for(int k=1;k<r.degree;++k)r.c[k]=k/Real(r.degree)*p.c[k-1]+(r.degree-k)/Real(r.degree)*p.c[k];
            p=r;
        }
        return p;
    }

    CHMT_HD inline void add(Poly& p,const Poly& q,Real factor){
        for(int k=0;k<=Ns;++k)p.c[k]+=q.c[k]*factor;
    }

    CHMT_HD inline Poly multiplyLinear(const Poly& p,Real left,Real right) {
        Poly result;
        result.degree=p.degree+1;
        for(int k=0;k<=p.degree;++k){
            result.c[k]+=(result.degree-k)/Real(result.degree)*left*p.c[k];
            result.c[k+1]+=(k+1)/Real(result.degree)*right*p.c[k];
        }
        return result;
    }

    CHMT_HD inline Poly polynomial(Real left,Real right,const Real* d,
    const Real* u,const Real* v,const Real* r) {
        // Rank-two determinant recurrence, O(Ns^2), entirely in the local Bernstein
        // basis. P is the diagonal product; A,B,C,D are its one-factor removals and
        // Q its rank-two minors. No monomial expansion or eigensystem is formed.
        Real uScale=1;
        for(int i=0;i<Ns;++i)uScale=maxValue(uScale,absValue(u[i]));
        Poly P,A,B,C,D,Q;
        P.c[0]=1;
        for(int i=0;i<Ns;++i){
            Real ui=u[i]/uScale;
            Real l=left+d[i],h=right+d[i];
            Poly nextP=multiplyLinear(P,l,h),nextA,nextB,nextC,nextD,nextQ;
            if(i){
                nextA=multiplyLinear(A,l,h);
                nextB=multiplyLinear(B,l,h);
                nextC=multiplyLinear(C,l,h);
                nextD=multiplyLinear(D,l,h);
            }
            for(int k=0;k<=P.degree;++k){
                nextA.c[k]+=r[i]*ui*P.c[k];
                nextB.c[k]+=v[i]*P.c[k];
                nextC.c[k]+=ui*P.c[k];
                nextD.c[k]+=r[i]*v[i]*P.c[k];
            }
            nextA.degree=nextB.degree=nextC.degree=nextD.degree=i;
            if(i){
                if(i>1)nextQ=multiplyLinear(Q,l,h);
                nextQ.degree=i-1;
                for(int k=0;k<=A.degree;++k)
                nextQ.c[k]+=v[i]*(A.c[k]-r[i]*C.c[k])+ui*(r[i]*B.c[k]-D.c[k]);
            }
            P=nextP;
            A=nextA;
            B=nextB;
            C=nextC;
            D=nextD;
            Q=nextQ;
        }
        add(P,elevate(B),-1);
        for(int k=0;k<=Ns;++k)P.c[k]/=uScale;
        add(P,elevate(A),-1);
        if(Ns>1)add(P,elevate(Q),1);
        Real maximum=0;
        for(int k=0;k<=Ns;++k)maximum=maxValue(maximum,absValue(P.c[k]));
        if(maximum>0)for(int k=0;k<=Ns;++k)P.c[k]/=maximum;
        return P;
    }

    CHMT_HD inline int variations(const Poly& p){
        int last=0,v=0;
        for(int k=0;k<=p.degree;++k){
            int s=p.c[k]>0?1:(p.c[k]<0?-1:0);
            if(s){
                if(last&&s!=last)++v;
                last=s;
            }
        }
        return v;
    }

    CHMT_HD inline void subdivide(Poly p,Poly& left,Poly& right){
        left.degree=right.degree=p.degree;
        left.c[0]=p.c[0];
        right.c[p.degree]=p.c[p.degree];
        for(int k=1;k<=p.degree;++k){
            for(int j=0;j<=p.degree-k;++j)p.c[j]=(p.c[j]+p.c[j+1])/2;
            left.c[k]=p.c[0];
            right.c[p.degree-k]=p.c[p.degree-k];
        }
    }

    struct Interval {
        Real lo=0,hi=1;
        Poly p;
        int depth=0;
    };

    enum class SolveState {
        NoCertifiedCandidate,Converged,AdmissibleApproximation
    };

    struct Info {
        SolveState state=SolveState::NoCertifiedCandidate;
        Real actualResidual=0;
        int structuredIterations=0;bool unique=false;
        int polynomialCount=0,nodeCount=0,candidates=0,legal=0,ambiguous=0,refine=0,criticalCandidates=0;
        Real maxResidual=0;
    };

    constexpr int ScalarCandidateCapacity=Ns*(Ns+1)/2+2;

    CHMT_HD inline Real polynomialValue(Poly p,Real t) {
        for(int k=p.degree;k>0;--k)for(int j=0;j<k;++j)p.c[j]=(1-t)*p.c[j]+t*p.c[j+1];
        return p.c[0];
    }

    CHMT_HD inline Real polynomialValueDerivative(Poly p,Real t,Real& derivative) {
        for(int k=p.degree;k>1;--k)for(int j=0;j<k;++j)p.c[j]=(1-t)*p.c[j]+t*p.c[j+1];
        derivative=p.degree*(p.c[1]-p.c[0]);return (1-t)*p.c[0]+t*p.c[1];
    }

    CHMT_HD inline int uniqueCandidates(Real* points,int count) {
        for(int i=1;i<count;++i){
            Real value=points[i];
            int j=i;
            while(j&&points[j-1]>value){
                points[j]=points[j-1];
                --j;
            }
            points[j]=value;
        }
        int n=0;
        for(int i=0;i<count;++i)if(points[i]>0&&points[i]<1&&(!n||points[i]!=points[n-1]))points[n++]=points[i];
        return n;
    }

    CHMT_HD inline void appendCandidate(Real* points,int& count,Real value) {
        if(!(value>0&&value<1))return;
        for(int j=0;j<count;++j)if(points[j]==value)return;
        if(count<ScalarCandidateCapacity)points[count++]=value;
    }
    // A multiple real root is a stationary point. A rounded polynomial may have
    // a tiny complex conjugate pair instead, so a sign-only root finder loses it.
    // The derivative hierarchy supplies all critical points without sampling:
    // roots of each derivative partition the next polynomial into monotone pieces.
    // Keep lower-level critical points as well, which covers higher multiplicity.
    // These are only mathematical candidates; the original equations certify them.

    CHMT_HD inline int criticalPoints(Poly original,Real* points) {
        Poly hierarchy[Ns];
        hierarchy[0]=original;
        for(int level=1;level<Ns;++level){
            const Poly& previous=hierarchy[level-1];
            Poly& p=hierarchy[level];
            p.degree=previous.degree-1;
            for(int j=0;j<=p.degree;++j)p.c[j]=previous.degree*(previous.c[j+1]-previous.c[j]);
        }
        int count=0;
        for(int level=Ns-1;level>0;--level){
            const Poly& p=hierarchy[level];
            Real previous[ScalarCandidateCapacity]{};
            int n=count;
            for(int j=0;j<n;++j)previous[j]=points[j];
            for(int segment=0;segment<=n;++segment){
                Real lo=segment?previous[segment-1]:0,hi=segment<n?previous[segment]:1;
                Real fl=polynomialValue(p,lo),fh=polynomialValue(p,hi);
                if(fl==0)appendCandidate(points,count,lo);
                if(fh==0)appendCandidate(points,count,hi);
                if(fl==0||fh==0||(fl>0)==(fh>0))continue;
                for(int iteration=0;iteration<80;++iteration){
                    Real mid=(lo+hi)/2;
                    if(mid==lo||mid==hi)break;
                    Real fm=polynomialValue(p,mid);
                    if(fm==0){
                        lo=hi=mid;
                        break;
                    }
                    if((fm>0)==(fl>0)){
                        lo=mid;
                        fl=fm;
                    }
                    else hi=mid;
                }
                appendCandidate(points,count,(lo+hi)/2);
            }
            count=uniqueCandidates(points,count);
        }
        return count;
    }

    CHMT_HD inline int roots(Poly p,Info& info,Real* output,bool* genuine=nullptr){
        int count=0;if(genuine)for(int i=0;i<ScalarCandidateCapacity;++i)genuine[i]=false;
        if(Ns==2){
            // A quadratic is the complete scalar problem for two species.
            const Real A=(p.c[0]-p.c[1])+(p.c[2]-p.c[1]);
            const Real B=2*(p.c[1]-p.c[0]),C=p.c[0];
            if(A==0){
                if(B!=0){
                    Real t=-C/B;
                    if(t>0&&t<1)output[count++]=t;
                }
                if(genuine)for(int i=0;i<count;++i)genuine[i]=true;
                return count;
            }
            const Real discriminant=::fma(B,B,-4*A*C);
            if(discriminant<0){
                Real t=-B/(2*A);
                if(t>0&&t<1){
                    output[count++]=t;
                    ++info.criticalCandidates;
                }
                return count;
            }
            const Real q=-Real(.5)*(B+::copysign(::sqrt(discriminant),B));
            if(q==0){
                Real t=-B/(2*A);
                if(t>0&&t<1)output[count++]=t;
            }
            else {
                Real t=q/A;
                if(t>0&&t<1)output[count++]=t;
                t=C/q;
                if(t>0&&t<1 && (!count||t!=output[0]))output[count++]=t;
            }
            if(count==2&&output[0]>output[1]){
                Real t=output[0];
                output[0]=output[1];
                output[1]=t;
            }
            if(genuine)for(int i=0;i<count;++i)genuine[i]=true;
            return count;
        }
        bool lostVariation=false;
        int size=0;
        Interval stack[Ns+1];
        stack[size++]={0,1,p,0};
        while(size){
            Interval a=stack[--size];
            ++info.nodeCount;
            int v=variations(a.p);
            if(!v)continue;
            if(v==1){
                // The interval already isolates one root. Safeguarded Newton
                // refines that same root; a step outside the bracket uses its
                // midpoint. No physical state, time or target is retried.
                Real lo=0,hi=1,t=.5;
                Real sign=0;for(int k=0;k<=Ns;++k)if(a.p.c[k]!=0){sign=a.p.c[k];break;}
                bool settled=false;
                for(int it=0;it<80;++it){
                    Real derivative=0,value=polynomialValueDerivative(a.p,t,derivative);
                    if(value==0){settled=true;break;}
                    if((value>0)==(sign>0))lo=t;else hi=t;
                    Real quotient=0,next=lo+(hi-lo)/2;
                    if(derivative!=0&&quotientFinite(value,derivative,quotient)){
                        Real proposed=t-quotient;if(finite(proposed)&&proposed>lo&&proposed<hi)next=proposed;
                    }
                    if(next==t||next==lo||next==hi){settled=true;break;}
                    t=next;
                }
                if(!settled)++info.ambiguous;
                if(count<ScalarCandidateCapacity)output[count++]=a.lo+(a.hi-a.lo)*t;
                else ++info.ambiguous;
                continue;
            }
            Real mid=(a.lo+a.hi)/2;
            if(a.depth>=80||mid==a.lo||mid==a.hi){
                ++info.ambiguous;
                continue;
            }
            Poly l,r;
            subdivide(a.p,l,r);
            if(l.c[Ns]==0){
                if(count<ScalarCandidateCapacity)output[count++]=mid;
                else ++info.ambiguous;
            }
            const int vr=variations(r),vl=variations(l);
            lostVariation=lostVariation || v-vl-vr>1;
            if(size+(vr>0)+(vl>0)>Ns+1){
                ++info.ambiguous;
                continue;
            }
            if(vr)stack[size++]={mid,a.hi,r,a.depth+1};
            if(vl)stack[size++]={a.lo,mid,l,a.depth+1};
        }
        if(genuine)for(int i=0;i<count;++i)genuine[i]=true;
        if(lostVariation){
            Real critical[ScalarCandidateCapacity]{};
            int n=criticalPoints(p,critical);
            for(int i=0;i<n&&count<ScalarCandidateCapacity;++i){
                output[count++]=critical[i];
                ++info.criticalCandidates;
            }
        }
        return count;
    }

    CHMT_HD inline bool linear(Real a[Ns][Ns+1],Real* dy) {
        for(int col=0;col<Ns;++col){
            int pivot=col;
            for(int row=col+1;row<Ns;++row)if(absValue(a[row][col])>absValue(a[pivot][col]))pivot=row;
            if(!finite(a[pivot][col])||a[pivot][col]==0)return false;
            for(int k=col;k<=Ns;++k){
                Real tmp=a[col][k];
                a[col][k]=a[pivot][k];
                a[pivot][k]=tmp;
            }
            const Real diagonal=a[col][col];
            for(int k=col;k<=Ns;++k)if(!quotientFinite(a[col][k],diagonal,a[col][k]))return false;
            for(int row=0;row<Ns;++row)if(row!=col){
                const Real factor=a[row][col];
                for(int k=col;k<=Ns;++k){
                    Real term=0;
                    if(!productFinite(factor,a[col][k],term)||!sumFinite(a[row][k],-term,a[row][k]))return false;
                }
            }
        }
        for(int i=0;i<Ns;++i)dy[i]=a[i][Ns];
        return true;
    }

    struct Selection {
        bool certified=false,converged=false;
        Real distance=0,residual=0,value[Ns]{};
    };

    CHMT_HD inline bool certify(Real* Y,const Real* bulk,const Real* Js,Real pressure,
    Real T,Real distance,const PhysicsConfig& supplied,Real tolerance,Selection& best, bool* solved=nullptr,bool allowApproximation=true,
    const CompositionResidualInputs* prepared=nullptr) {
        if(solved)*solved=false;
        Real sum=0;
        for(int i=0;i<Ns;++i){
            if(!finite(Y[i])||Y[i]<0||Y[i]>1)return false;
            sum+=Y[i];
        }
        // This bound describes only floating-point normalization error, never a
        // physical acceptance tolerance.  No normalization/clipping repairs a root.
        const Real roundoff=Ns*2.22044604925031308085e-16/(1-Ns*2.22044604925031308085e-16);
        if(absValue(sum-1)>roundoff*2)return false;
        const PhysicsConfig& p=supplied;
        Real f[Ns]{};
        if(!compositionResidualPolicy<false>(Y,bulk,Js,pressure,T,distance,p,f,nullptr,prepared))return false;
        Real norm=0,metric=0;
        for(int i=0;i<Ns;++i){
            norm=maxValue(norm,absValue(f[i]));
            Real d=Y[i]-bulk[i];
            metric+=d*d;
        }
        if(!finite(norm)||!finite(metric))return false;
        bool converged=norm<=tolerance;
        if(solved)*solved=converged;
        if((converged||allowApproximation) && (!best.certified || (converged&&!best.converged)
        || (converged==best.converged && (converged ? metric<best.distance : norm<best.residual)))){
            best.certified=true;
            best.converged=converged;
            best.distance=metric;
            best.residual=norm;
            for(int i=0;i<Ns;++i)best.value[i]=Y[i];
        }
        return true;
    }

    CHMT_HD inline void refine(Real* Y,const Real* bulk,const Real* Js,Real pressure,
    Real T,Real distance,const PhysicsConfig& supplied,Info& info,const CompositionResidualInputs* prepared=nullptr) {
        const PhysicsConfig& p=supplied;
        for(int it=0;it<8;++it){
            Real f[Ns]{},jac[Ns*Ns]{};
            if(!compositionResidualPolicy<false>(Y,bulk,Js,pressure,T,distance,p,f,jac,prepared))return;
            ++info.refine;
            Real m[Ns][Ns+1]{},sum=0;
            for(int i=0;i<Ns;++i)sum+=Y[i];
            for(int i=0;i<Ns;++i){
                for(int j=0;j<Ns;++j)m[i][j]=i==Ns-1?1:jac[i*Ns+j];
                m[i][Ns]=i==Ns-1?1-sum:-f[i];
            }
            Real dy[Ns]{};
            if(!linear(m,dy))return;
            Real md=0;
            for(int i=0;i<Ns;++i)md=maxValue(md,absValue(dy[i]));
            if(md<2.22044604925031308085e-16)return;
            Real next[Ns]{};
            for(int i=0;i<Ns;++i)if(!sumFinite(Y[i],dy[i],next[i]))return;
            for(int i=0;i<Ns;++i)Y[i]=next[i];
        }
    }

    CHMT_HD inline bool recoverRoot(Real z,const Real* d,const Real* u,const Real* v,
    const Real* r,Real* Y) {
        Real A=0,B=0,AR=0,BR=0;
        for(int i=0;i<Ns;++i){
            Real den=z+d[i],a=0,b=0,ar=0,br=0;
            if(!quotientFinite(u[i],den,a)||!quotientFinite(v[i],den,b)
            ||!productFinite(r[i],a,ar)||!productFinite(r[i],b,br)
            ||!sumFinite(A,a,A)||!sumFinite(B,b,B)||!sumFinite(AR,ar,AR)||!sumFinite(BR,br,BR))return false;
        }
        Real second=0,rhs=0;
        if(!sumFinite(1,-AR,second)||!sumFinite(1,-B,rhs))return false;
        Real scale=maxValue(absValue(A),absValue(second));
        if(!(scale>0)||!finite(scale))return false;
        Real x=A/scale,y=second/scale,firstRhs=0,secondRhs=0,first=0,last=0,numerator=0,rm=0;
        if(!quotientFinite(rhs,scale,firstRhs)||!quotientFinite(BR,scale,secondRhs)
        ||!productFinite(x,firstRhs,first)||!productFinite(y,secondRhs,last)
        ||!sumFinite(first,last,numerator)||!quotientFinite(numerator,x*x+y*y,rm))return false;
        for(int i=0;i<Ns;++i){
            Real numerator=0;
            if(!productFinite(u[i],rm,numerator)||!sumFinite(numerator,v[i],numerator)
            ||!quotientFinite(numerator,z+d[i],Y[i]))return false;
        }
        return true;
    }
    // Euclidean projection onto a prescribed mass simplex.  Subtract the largest
    // level before summation; only active levels are added, so no large common
    // multiplier can overflow this audit-free mathematical calculation.

    CHMT_HD inline void projectMass(const Real* bulk,const Real* r,const int* index,
    int count,Real mass,Real multiplier,Real* Y) {
        Real z[Ns]{};
        int order[Ns]{};
        for(int j=0;j<count;++j){
            order[j]=j;
            z[j]=bulk[index[j]]-multiplier*r[j];
        }
        for(int j=1;j<count;++j){
            int k=j,v=order[j];
            while(k&&z[order[k-1]]<z[v]){
                order[k]=order[k-1];
                --k;
            }
            order[k]=v;
        }
        Real maximum=z[order[0]],sum=0,theta=-mass;
        int active=1;
        for(int j=1;j<count;++j){
            Real delta=z[order[j]]-maximum;
            if(delta<=theta)break;
            sum+=delta;
            active=j+1;
            theta=(sum-mass)/active;
        }
        for(int j=0;j<count;++j)Y[index[j]]=maxValue(Real(0),(z[j]-maximum)-theta);
    }

    CHMT_HD inline Real projectedMoment(const Real* bulk,const Real* r,const int* index,
    int count,Real mass,Real multiplier,Real* Y) {
        projectMass(bulk,r,index,count,mass,multiplier,Y);
        Real value=0;
        for(int j=0;j<count;++j)value+=r[j]*Y[index[j]];
        return value;
    }
    // KKT gives y=max(b-lambda-mu*r,0).  The projected moment is continuous and
    // nonincreasing in mu; bracket and bisect this monotone scalar equation.

    CHMT_HD inline void simplexCandidate(const Real* raw,Real* Y) {
        Real centered[Ns]{},zero[Ns]{};int index[Ns]{};Real maximum=raw[0];
        for(int i=1;i<Ns;++i)maximum=maxValue(maximum,raw[i]);
        for(int i=0;i<Ns;++i){Real delta=0;
            if(!sumFinite(raw[i],-maximum,delta))delta=-1;
            centered[i]=maxValue(Real(-1),delta);index[i]=i;
        }
        projectMass(centered,zero,index,Ns,1,0,Y);
    }

    CHMT_HD inline bool projectMoment(const Real* bulk,const Real* gasR,const int* index,
    int count,Real mass,Real moment,Real* Y) {
        if(!count||mass<0||mass>1)return false;
        if(mass==0){
            for(int j=0;j<count;++j)Y[index[j]]=0;
            return moment==0;
        }
        Real rmin=gasR[index[0]],rmax=rmin;
        for(int j=1;j<count;++j){
            rmin=minValue(rmin,gasR[index[j]]);
            rmax=maxValue(rmax,gasR[index[j]]);
        }
        if(rmin!=rmax && (moment<rmin*mass||moment>rmax*mass))return false;
        if(rmin==rmax){
            Real zero[Ns]{};
            projectMass(bulk,zero,index,count,mass,0,Y);
            return true;
        }
        // Exact endpoint means permit mass only in its corresponding equal-R group.
        if(moment==rmin*mass || moment==rmax*mass){
            Real endpoint=moment==rmin*mass?rmin:rmax;
            int group[Ns]{},n=0;
            Real zero[Ns]{};
            for(int j=0;j<count;++j){
                Y[index[j]]=0;
                if(gasR[index[j]]==endpoint)group[n++]=index[j];
            }
            projectMass(bulk,zero,group,n,mass,0,Y);
            return true;
        }
        Real r[Ns]{};
        for(int j=0;j<count;++j)r[j]=(gasR[index[j]]-rmin)/(rmax-rmin);
        Real target=(moment-rmin*mass)/(rmax-rmin),g=projectedMoment(bulk,r,index,count,mass,0,Y);
        if(g==target)return true;
        Real lo=0,hi=0,step=g>target?1:-1;
        for(int iteration=0;iteration<1026;++iteration){
            Real value=projectedMoment(bulk,r,index,count,mass,step,Y);
            if((g>target&&value<=target)||(g<target&&value>=target)){
                lo=minValue(Real(0),step);
                hi=maxValue(Real(0),step);
                break;
            }
            if(absValue(step)>=Real(4.4942328371557893e307))return false;
            step*=2;
        }
        if(lo==hi)return false;
        // Full binary exponent range plus mantissa bits bounds adjacent-double work.
        for(int iteration=0;iteration<2200;++iteration){
            Real mid=lo+(hi-lo)/2;
            if(mid==lo||mid==hi)break;
            g=projectedMoment(bulk,r,index,count,mass,mid,Y);
            if(g==target)return true;
            if(g>target)lo=mid;
            else hi=mid;
        }
        projectedMoment(bulk,r,index,count,mass,lo+(hi-lo)/2,Y);
        return true;
    }

    CHMT_HD inline bool boundLinear(Real coefficient,Real rhs,Real& lo,Real& hi) {
        // coefficient*r <= rhs
        Real ratio=0;
        if(coefficient!=0&&!quotientFinite(rhs,coefficient,ratio)){
            bool positive=(rhs>0)==(coefficient>0);
            if(coefficient>0)return positive;
            return !positive;
        }
        if(coefficient>0)hi=minValue(hi,ratio);
        else if(coefficient<0)lo=maxValue(lo,ratio);
        else if(rhs<0)return false;
        return lo<=hi;
    }

    CHMT_HD inline bool poleProfile(Real rm,const int* index,int count,const Real* gasR,
    const Real* bulk,const Real* outsideA,const Real* outsideB,
    Real A,Real B,Real AR,Real BR,Real* Y,Real& objective) {
        Real product=0,mass=0,moment=0,coefficient=0;
        if(!productFinite(A,rm,product)||!sumFinite(1,-B,mass)||!sumFinite(mass,-product,mass)
        ||!sumFinite(1,-AR,coefficient)||!productFinite(coefficient,rm,moment)||!sumFinite(moment,-BR,moment))return false;
        if(!projectMoment(bulk,gasR,index,count,mass,moment,Y))return false;
        objective=0;
        bool inside[Ns]{};
        for(int j=0;j<count;++j)inside[index[j]]=true;
        for(int i=0;i<Ns;++i){
            if(!inside[i]){
                Real outside=0;
                if(!productFinite(outsideA[i],rm,outside)||!sumFinite(outside,outsideB[i],Y[i]))return false;
            }
            if(!finite(Y[i])||Y[i]<0||Y[i]>1)return false;
            Real d=Y[i]-bulk[i];
            objective+=d*d;
        }
        return finite(objective);
    }

    CHMT_HD inline bool recoverPole(Real d0,const Real* d,const Real* u,const Real* v,
    const Real* r,const Real* bulk,Real rmin,Real rmax,Real* Y) {
        int index[Ns]{},count=0;
        Real A=0,B=0,AR=0,BR=0,a[Ns]{},b[Ns]{};
        Real lo=rmin,hi=rmax,fixedR=0;
        bool fixed=false;
        for(int i=0;i<Ns;++i){
            if(d[i]==d0){
                index[count++]=i;
                if(u[i]!=0){
                    if(!fixed){
                        if(!quotientFinite(-v[i],u[i],fixedR))return false;
                        fixed=true;
                    }
                }
                else if(v[i]!=0)return false;
            }
            else{
                if(!quotientFinite(u[i],d[i]-d0,a[i])||!quotientFinite(v[i],d[i]-d0,b[i]))return false;
                Real ar=0,br=0;
                if(!productFinite(r[i],a[i],ar)||!productFinite(r[i],b[i],br)
                ||!sumFinite(A,a[i],A)||!sumFinite(B,b[i],B)||!sumFinite(AR,ar,AR)||!sumFinite(BR,br,BR))return false;
                if(!boundLinear(-a[i],b[i],lo,hi)||!boundLinear(a[i],1-b[i],lo,hi))return false;
            }
        }
        if(!count)return false;
        Real gmin=r[index[0]],gmax=gmin;
        for(int j=1;j<count;++j){
            gmin=minValue(gmin,r[index[j]]);
            gmax=maxValue(gmax,r[index[j]]);
        }
        // A common positive scale preserves each affine constraint, including its
        // solution ratio. In a pole eigenspace individual coefficients can approach
        // MAX even though the feasible R interval is small and representable.
        Real coefficientScale=maxValue(Real(1),maxValue(absValue(A),maxValue(absValue(B),maxValue(absValue(AR),absValue(BR)))));
        Real scaledA=A/coefficientScale,scaledB=B/coefficientScale;
        Real scaledAR=AR/coefficientScale,scaledBR=BR/coefficientScale,one=1/coefficientScale;
        Real massBound=one-scaledB;
        Real minCoefficient=(one-scaledAR)+gmin*scaledA,minRhs=scaledBR+gmin*massBound;
        Real maxCoefficient=(one-scaledAR)+gmax*scaledA,maxRhs=scaledBR+gmax*massBound;
        if(!fixed && gmin==gmax){
            Real coefficient=minCoefficient,rhs=minRhs;
            if(coefficient!=0){
                if(!quotientFinite(rhs,coefficient,fixedR))return false;
                fixed=true;
            }
        }
        if(fixed){
            Real objective=0;
            return poleProfile(fixedR,index,count,r,bulk,a,b,A,B,AR,BR,Y,objective);
        }
        if(!boundLinear(scaledA,massBound,lo,hi)
        || !boundLinear(-minCoefficient,-minRhs,lo,hi)
        || !boundLinear(maxCoefficient,maxRhs,lo,hi))return false;
        // With no nonzero group flux R remains free.  The feasible interval above is
        // convex, and squared distance to its affine simplex slice is convex in R.
        // Golden-section minimization selects the same nearest-bulk policy as simple
        // roots; it changes no physical time or imposed flux.
        Real left=lo,right=hi,x1=right-(right-left)*Real(.6180339887498948482);
        Real x2=left+(right-left)*Real(.6180339887498948482),f1=0,f2=0,y1[Ns]{},y2[Ns]{};
        bool v1=poleProfile(x1,index,count,r,bulk,a,b,A,B,AR,BR,y1,f1);
        bool v2=poleProfile(x2,index,count,r,bulk,a,b,A,B,AR,BR,y2,f2);
        if(!v1||!v2)return false;
        for(int iteration=0;iteration<160;++iteration){
            if(x1==x2||x1==left||x2==right)break;
            if(f1<=f2){
                right=x2;
                x2=x1;
                f2=f1;
                for(int i=0;i<Ns;++i)y2[i]=y1[i];
                x1=right-(right-left)*Real(.6180339887498948482);
                if(!poleProfile(x1,index,count,r,bulk,a,b,A,B,AR,BR,y1,f1))return false;
            }
            else{
                left=x1;
                x1=x2;
                f1=f2;
                for(int i=0;i<Ns;++i)y1[i]=y2[i];
                x2=left+(right-left)*Real(.6180339887498948482);
                if(!poleProfile(x2,index,count,r,bulk,a,b,A,B,AR,BR,y2,f2))return false;
            }
        }
        for(int i=0;i<Ns;++i)Y[i]=f1<=f2?y1[i]:y2[i];
        return true;
    }

    CHMT_HD inline bool publishSelection(const Selection& best,const PhysicsConfig& p,Real tolerance,
        Real* out,Info& info,Info* suppliedInfo) {
        if(suppliedInfo)*suppliedInfo=info;
        if(!best.certified)return false;
        // Selection owns the exact state and original-equation residual certified
        // together in this invocation. Publishing changes neither state nor inputs;
        // private certification suppresses only failures, with no success side effect.
        const Real finalResidual=best.residual;
        info.state=finalResidual<=tolerance?SolveState::Converged:SolveState::AdmissibleApproximation;
        info.actualResidual=finalResidual;
        if(suppliedInfo)*suppliedInfo=info;
        for(int i=0;i<Ns;++i)out[i]=best.value[i];
        if(info.state!=SolveState::Converged || info.ambiguous)reportPhysicsWarning(p,
        ugkp::diagnostics::Code::AccuracyResidual,3802,finalResidual,tolerance);
        return true;
    }

    CHMT_HD inline void boundaryCandidates(const Real* d,const Real* u,const Real* v,
        const Real* r,const Real* bulk,const Real* Js,Real pressure,Real T,Real distance,
        const PhysicsConfig& trial,Real tolerance,Real a,Real b,Real rmin,Real rmax,Selection& best,
        const CompositionResidualInputs* prepared) {
        // Pole eigenvalues are genuine possibilities (zero/repeated diffusion), not
        // removable singularities. Their normalized eigenspace is a convex slice.
        for(int i=0;i<Ns;++i){
            bool duplicate=false;
            for(int j=0;j<i;++j)duplicate=duplicate||d[j]==d[i];
            if(duplicate||-d[i]<a||-d[i]>b)continue;
            Real Y[Ns]{};
            if(recoverPole(d[i],d,u,v,r,bulk,rmin,rmax,Y))
            certify(Y,bulk,Js,pressure,T,distance,trial,tolerance,best,nullptr,false,prepared);
        }
        // The closed admissible interval can contain a pure-species boundary root.
        for(int endpoint=0;endpoint<2;++endpoint){
            Real Y[Ns]{};
            if(recoverRoot(endpoint?b:a,d,u,v,r,Y)){
                certify(Y,bulk,Js,pressure,T,distance,trial,tolerance,best,nullptr,false,prepared);
                Real projected[Ns]{};simplexCandidate(Y,projected);
                certify(projected,bulk,Js,pressure,T,distance,trial,tolerance,best,nullptr,false,prepared);
            }
        }
    }

    // In scalarSolve's normalized domain all off-diagonal arithmetic is
    // representable. Its strict margin is minimized at one excluding-self
    // r extremum: minimum for u>=0, maximum for u<0.
    CHMT_HD inline bool uniquePositiveMatrix(const Real* u,const Real* v,
        const Real* r,const Real* bulk,Real gamma) {
        Real lo=HUGE_VAL,secondLo=HUGE_VAL,hi=-HUGE_VAL,secondHi=-HUGE_VAL;
        int lowIndex=-1,highIndex=-1;
        for(int i=0;i<Ns;++i){
            if(!(bulk[i]>0))return false;
            if(r[i]<lo){secondLo=lo;lo=r[i];lowIndex=i;}else secondLo=minValue(secondLo,r[i]);
            if(r[i]>hi){secondHi=hi;hi=r[i];highIndex=i;}else secondHi=maxValue(secondHi,r[i]);
        }
        for(int i=0;i<Ns&&Ns>1;++i){
            const Real extreme=u[i]>=0?(i==lowIndex?secondLo:lo):(i==highIndex?secondHi:hi);
            Real product=0,value=0,magnitude=0;
            if(!productFinite(u[i],extreme,product)||!sumFinite(product,v[i],value)
                ||!sumFinite(absValue(product),absValue(v[i]),magnitude)
                ||value<=gamma*magnitude)return false;
        }
        return true;
    }

    CHMT_HD inline bool twoByTwo(Real a,Real b,Real e,Real c,Real d,Real f,Real& x,Real& y) {
        // Pivoted two-row elimination. Safe operations also cover remote scalar
        // iterates; a singular representation simply needs root isolation.
        if(absValue(c)>absValue(a)){Real t=a;a=c;c=t;t=b;b=d;d=t;t=e;e=f;f=t;}
        Real factor=0,term=0;
        if(!quotientFinite(c,a,factor)||!productFinite(factor,b,term)||!sumFinite(d,-term,d)
            ||!productFinite(factor,e,term)||!sumFinite(f,-term,f)||!quotientFinite(f,d,y)
            ||!productFinite(b,y,term)||!sumFinite(e,-term,e)||!quotientFinite(e,a,x))return false;
        return true;
    }

    // For a 2x2 irreducible Metzler matrix, the positive Perron vector
    // follows directly from its quadratic eigenvalue. Choose the sign branch
    // with an addition rather than a subtractive cancellation. sqrt(b)*sqrt(c)
    // avoids underflow of b*c. The original physical equations still certify it.
    CHMT_HD inline bool uniqueTwoSpecies(const Real* d,const Real* u,const Real* v,
        const Real* r,const Real* bulk,const Real* Js,Real pressure,Real T,
        Real distance,const PhysicsConfig& p,Real tolerance,Selection& best,
        const CompositionResidualInputs* prepared) {
        const Real a=u[0]*r[0]+v[0]-d[0],b=u[0]*r[1]+v[0];
        const Real c=u[1]*r[0]+v[1],e=u[1]*r[1]+v[1]-d[1];
        const Real half=(a-e)/2,root=::hypot(half,::sqrt(b)*::sqrt(c));
        Real Y[Ns]{},one=0,two=0;
        if(half>=0){one=root+half;two=c;}else{one=b;two=root-half;}
        const Real sum=one+two;
        if(!quotientFinite(one,sum,Y[0])||!quotientFinite(two,sum,Y[1]))return false;
        bool solved=false;certify(Y,bulk,Js,pressure,T,distance,p,tolerance,best,&solved,false,prepared);
        return solved;
    }

    // Newton for (z+d_i)Y_i-u_i(r.Y)-v_i=0, sum(Y)=1.
    // Its bordered diagonal/rank-one linearization eliminates to two scalars,
    // delta(r.Y) and delta(z), with O(Ns) work. The mixture density is variable.
    // Only a proved unique case may return early; all other roots retain the
    // established isolation/nearest-bulk selection policy below.
    CHMT_HD inline bool uniqueNewton(const Real* d,const Real* u,const Real* v,
        const Real* r,Real J,Real Db,const Real* bulk,const Real* Js,Real pressure,
        Real T,Real distance,const PhysicsConfig& p,Real tolerance,Selection& best,Info& info,
        const CompositionResidualInputs* prepared) {
        Real Y[Ns];for(int i=0;i<Ns;++i)Y[i]=bulk[i];
        for(int iteration=0;iteration<8;++iteration){
            Real rm=0,dy=0;int dependent=0;
            for(int i=0;i<Ns;++i){rm+=r[i]*Y[i];dy+=d[i]*Y[i];if(Y[i]>Y[dependent])dependent=i;}
            const Real z=J*rm+Db-dy;
            Real U[Ns],V[Ns],H[Ns],A=0,B=0,C=-1,D=0,E=0,F=0;
            for(int i=0;i<Ns;++i){
                const Real denominator=z+d[i];Real term=0,residual=0;
                if(!productFinite(denominator,Y[i],residual)||!productFinite(u[i],rm,term)
                    ||!sumFinite(residual,-term,residual)||!sumFinite(residual,-v[i],residual)
                    ||!quotientFinite(u[i],denominator,U[i])||!quotientFinite(Y[i],denominator,V[i])
                    ||!quotientFinite(residual,denominator,H[i])
                    ||!sumFinite(A,U[i],A)||!sumFinite(B,-V[i],B)||!sumFinite(E,H[i],E)
                    ||!productFinite(r[i],U[i],term)||!sumFinite(C,term,C)
                    ||!productFinite(r[i],-V[i],term)||!sumFinite(D,term,D)
                    ||!productFinite(r[i],H[i],term)||!sumFinite(F,term,F))return false;
            }
            Real deltaR=0,deltaZ=0;
            if(!twoByTwo(A,B,E,C,D,F,deltaR,deltaZ))return false;
            Real next[Ns],sum=0;bool changed=false;
            for(int i=0;i<Ns;++i)if(i!=dependent){
                Real first=0,second=0,delta=0;
                if(!productFinite(U[i],deltaR,first)||!productFinite(V[i],deltaZ,second)
                    ||!sumFinite(first,-second,delta)||!sumFinite(delta,-H[i],delta)
                    ||!sumFinite(Y[i],delta,next[i])||next[i]<0||next[i]>1)return false;
                sum+=next[i];changed=changed||next[i]!=Y[i];
            }
            next[dependent]=1-sum;
            if(next[dependent]<0||next[dependent]>1)return false;
            changed=changed||next[dependent]!=Y[dependent];
            ++info.structuredIterations;
            bool solved=false;
            certify(next,bulk,Js,pressure,T,distance,p,tolerance,best,&solved,false,prepared);
            if(solved)return true;
            if(!changed)return false;
            for(int i=0;i<Ns;++i)Y[i]=next[i];
        }
        return false;
    }

    CHMT_HD inline bool scalarSolve(const Real* Js,const Real* bulk,Real pressure,
    Real T,Real distance,const PhysicsConfig& p,Real tolerance,Real* out,
    Info* suppliedInfo=nullptr,const CompositionResidualInputs* suppliedInputs=nullptr) {
        Info info;if(suppliedInfo)*suppliedInfo=info;
        CompositionResidualInputs localInputs;
        if(!suppliedInputs&&!prepareCompositionResidual<false>(bulk,Js,p,localInputs))return false;
        const auto* prepared=suppliedInputs?suppliedInputs:&localInputs;
        Selection best;
        const PhysicsConfig& trial=p;
        Real initialY[Ns]{};
        for(int i=0;i<Ns;++i)initialY[i]=bulk[i];
        if(certify(initialY,bulk,Js,pressure,T,distance,trial,tolerance,best,nullptr,true,prepared)&&best.converged){
            return publishSelection(best,p,tolerance,out,info,suppliedInfo);
        }
        best={};
        Real d[Ns]{},u[Ns]{},v[Ns]{},r[Ns]{},J=0,Db=0;
        Real Rref=0,scale=0,rmin=1,rmax=0,dmin=1.7976931348623157e308,dmax=0;
        for(int i=0;i<Ns;++i)Rref=maxValue(Rref,prepared->gasR[i]);
        DiffusionState referenceDiffusion(pressure,T,distance,Rref);
        for(int i=0;i<Ns;++i){
            r[i]=prepared->gasR[i]/Rref;
            if(!referenceDiffusion.scale(gasMolecularDiffusivity(p,i,T,pressure),d[i]))return false;
            scale=maxValue(scale,maxValue(absValue(Js[i]),d[i]));
        }
        if(!(scale>0)||!finite(scale))return false;
        for(int i=0;i<Ns;++i){
            d[i]/=scale;
            u[i]=Js[i]/scale;
            v[i]=d[i]*bulk[i];
            J+=u[i];
            Db+=v[i];
            rmin=minValue(rmin,r[i]);
            rmax=maxValue(rmax,r[i]);
            dmin=minValue(dmin,d[i]);
            dmax=maxValue(dmax,d[i]);
        }
        Real a=minValue(J*rmin,J*rmax)+Db-dmax;
        Real b=maxValue(J*rmin,J*rmax)+Db-dmin;
        // Forward-error bounds for the signed sum J, positive sum Db and
        // endpoint expressions. gamma_n is an operation-count bound, not a
        // case tolerance. Bounds describe the represented matrix coefficients.
        const Real epsilon=2.22044604925031308085e-16;
        Real absU=0,absV=0;for(int i=0;i<Ns;++i){absU+=absValue(u[i]);absV+=absValue(v[i]);}
        const Real gamma=(2*Ns+8)*epsilon/(1-(2*Ns+8)*epsilon);
        const Real boundError=gamma*(absU*rmax+absV+dmax);
        a=::nextafter(a-boundError,-HUGE_VAL);b=::nextafter(b+boundError,HUGE_VAL);
        Real intervalScale=maxValue(maxValue(absValue(a),absValue(b)),dmax);
        if(!(intervalScale>0)||!finite(intervalScale))return false;
        a/=intervalScale;
        b/=intervalScale;
        for(int i=0;i<Ns;++i)if(!quotientFinite(d[i],intervalScale,d[i])
        ||!quotientFinite(u[i],intervalScale,u[i])||!quotientFinite(v[i],intervalScale,v[i]))return false;
        // Strictly positive off-diagonals make M irreducible Metzler. Its only
        // nonnegative normalized eigenvector is the Perron vector. Collatz bounds
        // from the positive bulk vector narrow this SAME scalar isolation problem;
        // they are not an alternate implementation or a physical empirical switch.
        Real sumBulk=0,rBulk=0;
        const Real matrixGamma=3*epsilon/(1-3*epsilon);
        const bool positive=uniquePositiveMatrix(u,v,r,bulk,matrixGamma);info.unique=positive;
        for(int i=0;i<Ns;++i){sumBulk+=bulk[i];rBulk+=r[i]*bulk[i];}
        Real scaledJ=0,scaledDb=0;for(int i=0;i<Ns;++i){scaledJ+=u[i];scaledDb+=v[i];}
        if(positive){
            const bool solved=Ns==2
                ?uniqueTwoSpecies(d,u,v,r,bulk,Js,pressure,T,distance,p,tolerance,best,prepared)
                :uniqueNewton(d,u,v,r,scaledJ,scaledDb,bulk,Js,pressure,T,distance,p,tolerance,best,info,prepared);
            if(solved)return publishSelection(best,p,tolerance,out,info,suppliedInfo);
        }
        if(positive){Real lower=1.7976931348623157e308,upper=-lower;bool bounded=true;
            const Real rowGamma=(4*Ns+16)*epsilon/(1-(4*Ns+16)*epsilon);
            for(int i=0;i<Ns;++i){Real one=0,two=0,three=0,numerator=0,magnitude=0,ratio=0,error=0;
                // Use v_i as represented; do not silently assume v_i=d_i*b_i
                // has no multiplication rounding error.
                if(!productFinite(u[i],rBulk,one)||!productFinite(v[i],sumBulk,two)
                    ||!productFinite(d[i],bulk[i],three)||!sumFinite(one,two,numerator)
                    ||!sumFinite(numerator,-three,numerator)||!quotientFinite(numerator,bulk[i],ratio)
                    ||!sumFinite(absValue(one),absValue(two),magnitude)||!sumFinite(magnitude,absValue(three),magnitude)
                    ||!productFinite(rowGamma,magnitude,error)||!quotientFinite(error,bulk[i],error)){
                    bounded=false;break;
                }
                Real lowEdge=0,highEdge=0;
                if(!sumFinite(ratio,-error,lowEdge)||!sumFinite(ratio,error,highEdge)){bounded=false;break;}
                lower=minValue(lower,::nextafter(lowEdge,-HUGE_VAL));
                upper=maxValue(upper,::nextafter(highEdge,HUGE_VAL));
            }
            if(bounded){Real narrowedA=maxValue(a,lower),narrowedB=minValue(b,upper);
                if(narrowedA<=narrowedB){a=narrowedA;b=narrowedB;}
            }
        }
        Real cuts[Ns+2]{};
        int cutCount=2;
        cuts[0]=a;
        cuts[1]=b;
        for(int i=0;i<Ns;++i){
            Real pole=-d[i];
            if(pole>a&&pole<b)cuts[cutCount++]=pole;
        }
        for(int i=1;i<cutCount;++i){
            Real item=cuts[i];
            int j=i;
            while(j&&cuts[j-1]>item){
                cuts[j]=cuts[j-1];
                --j;
            }
            cuts[j]=item;
        }
        int uniqueCount=0;
        for(int i=0;i<cutCount;++i)if(!uniqueCount||cuts[i]!=cuts[uniqueCount-1])cuts[uniqueCount++]=cuts[i];
        cutCount=uniqueCount;
        // Preserve pole/endpoint ordering when multiple roots can compete.
        if(!positive)boundaryCandidates(d,u,v,r,bulk,Js,pressure,T,distance,trial,tolerance,a,b,rmin,rmax,best,prepared);
        for(int segment=1;segment<cutCount;++segment){
            Real left=cuts[segment-1],right=cuts[segment];
            Poly poly=polynomial(left,right,d,u,v,r);
            ++info.polynomialCount;
            Real isolated[ScalarCandidateCapacity]{};bool genuine[ScalarCandidateCapacity]{};
            int rootCount=roots(poly,info,isolated,genuine);
            for(int root=0;root<rootCount;++root){
                ++info.candidates;
                Real z=left+(right-left)*isolated[root],Y[Ns]{};
                if(!recoverRoot(z,d,u,v,r,Y))continue;
                bool solved=false;
                bool rawPhysical=certify(Y,bulk,Js,pressure,T,distance,trial,tolerance,best,&solved,genuine[root],prepared);
                if(positive&&best.converged)return publishSelection(best,p,tolerance,out,info,suppliedInfo);
                bool projectedSolved=false;
                if(!solved){
                Real projected[Ns]{};simplexCandidate(Y,projected);Selection projectedResult;
                certify(projected,bulk,Js,pressure,T,distance,trial,tolerance,projectedResult,&projectedSolved,true,prepared);
                // Projection is only a mathematical candidate. A remote
                // nonphysical eigenvector may not become a published finite
                // approximation merely by projecting it into the simplex.
                if(projectedSolved)certify(projected,bulk,Js,pressure,T,distance,trial,tolerance,best,nullptr,false,prepared);
                if(!solved&&!projectedSolved){
                    refine(Y,bulk,Js,pressure,T,distance,trial,info,prepared);
                    certify(Y,bulk,Js,pressure,T,distance,trial,tolerance,best,&solved,genuine[root]&&rawPhysical,prepared);
                    simplexCandidate(Y,projected);projectedResult={};
                    certify(projected,bulk,Js,pressure,T,distance,trial,tolerance,projectedResult,&projectedSolved,true,prepared);
                    if(projectedSolved)certify(projected,bulk,Js,pressure,T,distance,trial,tolerance,best,nullptr,false,prepared);
                }
                }
                if(solved||projectedSolved)++info.legal;
                if(positive&&best.converged)return publishSelection(best,p,tolerance,out,info,suppliedInfo);
            }
        }
        if(positive)boundaryCandidates(d,u,v,r,bulk,Js,pressure,T,distance,trial,tolerance,a,b,rmin,rmax,best,prepared);
        return publishSelection(best,p,tolerance,out,info,suppliedInfo);
    }

    CHMT_HD inline bool linearComposition(const Real* Js,const Real* bulk,Real J,
    Real pressure,Real T,Real distance,const PhysicsConfig& p,Real tolerance,Real* out,
    Info* suppliedInfo=nullptr,const CompositionResidualInputs* suppliedInputs=nullptr) {
        if(suppliedInfo)*suppliedInfo=Info{};
        Real centered[Ns]{},Y[Ns]{},zero[Ns]{};
        int index[Ns]{};
        if(J!=0){
            int peak=0;
            for(int i=1;i<Ns;++i)if(J>0?Js[i]>Js[peak]:Js[i]<Js[peak])peak=i;
            for(int i=0;i<Ns;++i){
                Real difference=0,delta=0;
                if(!sumFinite(Js[i],-Js[peak],difference)||!quotientFinite(difference,J,delta))delta=-1;
                // A level at least one below the highest cannot be active in a mass-one
                // simplex projection. This is an exact bound, not a physical limiter.
                centered[i]=maxValue(Real(-1),delta);
            }
        }
        else for(int i=0;i<Ns;++i)centered[i]=bulk[i];
        for(int i=0;i<Ns;++i)index[i]=i;
        projectMass(centered,zero,index,Ns,1,0,Y);
        const PhysicsConfig& trial=p;
        CompositionResidualInputs localInputs;
        if(!suppliedInputs&&!prepareCompositionResidual<false>(bulk,Js,p,localInputs))return false;
        const auto* prepared=suppliedInputs?suppliedInputs:&localInputs;
        Selection selected;
        if(!certify(Y,bulk,Js,pressure,T,distance,trial,tolerance,selected,nullptr,true,prepared))return false;
        const Real finalResidual=selected.residual;
        if(suppliedInfo){
            suppliedInfo->state=finalResidual<=tolerance?SolveState::Converged:SolveState::AdmissibleApproximation;
            suppliedInfo->actualResidual=finalResidual;
        }
        for(int i=0;i<Ns;++i)out[i]=selected.value[i];
        if(finalResidual>tolerance)reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,3807,finalResidual,tolerance);
        return true;
    }
}
#endif
