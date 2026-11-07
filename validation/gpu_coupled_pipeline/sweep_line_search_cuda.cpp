#define SWEEP_RESIDUAL_REFERENCE_ONLY
#include "sweep_residual_cuda.cpp"
#include "sweep_line_search_probe.H"
static std::vector<SweepSearchProbe> nativeSearch(const g::SweepResidualDefinition& d,const std::vector<g::Point>& origin,const std::vector<g::Point>& correction,double initial,int limit){
    std::vector<SweepSearchProbe> out;int next=0;
    for(;;){SweepSearchProbe batch;
        for(;next<std::max(1,limit);){const double alpha=std::ldexp(1.,-next);if(alpha==0)break;++next;
            auto trial=origin;bool representable=true,changed=false;
            for(std::size_t p=0;p<trial.size();++p){double* values[]={&trial[p].x,&trial[p].y,&trial[p].z};const double start[]={origin[p].x,origin[p].y,origin[p].z},delta[]={correction[p].x,correction[p].y,correction[p].z};
                for(int k=0;k<3;++k){const long double value=static_cast<long double>(start[k])+static_cast<long double>(alpha)*delta[k];
                    if(!std::isfinite(value)||std::abs(value)>std::numeric_limits<double>::max()){representable=false;break;}
                    *values[k]=start[k]+alpha*delta[k];changed=changed||*values[k]!=start[k];}
                if(!representable)break;}
            if(!representable)continue;if(!changed){batch.status.exhausted=true;break;}
            ++batch.status.trials;batch.status.evaluated=true;batch.status.trial.residual=reference(d,trial).status;
            batch.status.trial.coordinate={alpha,true,true,-1};const auto result=batch.status.trial.residual;
            if(result.failure==g::SweepResidualFailure::None&&(result.metric<=1||result.metric<initial*(1-1e-4*alpha))){batch.status.candidate=true;batch.status.trial.sufficientDecrease=true;batch.points=std::move(trial);break;}
        }
        batch.status.nextBacktrack=next;batch.status.exhausted=batch.status.exhausted||next>=std::max(1,limit)||std::ldexp(1.,-next)==0;
        out.push_back(batch);if(!batch.status.candidate)break;
    }return out;
}
int main(){int cases=0;
    for(int faces:{1,129})for(double scale:{1e-3,1.,1e3})for(int mode=0;mode<5;++mode)for(int limit:{1,8}){
        g::SweepResidualDefinition d;d.topology.faceOffsets={0};d.absoluteTolerance=1e-10*scale*scale*scale;d.rankTolerance=1e-12;
        for(int f=0;f<faces;++f){int p=d.old.size();d.old.insert(d.old.end(),{{scale*2*f,0,0},{scale*(2*f+1),0,0},{scale*(2*f+1),scale,0},{scale*2*f,scale,0}});
            for(int k=0;k<4;++k)d.topology.facePoints.push_back(p+k);d.topology.faceOffsets.push_back(d.topology.facePoints.size());d.faces.push_back(f);d.targets.push_back(.01*scale*scale*scale);d.scales.push_back(scale*scale*scale);}
        auto origin=d.old;std::vector<g::Point> correction(d.old.size(),{0,0,scale*(mode==1?.2:mode==2?-.03:mode==3?0:.01)});
        if(mode==4)correction.back().z=std::numeric_limits<double>::quiet_NaN();
        const double metric=reference(d,origin).status.metric;const auto expected=nativeSearch(d,origin,correction,metric,limit),actual=probeSweepSearch(d,origin,correction,metric,limit);
        require(expected.size()==actual.size(),"candidate sequence length differs",cases);
        for(std::size_t batch=0;batch<expected.size();++batch){const auto& a=expected[batch];const auto& b=actual[batch];
            require(a.status.nextBacktrack==b.status.nextBacktrack&&a.status.trials==b.status.trials&&a.status.evaluated==b.status.evaluated&&a.status.exhausted==b.status.exhausted&&a.status.candidate==b.status.candidate,"line-search control differs",cases);
            if(a.status.evaluated){require(a.status.trial.residual.failure==b.status.trial.residual.failure,"trial residual failure differs",cases);if(a.status.trial.residual.failure==g::SweepResidualFailure::None)ratio(a.status.trial.residual.metric,b.status.trial.residual.metric,cases);}
            if(a.status.candidate){require(a.points.size()==b.points.size(),"candidate shape differs",cases);for(std::size_t p=0;p<a.points.size();++p){exact(a.points[p].x,b.points[p].x,cases);exact(a.points[p].y,b.points[p].y,cases);exact(a.points[p].z,b.points[p].z,cases);}}}
        ++cases;
    }
    std::cout<<"PASS "<<cases<<" frozen native line-search sequences, forced certification rejection/resume, range failures, zero updates, cross-block geometry and composed replay\n";
}
