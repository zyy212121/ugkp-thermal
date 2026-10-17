#pragma once
#include "WallModelTypes.H"
#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace ugkwp { namespace gaswall {
// Host-only, application-neutral mesh. Face winding is outward from owner.
// All physical walls must be marked, including walls not using this model.
// A version identifies immutable geometry AND wall/constraint classifications.
template<class R> struct WallGeometryMesh {
    std::vector<std::array<R,3>> points,cellCentres;
    std::vector<R> volumes;
    std::vector<int> faceOffsets,facePoints,owner,neighbour;
    std::vector<unsigned char> physicalWall,omegaConstrained,gasCell;
    std::uint64_t geometryVersion=0;
};
template<class R> struct WallGeometryOptions {
    ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
    R relativeTolerance=R(1e-9);
    int maxRayCells=32,quadratureRefinement=1;
    bool constructQuadrature=true; // Optional direct diagnostics; production adapters disable.
    // When requested, zero uses a purely geometric rule. Direct SST research
    // supplies BVP nodes/stretch to integrate its source reconstruction.
    int profileNodes=0,firstSegmentGaussOrder=8;
    R profileStretch=R(2);
};
template<class R> struct WallGeometryDescriptor {
    int ownerCell=-1,wallFace=-1;
    // Positive containing-cell sampling is piecewise constant (order zero),
    // never advertised as an affine-exact reconstruction at matchingPoint.
    int samplingOrder=0;
    R matchingDonorDistance=0; // |donor cell centre - matchingPoint|.
    std::uint64_t geometryVersion=0;
    std::array<R,3> origin{},normal{},tangent1{},tangent2{},matchingPoint{},firstMomentVector{};
    R ownerDistance=0,matchingDistance=0,maximumOwnerDistance=0;
    R volume=0,firstMoment=0,geometryVolume=0;
    std::vector<R> distance,volumeWeight,matchingWeights;
    // Legacy tetra nodes are interior. Profile-aware nodes instead report the
    // full section's area centroid for moment diagnostics: this can lie outside
    // a concave/disconnected section. Core uses only its positive y measure.
    bool quadraturePointsAreInterior=true;
    std::vector<std::array<R,3>> quadraturePoints;
    std::vector<int> matchingCells;
    WallOwnerQuadrature<R> quadrature() const {
        WallOwnerQuadrature<R> q;
        q.distance=distance.data();q.volumeWeight=volumeWeight.data();
        q.count=static_cast<int>(distance.size());q.volume=volume;q.firstMoment=firstMoment;
        return q;
    }
};
namespace wallGeometryDetail {
template<class R> using Vec=std::array<R,3>;
template<class R> inline Vec<R> add(const Vec<R>&a,const Vec<R>&b){return {{a[0]+b[0],a[1]+b[1],a[2]+b[2]}};}
template<class R> inline Vec<R> sub(const Vec<R>&a,const Vec<R>&b){return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}};}
template<class R> inline Vec<R> mul(const Vec<R>&a,R b){return {{a[0]*b,a[1]*b,a[2]*b}};}
template<class R> inline R dot(const Vec<R>&a,const Vec<R>&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
template<class R> inline Vec<R> cross(const Vec<R>&a,const Vec<R>&b){return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};}
template<class R> inline R norm(const Vec<R>&a){return std::sqrt(dot(a,a));}
template<class R> inline bool finite(const Vec<R>&a){return gasFinite(a[0])&&gasFinite(a[1])&&gasFinite(a[2]);}
template<class R> struct Triangle {int a,b,c;};
template<class R> struct Face {std::vector<Triangle<R>> triangles;Vec<R> normal{},centre{};R scale=0;};
template<class R> struct Context {
    std::vector<Face<R>> faces;
    std::vector<std::vector<int>> cellFaces;
    std::vector<unsigned char> constrained;
    std::vector<R> firstGaussPoint,firstGaussWeight;
};
inline bool invalid(WallStatus& s,int face=-1){s.code=WallCode::InvalidGeometry;s.node=face;return false;}
template<class R> inline R turn(const Vec<R>& a,const Vec<R>& b,const Vec<R>&c,const Vec<R>&n){return dot(cross(sub(b,a),sub(c,a)),n);}
// Ear clipping gives positive, non-overlapping triangles for a simple planar
// polygon, including concave faces. No unsigned fan across a concavity.
template<class R> bool triangulate(const WallGeometryMesh<R>&m,int f,Face<R>&out,const WallGeometryOptions<R>&){
    std::vector<int> ring(m.facePoints.begin()+m.faceOffsets[f],m.facePoints.begin()+m.faceOffsets[f+1]);
    const Vec<R> anchor=m.points[ring[0]];Vec<R> area{};
    for(std::size_t i=0;i<ring.size();++i){
        const Vec<R> a=sub(m.points[ring[i]],anchor),b=sub(m.points[ring[(i+1)%ring.size()]],anchor);
        area=add(area,cross(a,b));out.scale=std::max(out.scale,norm(a));
    }
    const R amag=norm(area),eps=R(128)*std::numeric_limits<R>::epsilon();
    if(!(out.scale>0)||!(amag>eps*out.scale*out.scale))return false;
    out.normal=mul(area,R(1)/amag);
    const R atol=eps*out.scale*out.scale;
    // A repeated vertex, crossing edge or self-touching boundary is not a face.
    for(std::size_t i=0;i<ring.size();++i)for(std::size_t j=i+1;j<ring.size();++j){
        if(ring[i]==ring[j])return false;
        if(j==i+1||(i==0&&j+1==ring.size()))continue;
        const auto &a=m.points[ring[i]],&b=m.points[ring[(i+1)%ring.size()]],&c=m.points[ring[j]],&d=m.points[ring[(j+1)%ring.size()]];
        const R u=turn(a,b,c,out.normal),v=turn(a,b,d,out.normal),w=turn(c,d,a,out.normal),z=turn(c,d,b,out.normal);
        if(((u>atol&&v<-atol)||(u<-atol&&v>atol))&&((w>atol&&z<-atol)||(w<-atol&&z>atol)))return false;
    }
    // Collinear edge points carry no triangle area but remain in the mesh's

    bool changed=true;
    while(changed&&ring.size()>3){changed=false;for(std::size_t i=0;i<ring.size();++i){
        const auto&a=m.points[ring[(i+ring.size()-1)%ring.size()]],&b=m.points[ring[i]],&c=m.points[ring[(i+1)%ring.size()]];
        if(std::abs(turn(a,b,c,out.normal))<=atol&&dot(sub(b,a),sub(b,c))<=0){ring.erase(ring.begin()+i);changed=true;break;}
    }}
    while(ring.size()>3){
        bool cut=false;
        for(std::size_t i=0;i<ring.size();++i){
            const int ia=ring[(i+ring.size()-1)%ring.size()],ib=ring[i],ic=ring[(i+1)%ring.size()];
            const auto&a=m.points[ia],&b=m.points[ib],&c=m.points[ic];
            if(turn(a,b,c,out.normal)<=atol)continue;
            bool occupied=false;for(int ip:ring){if(ip==ia||ip==ib||ip==ic)continue;const auto&p=m.points[ip];
                if(turn(a,b,p,out.normal)>=-atol&&turn(b,c,p,out.normal)>=-atol&&turn(c,a,p,out.normal)>=-atol){occupied=true;break;}}
            if(occupied)continue;
            out.triangles.push_back({ia,ib,ic});ring.erase(ring.begin()+i);cut=true;break;
        }
        if(!cut)return false;
    }
    if(turn(m.points[ring[0]],m.points[ring[1]],m.points[ring[2]],out.normal)<=atol)return false;
    out.triangles.push_back({ring[0],ring[1],ring[2]});
    R sum=0;Vec<R> moment{};
    for(const auto&t:out.triangles){const auto a=sub(m.points[t.a],anchor),b=sub(m.points[t.b],anchor),c=sub(m.points[t.c],anchor);
        R w=norm(cross(sub(b,a),sub(c,a)));sum+=w;moment=add(moment,mul(add(add(a,b),c),w/R(3)));}
    if(!(sum>0))return false;
    out.centre=add(anchor,mul(moment,R(1)/sum));return finite(out.centre);
}




template<class R> bool gaussRule(int order,std::vector<R>&x,std::vector<R>&weight){
    if(order<=0)return false;
    x.resize(order);weight.resize(order);
    const R pi=std::acos(R(-1)),eps=R(8)*std::numeric_limits<R>::epsilon();
    for(int i=0;i<(order+1)/2;++i){
        R z=std::cos(pi*(R(i)+R(.75))/(R(order)+R(.5))),derivative=0;
        for(int iteration=0;iteration<32;++iteration){
            R p0=1,p1=z;
            for(int j=2;j<=order;++j){const R p=((R(2*j-1)*z*p1)-R(j-1)*p0)/R(j);p0=p1;p1=p;}
            derivative=R(order)*(z*p1-p0)/(z*z-R(1));
            const R step=p1/derivative;z-=step;
            if(std::abs(step)<=eps)break;
        }
        R p0=1,p1=z;
        for(int j=2;j<=order;++j){const R p=(R(2*j-1)*z*p1-R(j-1)*p0)/R(j);p0=p1;p1=p;}
        derivative=R(order)*(z*p1-p0)/(z*z-R(1));
        const R w=R(2)/((R(1)-z*z)*derivative*derivative);
        x[i]=-z;x[order-1-i]=z;weight[i]=weight[order-1-i]=w;
    }
    return true;
}
template<class R> bool prepare(const WallGeometryMesh<R>&m,Context<R>&ctx,WallStatus&s,const WallGeometryOptions<R>&opt){
    const std::size_t nc=m.cellCentres.size(),nf=m.owner.size();
    if(nc==0||nf==0||m.neighbour.size()<nf||m.physicalWall.size()<nf
        ||m.faceOffsets.size()<nf+1
        ||(!m.omegaConstrained.empty()&&m.omegaConstrained.size()<nc)||(!m.gasCell.empty()&&m.gasCell.size()<nc))return invalid(s);
    if(opt.constructQuadrature&&opt.profileNodes&&opt.profileNodes<2)return invalid(s);
    if(opt.constructQuadrature&&opt.profileNodes&& !gaussRule(opt.firstSegmentGaussOrder,ctx.firstGaussPoint,ctx.firstGaussWeight))return invalid(s);
    ctx.faces.resize(nf);ctx.cellFaces.resize(nc);ctx.constrained=m.omegaConstrained;
    if(ctx.constrained.empty())ctx.constrained.resize(nc,0);
    for(std::size_t f=0;f<nf;++f){
        if(m.owner[f]<0||m.owner[f]>=static_cast<int>(nc)||m.neighbour[f]<-1||m.neighbour[f]>=static_cast<int>(nc)||m.neighbour[f]==m.owner[f]
            ||m.faceOffsets[f]<0||m.faceOffsets[f+1]<m.faceOffsets[f]
            ||m.faceOffsets[f+1]>static_cast<int>(m.facePoints.size())
            ||m.faceOffsets[f+1]-m.faceOffsets[f]<3)return invalid(s,static_cast<int>(f));
        for(int k=m.faceOffsets[f];k<m.faceOffsets[f+1];++k)if(m.facePoints[k]<0||m.facePoints[k]>=static_cast<int>(m.points.size()))return invalid(s,static_cast<int>(f));
        if(!triangulate(m,static_cast<int>(f),ctx.faces[f],opt))return invalid(s,static_cast<int>(f));
        ctx.cellFaces[m.owner[f]].push_back(static_cast<int>(f));
        if(m.neighbour[f]>=0)ctx.cellFaces[m.neighbour[f]].push_back(static_cast<int>(f));
        // Even an omitted or erroneous explicit mask cannot make a wall owner
        // into a supposedly unconstrained matching donor.
        if(m.physicalWall[f])ctx.constrained[m.owner[f]]=1;
    }
    return true;
}
template<class R> bool cellSeed(const WallGeometryMesh<R>&m,const Context<R>&ctx,int cell,Vec<R>&seed,R&scale,ugkp::diagnostics::DeviceContext* diagnostics=nullptr){
    if(ctx.cellFaces[cell].size()<4)return false;
    scale=0;seed=m.cellCentres[cell];
    for(int f:ctx.cellFaces[cell])
        for(int k=m.faceOffsets[f];k<m.faceOffsets[f+1];++k)
            scale=std::max(scale,norm(sub(m.points[m.facePoints[k]],seed)));
    if(!(scale>0))return false;
    // One half-space projection solve with an a-priori local-scale margin.
    // Never restart from the cell centre with a relaxed margin.
    R separation=scale;
    for(int f:ctx.cellFaces[cell]){
        const R distance=std::abs(dot(sub(seed,ctx.faces[f].centre),ctx.faces[f].normal));
        if(distance>0)separation=std::min(separation,distance);
    }
    const R margin=std::max(R(.01)*separation,R(256)*std::numeric_limits<R>::epsilon()*scale);
    for(int pass=0;pass<256;++pass){
        bool good=true;
        for(int f:ctx.cellFaces[cell]){
            const auto n=mul(ctx.faces[f].normal,m.owner[f]==cell?R(1):R(-1));
            const R violation=dot(sub(seed,ctx.faces[f].centre),n)+margin;
            if(violation>0){seed=sub(seed,mul(n,violation+margin));good=false;}
        }
        if(good)return finite(seed);
    }
    R residual=0;
    for(int f:ctx.cellFaces[cell]){
        const auto n=mul(ctx.faces[f].normal,m.owner[f]==cell?R(1):R(-1));
        residual=std::max(residual,dot(sub(seed,ctx.faces[f].centre),n)+margin);
    }
    if(finite(seed))ugkp::diagnostics::recordWarning(diagnostics,
        ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::AccuracyResidual,2830,33,0,
            ugkp::diagnostics::EntityKind::Cell,cell,256,residual,margin,0,ugkp::diagnostics::Quantity::Accuracy));
    return finite(seed);
}
template<class R> void tetraQuadrature(const std::array<Vec<R>,4>&p,R volume,int level,WallGeometryDescriptor<R>&d){
    if(level>0){
        const Vec<R> centre=mul(add(add(p[0],p[1]),add(p[2],p[3])),R(.25));
        for(int i=0;i<4;++i){auto child=p;child[i]=centre;tetraQuadrature(child,volume/R(4),level-1,d);}return;
    }
    const R a=(R(5)+R(3)*std::sqrt(R(5)))/R(20),b=(R(5)-std::sqrt(R(5)))/R(20);
    for(int i=0;i<4;++i){Vec<R> q{};for(int j=0;j<4;++j)q=add(q,mul(p[j],i==j?a:b));
        d.quadraturePoints.push_back(q);d.volumeWeight.push_back(volume/R(4));d.distance.push_back(dot(sub(q,d.origin),d.normal));}
}
// Actual tetrahedron/normal-plane intersection. Its triangle or quadrilateral
// section is convex; the positive area centroid stays inside this tetrahedron,
// including when the complete owner's section is disconnected or concave.
template<class R> bool tetraSection(const std::array<Vec<R>,4>&p,const std::array<R,4>&height,
 R y,const WallGeometryDescriptor<R>&d,R&area,Vec<R>&centroid){
    // Keep tiny near-tip sections out of absolute mesh coordinates. Adding a
    // one-ulp-sized section to its world position can collapse or reverse its
    // triangles. Interpolate from the nearer endpoint in a local frame, then
    // restore the world origin only for the final diagnostic centroid.
    int anchor=0;
    for(int i=1;i<4;++i)if(std::abs(height[i]-y)<std::abs(height[anchor]-y))anchor=i;
    std::array<Vec<R>,4> local;
    for(int i=0;i<4;++i)local[i]=sub(p[i],p[anchor]);
    std::array<Vec<R>,4> section;int count=0;
    for(int a=0;a<4;++a)for(int b=a+1;b<4;++b){
        if(!((height[a]<y&&y<height[b])||(height[b]<y&&y<height[a])))continue;
        if(count==4)return false;
        const int from=std::abs(y-height[a])<=std::abs(y-height[b])?a:b,to=from==a?b:a;
        section[count++]=add(local[from],mul(sub(local[to],local[from]),(y-height[from])/(height[to]-height[from])));
    }
    if(count<3)return false;
    Vec<R> centre=section[0];
    for(int i=1;i<count;++i)centre=add(centre,mul(sub(section[i],section[0]),R(1)/R(count)));
    std::array<std::pair<R,int>,4> angle;
    for(int i=0;i<count;++i){const auto v=sub(section[i],centre);angle[i]={std::atan2(dot(v,d.tangent2),dot(v,d.tangent1)),i};}
    std::sort(angle.begin(),angle.begin()+count);
    area=0;Vec<R> offset{};
    for(int i=0;i<count;++i){const auto a=sub(section[angle[i].second],centre),b=sub(section[angle[(i+1)%count].second],centre);
        const R w=dot(cross(a,b),d.normal)/R(2);
        area+=w;offset=add(offset,mul(add(a,b),w/R(3)));
    }
    if(!gasFinite(area)||area==0)return false;
    centroid=add(p[anchor],add(centre,mul(offset,R(1)/area)));return finite(centroid);
}
// The section area is quadratic between projected tetrahedron vertices.
// Cutting again at every BVP node makes area * reconstructed source cubic,
// integrated exactly by two positive Gauss points. The nonlinear first BVP
// interval uses a higher-order rule in eta=sqrt(y/y1), never a singular wall
// point. Geometry and xyz first moments remain polynomial-exact in eta.
template<class R> bool profileOwnerQuadrature(const std::vector<std::array<Vec<R>,4>>&tetra,
 const std::vector<R>&profile,const Context<R>&ctx,WallGeometryDescriptor<R>&d){
    std::vector<std::array<R,4>> height(tetra.size());
    std::vector<R> cut;
    const R roundoff=R(64)*std::numeric_limits<R>::epsilon()*d.maximumOwnerDistance;
    for(std::size_t t=0;t<tetra.size();++t)for(int i=0;i<4;++i){
        R y=dot(sub(tetra[t][i],d.origin),d.normal);if(y<roundoff)y=0;
        height[t][i]=y;cut.push_back(y);
    }
    std::sort(cut.begin(),cut.end());
    std::vector<R> vertices;
    for(R y:cut)if(vertices.empty()||y-vertices.back()>roundoff)vertices.push_back(y);
    // Prefer an exact BVP knot when a geometric break differs only by
    // coordinate roundoff. Otherwise a one-ulp interval has no interior Gauss
    // point. Never move the physical wall at y=0 to a small positive knot.
    for(R& y:vertices)if(y>0){
        auto at=std::lower_bound(profile.begin(),profile.end(),y);
        if(at==profile.end()||(at!=profile.begin()&&std::abs(*(at-1)-y)<std::abs(*at-y)))--at;
        if(*at>0&&std::abs(*at-y)<=roundoff)y=*at;
    }
    vertices.erase(std::unique(vertices.begin(),vertices.end()),vertices.end());
    for(auto&h:height)for(R&y:h){
        auto at=std::lower_bound(vertices.begin(),vertices.end(),y);
        if(at==vertices.end()||(at!=vertices.begin()&&std::abs(*(at-1)-y)<std::abs(*at-y)))--at;
        y=*at;
    }
    cut=vertices;
    for(R y:profile)if(y>vertices.front()&&y<vertices.back())cut.push_back(y);
    std::sort(cut.begin(),cut.end());cut.erase(std::unique(cut.begin(),cut.end()),cut.end());
    const R first=profile[1],gauss=R(1)/std::sqrt(R(3));
    d.quadraturePointsAreInterior=false;
    for(std::size_t i=0;i+1<cut.size();++i){
        const R a=cut[i],b=cut[i+1];if(!(b>a))continue;
        const bool firstInterval=b<=first;
        const R lower=firstInterval?std::sqrt(a/first):a,upper=firstInterval?std::sqrt(b/first):b;
        const R mid=(lower+upper)/R(2),half=(upper-lower)/R(2);
        const int order=firstInterval?static_cast<int>(ctx.firstGaussPoint.size()):2;
        for(int q=0;q<order;++q){
            const R x=firstInterval?ctx.firstGaussPoint[q]:(q==0?-gauss:gauss);
            const R coordinate=mid+half*x,y=firstInterval?first*coordinate*coordinate:coordinate;
            const R dy=firstInterval?half*ctx.firstGaussWeight[q]*R(2)*first*coordinate:half;
            R totalArea=0;Vec<R> firstAreaMoment{};
            for(std::size_t t=0;t<tetra.size();++t){
                const R low=*std::min_element(height[t].begin(),height[t].end()),high=*std::max_element(height[t].begin(),height[t].end());
                if(!(y>low&&y<high))continue;
                R area=0;Vec<R> point;
                if(!tetraSection(tetra[t],height[t],y,d,area,point))return false;
                totalArea+=area;firstAreaMoment=add(firstAreaMoment,mul(sub(point,d.origin),area));
            }
            const R weight=totalArea*dy;
            if(!gasFinite(totalArea)||totalArea==0||!gasFinite(weight))return false;
            const Vec<R> centroid=add(d.origin,mul(firstAreaMoment,R(1)/totalArea));
            d.distance.push_back(y);d.volumeWeight.push_back(weight);d.quadraturePoints.push_back(centroid);
        }
    }
    return true;
}
template<class R> bool rayTriangle(const Vec<R>&o,const Vec<R>&n,const Vec<R>&a,const Vec<R>&b,const Vec<R>&c,R&time){
    const Vec<R> e1=sub(b,a),e2=sub(c,a),p=cross(n,e2);const R det=dot(e1,p);
    const R eps=R(256)*std::numeric_limits<R>::epsilon();
    if(std::abs(det)<=eps*norm(e1)*norm(e2))return false;
    const Vec<R> delta=sub(o,a),q=cross(delta,e1);
    const R u=dot(delta,p)/det,v=dot(n,q)/det;
    if(u<-eps||v<-eps||u+v>R(1)+eps)return false;
    time=dot(e2,q)/det;return gasFinite(time);
}
template<class R> bool matching(const WallGeometryMesh<R>&m,const Context<R>&ctx,WallGeometryDescriptor<R>&d,WallStatus&s,const WallGeometryOptions<R>&,R scale){
    const R tol=R(1024)*std::numeric_limits<R>::epsilon()*scale;
    R entry=0;int cell=d.ownerCell;



    for(;;){
        if(cell<0||(!m.gasCell.empty()&&!m.gasCell[cell]))return invalid(s,d.wallFace);
        R exit=std::numeric_limits<R>::max();int exitFace=-1;bool ambiguous=false;
        for(int f:ctx.cellFaces[cell])for(const auto&t:ctx.faces[f].triangles){
            R hit=0;if(!rayTriangle(d.origin,d.normal,m.points[t.a],m.points[t.b],m.points[t.c],hit)||hit<=entry+tol)continue;
            if(hit<exit-tol){exit=hit;exitFace=f;ambiguous=false;}
            else if(std::abs(hit-exit)<=tol&&f!=exitFace){ambiguous=true;}
        }
        if(exitFace<0||ambiguous)return invalid(s,d.wallFace);
        const R lower=std::max(entry,d.maximumOwnerDistance);
        if(cell!=d.ownerCell&&!ctx.constrained[cell]&&exit>lower+R(4)*tol){
            d.matchingDistance=(lower+exit)/R(2);d.matchingPoint=add(d.origin,mul(d.normal,d.matchingDistance));
            d.matchingDonorDistance=norm(sub(m.cellCentres[cell],d.matchingPoint));
            d.matchingCells.push_back(cell);d.matchingWeights.push_back(R(1));return true;
        }
        if(m.physicalWall[exitFace])return invalid(s,exitFace);
        int next=m.owner[exitFace]==cell?m.neighbour[exitFace]:m.owner[exitFace];
        if(next<0)return invalid(s,exitFace);
        entry=exit;cell=next;
    }
}
template<class R> bool build(const WallGeometryMesh<R>&m,const Context<R>&ctx,int wallFace,WallGeometryDescriptor<R>&out,WallStatus&s,const WallGeometryOptions<R>&opt){
    if(wallFace<0||wallFace>=static_cast<int>(m.owner.size()))return invalid(s,wallFace);
    WallGeometryDescriptor<R>d;d.wallFace=wallFace;d.ownerCell=m.owner[wallFace];d.geometryVersion=m.geometryVersion;
    if(std::size_t(d.ownerCell)>=m.volumes.size()||(!m.gasCell.empty()&&!m.gasCell[d.ownerCell]))return invalid(s,wallFace);
    d.origin=ctx.faces[wallFace].centre;d.normal=mul(ctx.faces[wallFace].normal,R(-1));
    // Choose the construction from polygon topology before tracing a ray:
    // convex polygons use their area centroid; concave polygons use an
    // interior triangle. Neither an attempted ray nor a failed point test
    // selects a second construction.
    bool concave=false;
    const int begin=m.faceOffsets[wallFace],end=m.faceOffsets[wallFace+1],count=end-begin;
    for(int i=0;i<count;++i)
        if(turn(m.points[m.facePoints[begin+i]],m.points[m.facePoints[begin+(i+1)%count]],
            m.points[m.facePoints[begin+(i+2)%count]],ctx.faces[wallFace].normal)<0)concave=true;
    if(concave){
        R largest=0;
        for(const auto&t:ctx.faces[wallFace].triangles){
            const auto&a=m.points[t.a],&b=m.points[t.b],&c=m.points[t.c];
            const R area=norm(cross(sub(b,a),sub(c,a)));
            if(area>largest){largest=area;d.origin=add(a,mul(add(sub(b,a),sub(c,a)),R(1)/R(3)));}
        }
    }
    Vec<R> seed;R scale=0;if(!cellSeed(m,ctx,d.ownerCell,seed,scale,opt.diagnostics))return invalid(s,wallFace);
    for(int f:ctx.cellFaces[d.ownerCell])for(int k=m.faceOffsets[f];k<m.faceOffsets[f+1];++k){
        R y=dot(sub(m.points[m.facePoints[k]],d.origin),d.normal);
        d.maximumOwnerDistance=std::max(d.maximumOwnerDistance,y);
    }
    d.ownerDistance=dot(sub(m.cellCentres[d.ownerCell],d.origin),d.normal);
    int axis=0;for(int k=1;k<3;++k)if(std::abs(d.normal[k])<std::abs(d.normal[axis]))axis=k;
    Vec<R> coordinate{};coordinate[axis]=1;d.tangent1=cross(d.normal,coordinate);d.tangent1=mul(d.tangent1,R(1)/norm(d.tangent1));d.tangent2=cross(d.normal,d.tangent1);
    if(!matching(m,ctx,d,s,opt,scale))return false;
    std::vector<R> profile;
    if(opt.constructQuadrature&&opt.profileNodes){
        profile.resize(opt.profileNodes);
        for(int i=0;i<opt.profileNodes;++i)profile[i]=d.matchingDistance*std::pow(R(i)/R(opt.profileNodes-1),opt.profileStretch);
    }
    R volume=0;Vec<R> moment{};
    std::vector<std::array<Vec<R>,4>> tetra;
    for(int f:ctx.cellFaces[d.ownerCell])for(auto t:ctx.faces[f].triangles){
        if(m.owner[f]!=d.ownerCell)std::swap(t.b,t.c);
        const auto&a=m.points[t.a],&b=m.points[t.b],&c=m.points[t.c];
        R v=dot(sub(a,seed),cross(sub(b,seed),sub(c,seed)))/R(6);
        const Vec<R> centre=mul(add(add(seed,a),add(b,c)),R(.25));
        volume+=v;moment=add(moment,mul(sub(centre,m.cellCentres[d.ownerCell]),v));
        if(!opt.constructQuadrature)continue;
        if(opt.profileNodes)tetra.push_back({{seed,a,b,c}});
        else tetraQuadrature<R>({{seed,a,b,c}},v,opt.quadratureRefinement,d);
    }
    d.geometryVolume=volume;d.volume=m.volumes[d.ownerCell];
    if(!gasFinite(volume)||volume==0||!gasFinite(d.volume)||!finite(moment))return invalid(s,wallFace);
    if(!opt.constructQuadrature){

        const Vec<R> centroid=add(m.cellCentres[d.ownerCell],mul(moment,R(1)/volume));
        d.firstMoment=d.volume*(d.ownerDistance+dot(moment,d.normal)/volume);
        d.firstMomentVector=mul(centroid,d.volume);
        out=std::move(d);return true;
    }
    if(opt.profileNodes&&!profileOwnerQuadrature(tetra,profile,ctx,d))return invalid(s,wallFace);


    R quadratureVolume=0;for(R weight:d.volumeWeight)quadratureVolume+=weight;
    if(!gasFinite(quadratureVolume)||quadratureVolume==0)return invalid(s,wallFace);
    const R factor=d.volume/quadratureVolume;
    for(std::size_t i=0;i<d.distance.size();++i){
        d.volumeWeight[i]*=factor;d.firstMoment+=d.volumeWeight[i]*d.distance[i];
        d.firstMomentVector=add(d.firstMomentVector,mul(d.quadraturePoints[i],d.volumeWeight[i]));
    }
    out=std::move(d);return true;
}
} // wallGeometryDetail
// One-face convenience; production adapters should use the shared cache to
// prepare connectivity and triangulation only once per geometry version.
template<class R> bool buildWallGeometry(const WallGeometryMesh<R>&mesh,int wallFace,WallGeometryDescriptor<R>&out,WallStatus&status,const WallGeometryOptions<R>&options=WallGeometryOptions<R>()){
    status=WallStatus{};wallGeometryDetail::Context<R> context;
    return wallGeometryDetail::prepare(mesh,context,status,options)&&wallGeometryDetail::build(mesh,context,wallFace,out,status,options);
}
template<class R> class WallGeometryCache {
    bool ready_=false;
    std::uint64_t version_=0;
    std::vector<int> wallFaces_;
    WallGeometryOptions<R> options_;
public:
    std::vector<WallGeometryDescriptor<R>> descriptors;
    void clear(){ready_=false;descriptors.clear();wallFaces_.clear();}
    bool rebuild(const WallGeometryMesh<R>&mesh,const std::vector<int>&wallFaces,WallStatus&status,const WallGeometryOptions<R>&options=WallGeometryOptions<R>()){
        status=WallStatus{};
        if(ready_&&version_==mesh.geometryVersion&&wallFaces_==wallFaces
            &&options_.constructQuadrature==options.constructQuadrature
            &&(!options.constructQuadrature || (options_.profileNodes==options.profileNodes
                &&(options.profileNodes
                    ? options_.profileStretch==options.profileStretch&&options_.firstSegmentGaussOrder==options.firstSegmentGaussOrder
                    : options_.quadratureRefinement==options.quadratureRefinement))))return true;
        if(wallFaces.empty()){clear();return true;}
        wallGeometryDetail::Context<R> context;
        if(!wallGeometryDetail::prepare(mesh,context,status,options))return false;
        std::vector<WallGeometryDescriptor<R>> candidate(wallFaces.size());
        for(std::size_t i=0;i<wallFaces.size();++i)if(!wallGeometryDetail::build(mesh,context,wallFaces[i],candidate[i],status,options))return false;
        descriptors.swap(candidate);wallFaces_=wallFaces;version_=mesh.geometryVersion;options_=options;ready_=true;return true;
    }
};
}} // namespace ugkwp::gaswall
