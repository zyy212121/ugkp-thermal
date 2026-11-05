#ifndef UGKP_GPU_POLYHEDRAL_GEOMETRY_MATH_H
#define UGKP_GPU_POLYHEDRAL_GEOMETRY_MATH_H
#include <cmath>
#include <limits>
#ifdef __CUDACC__
#define UGKP_GEOMETRY_HD __host__ __device__
#else
#define UGKP_GEOMETRY_HD
#endif
namespace ugkp { namespace gpu { namespace geometry {
struct Point { double x=0,y=0,z=0; };
UGKP_GEOMETRY_HD inline Point operator+(Point a,Point b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
UGKP_GEOMETRY_HD inline Point operator-(Point a,Point b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
UGKP_GEOMETRY_HD inline Point operator*(Point a,double b){return {a.x*b,a.y*b,a.z*b};}
UGKP_GEOMETRY_HD inline Point operator/(Point a,double b){return a*(1/b);}
UGKP_GEOMETRY_HD inline Point& operator+=(Point& a,Point b){a=a+b;return a;}
UGKP_GEOMETRY_HD inline double dot(Point a,Point b){return a.x*b.x+a.y*b.y+a.z*b.z;}
UGKP_GEOMETRY_HD inline Point cross(Point a,Point b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
UGKP_GEOMETRY_HD inline double magnitude(Point a){return ::sqrt(dot(a,a));}
UGKP_GEOMETRY_HD inline bool finite(Point a){return std::isfinite(a.x)&&std::isfinite(a.y)&&std::isfinite(a.z);}
UGKP_GEOMETRY_HD inline Point triangleArea(Point a,Point b,Point c){return cross(b-a,c-a)*.5;}
struct TopologyView {
    const int *faceOffsets,*facePoints,*cellOffsets,*cellFaces,*cellSigns,*vertexOffsets,*vertices;
};
struct FaceResult {Point area,centre;int invalid=0;};
struct CellResult {Point centre,closure;double volume=0,areaSum=0;int invalid=0;};
struct SweepResult {Point area;double volume=0;int invalid=0;};
UGKP_GEOMETRY_HD inline FaceResult faceGeometry(int face,TopologyView t,const Point* points){
    FaceResult r;const int begin=t.faceOffsets[face],end=t.faceOffsets[face+1];const Point a=points[t.facePoints[begin]];
    for(int j=begin+1;j+1<end;++j)r.area+=triangleArea(a,points[t.facePoints[j]],points[t.facePoints[j+1]]);
    const double area=magnitude(r.area);if(!(area>0)||!std::isfinite(area)){r.invalid=1;return r;}
    const Point normal=r.area/area;double weight=0;
    for(int j=begin+1;j+1<end;++j){const Point b=points[t.facePoints[j]],c=points[t.facePoints[j+1]];
        const double w=dot(triangleArea(a,b,c),normal);r.centre+=(a+b+c)*(w/3);weight+=w;}
    r.centre=r.centre/weight;return r;
}
UGKP_GEOMETRY_HD inline CellResult cellGeometry(int cell,TopologyView t,const Point* points,const Point* area){
    CellResult r;Point reference;
    for(int k=t.vertexOffsets[cell];k<t.vertexOffsets[cell+1];++k)reference+=points[t.vertices[k]];
    reference=reference/double(t.vertexOffsets[cell+1]-t.vertexOffsets[cell]);Point moment;
    for(int k=t.cellOffsets[cell];k<t.cellOffsets[cell+1];++k){const int face=t.cellFaces[k],sign=t.cellSigns[k];
        r.closure+=area[face]*sign;r.areaSum+=magnitude(area[face]);
        const int begin=t.faceOffsets[face],end=t.faceOffsets[face+1];const Point a=points[t.facePoints[begin]];
        for(int j=begin+1;j+1<end;++j){const Point b=points[t.facePoints[j]],d=points[t.facePoints[j+1]];
            const double tetra=sign*dot(a-reference,cross(b-reference,d-reference))/6;
            r.volume+=tetra;moment+=(reference+a+b+d)*(tetra/4);}}
    if(!std::isfinite(r.volume)||r.volume<=0){r.invalid=1;return r;}
    r.centre=moment/r.volume;if(!finite(r.centre))r.invalid=2;return r;
}
UGKP_GEOMETRY_HD inline SweepResult faceSweep(int face,TopologyView t,const Point* old,const Point* endpoint){
    SweepResult r;const int begin=t.faceOffsets[face],end=t.faceOffsets[face+1],ia=t.facePoints[begin];
    for(int j=begin+1;j+1<end;++j){const int ib=t.facePoints[j],ic=t.facePoints[j+1];
        const Point a=old[ia],b=old[ib],c=old[ic],da=endpoint[ia]-a,db=endpoint[ib]-b,dc=endpoint[ic]-c;
        const Point average=(triangleArea(a,b,c)+triangleArea(a+da*.5,b+db*.5,c+dc*.5)*4+triangleArea(a+da,b+db,c+dc))/6;
        r.area+=average;r.volume+=dot((da+db+dc)/3,average);}
    if(!finite(r.area)||!std::isfinite(r.volume))r.invalid=1;return r;
}
struct PlanarConstraint {int a,b,c,d;double scale;};
struct Polynomial {double value[3];};
UGKP_GEOMETRY_HD inline double triple(Point a,Point b,Point c){return dot(a,cross(b,c));}
UGKP_GEOMETRY_HD inline double tripleDerivative(Point a,Point b,Point c,Point da,Point db,Point dc){
    return triple(da,b,c)+triple(a,db,c)+triple(a,b,dc);
}
UGKP_GEOMETRY_HD inline double sweepDerivative(TopologyView t,const Point* old,const Point* end,const Point* mode,int f){
    double result=0;const int begin=t.faceOffsets[f],stop=t.faceOffsets[f+1],ia=t.facePoints[begin];
    for(int k=begin+1;k+1<stop;++k){const int ib=t.facePoints[k],ic=t.facePoints[k+1];
        const Point a=old[ia],b=old[ib],c=old[ic],da=end[ia]-a,db=end[ib]-b,dc=end[ic]-c;
        const Point ea=mode[ia],eb=mode[ib],ec=mode[ic];
        const Point u=b-a,v=c-a,du=db-da,dv=dc-da,eu=eb-ea,ev=ec-ea;
        const Point area=cross(u,v)*.5+(cross(du,v)+cross(u,dv))*.25+cross(du,dv)/6;
        const Point derivative=(cross(eu,v)+cross(u,ev))*.25+(cross(eu,dv)+cross(du,ev))/6;
        result+=dot((ea+eb+ec)/3,area)+dot((da+db+dc)/3,derivative);
    }return result;
}
UGKP_GEOMETRY_HD inline Polynomial planarPolynomial(const Point* old,const Point* end,PlanarConstraint p,const Point* mode=nullptr){
    Polynomial out;const Point u=old[p.b]-old[p.a],v=old[p.c]-old[p.a],w=old[p.d]-old[p.a];
    const Point da=end[p.a]-old[p.a],du=end[p.b]-old[p.b]-da,dv=end[p.c]-old[p.c]-da,dw=end[p.d]-old[p.d]-da;
    if(!mode){
        out.value[0]=triple(du,v,w)+triple(u,dv,w)+triple(u,v,dw);
        out.value[1]=triple(du,dv,w)+triple(du,v,dw)+triple(u,dv,dw);
        out.value[2]=triple(du,dv,dw);
    }else{
        const Point eu=mode[p.b]-mode[p.a],ev=mode[p.c]-mode[p.a],ew=mode[p.d]-mode[p.a];
        out.value[0]=triple(eu,v,w)+triple(u,ev,w)+triple(u,v,ew);
        out.value[1]=tripleDerivative(du,dv,w,eu,ev,{})+tripleDerivative(du,v,dw,eu,{},ew)+tripleDerivative(u,dv,dw,{},ev,ew);
        out.value[2]=tripleDerivative(du,dv,dw,eu,ev,ew);
    }return out;
}
UGKP_GEOMETRY_HD inline Point normalized(Point a){const double m=magnitude(a);return m>0&&std::isfinite(m)?a/m:Point{};}
// Face identities use volume-mesh indices; periodic neighbour endpoint indices
// retain the canonical topological correspondence supplied by the application.
struct SurfaceEdge {int a,b,ownerFace,neighbourFace,neighbourA,neighbourB;};
struct EdgeResult {Point conormal,ownerOffset,neighbourOffset;double length=0,sweep=0,residual=0,tolerance=0;int invalid=0,converged=0;};
constexpr double geometryEpsilon=std::numeric_limits<double>::epsilon();
constexpr double undefinedGeometry=std::numeric_limits<double>::quiet_NaN();
UGKP_GEOMETRY_HD inline Point trajectoryPoint(const Point* old,const Point* end,int p,double time,bool empty){
    return empty?old[p]+(end[p]-old[p])*time:old[p]*(1-time)+end[p]*time;
}
UGKP_GEOMETRY_HD inline Point faceAreaAt(TopologyView t,const Point* old,const Point* end,int f,double time,bool empty){
    Point result;const int begin=t.faceOffsets[f],stop=t.faceOffsets[f+1];
    const Point a=trajectoryPoint(old,end,t.facePoints[begin],time,empty);
    for(int k=begin+1;k+1<stop;++k)result+=triangleArea(a,trajectoryPoint(old,end,t.facePoints[k],time,empty),trajectoryPoint(old,end,t.facePoints[k+1],time,empty));
    return result;
}
UGKP_GEOMETRY_HD inline double edgeIntegrand(TopologyView t,const Point* old,const Point* end,SurfaceEdge edge,double time,bool empty){
    const Point a=trajectoryPoint(old,end,edge.a,time,empty),b=trajectoryPoint(old,end,edge.b,time,empty);
    Point normal=normalized(faceAreaAt(t,old,end,edge.ownerFace,time,empty));
    if(magnitude(normal)==0)return undefinedGeometry;
    if(edge.neighbourFace>=0){const Point other=normalized(faceAreaAt(t,old,end,edge.neighbourFace,time,empty));
        if(magnitude(other)==0)return undefinedGeometry;normal=normalized(normal+other);}
    if(magnitude(normal)<=1e-12||magnitude(b-a)<=0)return undefinedGeometry;
    const Point displacement=(end[edge.a]-old[edge.a]+end[edge.b]-old[edge.b])*.5;
    return dot(displacement,cross(b-a,normal));
}
UGKP_GEOMETRY_HD inline double edgeQuadrature(TopologyView t,const Point* old,const Point* end,SurfaceEdge edge,int segments,bool empty){
    constexpr double x=.774596669241483377;double sum=0;
    for(int segment=0;segment<segments;++segment){const double a=double(segment)/segments,b=double(segment+1)/segments,c=.5*(a+b),h=.5*(b-a);
        sum+=h*((5.0/9)*edgeIntegrand(t,old,end,edge,c-h*x,empty)+(8.0/9)*edgeIntegrand(t,old,end,edge,c,empty)
            +(5.0/9)*edgeIntegrand(t,old,end,edge,c+h*x,empty));}
    return sum;
}
UGKP_GEOMETRY_HD inline EdgeResult surfaceEdge(TopologyView t,const Point* old,const Point* end,SurfaceEdge edge,
    bool empty,int prescribedSegments,double meshScale){
    EdgeResult r;
    if(prescribedSegments){r.sweep=edgeQuadrature(t,old,end,edge,prescribedSegments,empty);r.invalid=std::isfinite(r.sweep)?0:2;r.converged=!r.invalid;return r;}
    const Point tangent=end[edge.b]-end[edge.a];const auto owner=faceGeometry(edge.ownerFace,t,end);
    Point normal=normalized(owner.area);FaceResult neighbour;
    if(edge.neighbourFace>=0){neighbour=faceGeometry(edge.neighbourFace,t,end);normal=normalized(normal+normalized(neighbour.area));}
    r.conormal=normalized(cross(tangent,normal));if(magnitude(r.conormal)==0){r.invalid=1;return r;}
    const Point centre=(end[edge.a]+end[edge.b])*.5;r.ownerOffset=centre-owner.centre;
    if(edge.neighbourFace>=0)r.neighbourOffset=(end[edge.neighbourA]+end[edge.neighbourB])*.5-neighbour.centre;
    r.length=magnitude(tangent);r.sweep=edgeQuadrature(t,old,end,edge,8,empty);const double scale=meshScale*meshScale;
    for(int segments=16;segments<=1024;segments*=2){const double refined=edgeQuadrature(t,old,end,edge,segments,empty);
        if(!std::isfinite(refined)||!std::isfinite(r.sweep)){r.invalid=2;return r;}
        const double a=::fabs(refined),b=::fabs(r.sweep);
        r.tolerance=256*geometryEpsilon*scale+1e-11*(a>b?a:b);
        r.residual=::fabs(refined-r.sweep);r.sweep=refined;
        if(r.residual<=r.tolerance){r.converged=1;break;}
    }return r;
}
}}}
#undef UGKP_GEOMETRY_HD
#endif
