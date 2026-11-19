
#include "gpu/CoupledHistoryArchive.cuh"
#include "materials/MaterialDrive.H"
#include "coupling/IntervalAudit.H"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace chmt;
static void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
__global__ void fill(CoupledHistoryArchive::Frame frame,int index,int faces){
    const int f=threadIdx.x;if(f>=faces)return;
    auto& row=frame.surface[f];row.bulk.temperature=300+index+f*.25;
    row.bulk.pressure=100000;row.bulk.rho=1;row.bulk.Y[0]=1;
    row.primary.kind=ExchangeKind::GasSolid;row.primary.solidCell=f;row.primary.gasCell=f;
    row.primary.step=index;row.primary.face=f;row.primary.consumerMask=ConsumeGas;
    row.primary.energy=index*.125+f;row.radiationReceiver=ConsumeSolid;
    if(frame.wall)frame.wall[f].matching.pressure=12345+index+f;
    if(!f){frame.interval->begin=ugkp::CompensatedTime(index/1024.);
        frame.interval->end=ugkp::CompensatedTime((index+1)/1024.);frame.interval->actualDt=1./1024;}
}
int main(){
    constexpr int count=1025,faces=128;
    cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CoupledHistoryArchive::TransferPool pool{};
    for(int length:{0,1,255,256,257}){
        auto prior=std::make_shared<CoupledHistoryArchive>(1,false,&pool);
        for(int i=0;i<length;++i){
            fill<<<1,1,0,stream>>>(prior->next(stream),i,1);
            prior->accepted(stream);
        }
        prior->seal(stream);
        assert(prior->spilled()==(length>256));
        prior->releaseProducer(stream);
        auto next=std::make_shared<CoupledHistoryArchive>(1,false,&pool);
        fill<<<1,1,0,stream>>>(next->next(stream),9000,1);
        next->accepted(stream);next->seal(stream);
        if(length){
            const auto verify=[&]{
                auto tail=prior->chunk(length-1);
                assert(tail->surface[length-1-tail->first].bulk.temperature==300+length-1);
            };
            std::thread reader(verify);verify();reader.join();
            int visited=0;
            prior->forEachSurfaceFrame([&](const CoupledSurfaceDownload* row,std::size_t n){
                assert(n==1 && row->bulk.temperature==300+visited);++visited;
            });
            assert(visited==length);
        }
        assert(next->chunk(0)->surface[0].bulk.temperature==9300);
    }
    auto archive=std::make_shared<CoupledHistoryArchive>(faces,true);
    std::size_t available=0,total=0;check(cudaMemGetInfo(&available,&total));
    for(int i=0;i<count;++i){const auto frame=archive->next(stream);
        check(cudaMemsetAsync(frame.surface,0,faces*sizeof(CoupledSurfaceDownload),stream));
        check(cudaMemsetAsync(frame.wall,0,faces*sizeof(CoupledWallDownload),stream));
        check(cudaMemsetAsync(frame.interval,0,sizeof(ResidentMaterialDriveInterval),stream));
        fill<<<1,128,0,stream>>>(frame,i,faces);check(cudaGetLastError());archive->accepted(stream);

        if(i==511){check(cudaStreamSynchronize(stream));check(cudaMemGetInfo(&available,&total));}
    }
    archive->seal(stream);assert(archive->frames()==count && archive->spilled());
    std::size_t after=0;check(cudaMemGetInfo(&after,&total));
    assert(after+16*1024*1024>=available);
    for(int i:{1024,0,255,256,511,512,1,768}){
        const auto chunk=archive->chunk(i);const auto row=(i-chunk->first)*faces;
        for(int f=0;f<faces;++f){assert(chunk->surface[row+f].bulk.temperature==300+i+f*.25);
            assert(chunk->surface[row+f].primary.energy==i*.125+f);
            assert(chunk->wall[row+f].matching.pressure==12345+i+f);}
        assert(chunk->interval[i-chunk->first].actualDt==1./1024);
    }
    bool missing=false;try{archive->chunk(count);}catch(const std::runtime_error& e){missing=std::strstr(e.what(),"code=8");}
    assert(missing);
    CouplingInterval window;window.end=count/1024.;IntervalHistory eager,lazy;std::string error;
    assert(eager.begin(window,error)&&lazy.begin(window,error));
    const auto populate=[archive](std::size_t frame,GasIntervalRecord& record){
        const auto chunk=archive->chunk(frame);const auto offset=(frame-chunk->first)*faces;
        for(int f=0;f<faces;++f){const auto& row=chunk->surface[offset+f];
            record.gasTrace.push_back(row.bulk);record.gasGradient.push_back(row.gradient);
            record.gasTraction.push_back(row.traction);record.packets.push_back(row.primary);
            record.radiationEnergy.push_back(0);record.radiationReceiver.push_back(ConsumeSolid);
        }
    };
    for(int i=0;i<count;++i){GasIntervalRecord record;record.microSequence=i+1;
        record.begin=i/1024.;record.end=(i+1)/1024.;record.actualDt=1./1024;
        assert(lazy.appendAccepted(record,error));populate(i,record);assert(eager.appendAccepted(record,error));}
    lazy.setPayloadReader(populate);assert(lazy.complete());
    auto a=eager.cursor(),b=lazy.cursor();
    for(double end:{.1255,.50025,.75,count/1024.}){
        std::vector<ExchangePacket> p,q;
        using Energy=decltype(ExchangePacket{}.energy);std::vector<Energy> r,s;
        assert(a.takeThroughLocal(WindowTime(end),p,r,error)&&b.takeThroughLocal(WindowTime(end),q,s,error));
        assert(p.size()==q.size()&&r==s);
        for(std::size_t i=0;i<p.size();++i){assert(p[i].energy==q[i].energy&&p[i].step==q[i].step&&p[i].face==q[i].face);}
    }
    assert(a.complete()&&b.complete());
    MaterialDriveCursor eagerDrive(eager),lazyDrive(lazy);
    for(double begin:{0.,.25,.5,.75}){WindowTime first,last;
        assert(eagerDrive.slabEndLocal(WindowTime(begin),WindowTime(window.end),.05,1,first,error));
        assert(lazyDrive.slabEndLocal(WindowTime(begin),WindowTime(window.end),.05,1,last,error));assert(first==last);
    }
    HostState lhs,rhs;assert(finalizeIntervalLedger(eager,lhs,error)&&finalizeIntervalLedger(lazy,rhs,error));
    const auto& p=lhs.ledger.values();const auto& q=rhs.ledger.values();assert(p.size()==q.size());
    for(std::size_t i=0;i<p.size();++i)assert(p[i].energy==q[i].energy&&p[i].consumerMask==q[i].consumerMask);
    check(cudaStreamDestroy(stream));
    std::cout<<"bounded archive: four full slabs plus tail, random access, exact partial slices, drive selection, consumed ledger passed\n";
}
