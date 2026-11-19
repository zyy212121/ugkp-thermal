    static std::size_t historyBatchSize(std::size_t faces){
        return std::max(std::size_t(1),std::size_t(4096)/std::max(std::size_t(1),faces));
    }

    bool evaluateHistory(const HostState& state,const PhysicsConfig& p,
        const std::vector<GasIntervalRecord>& records,Real dt,
        const std::vector<Real>* sideVolume,std::vector<CpuSurfaceResult>& output,std::string& error,int localAudit=-1){
        terminalWallFailure=false;surfaceNumericalFailure=false;
        std::vector<std::vector<GasWallClosureContext>> contexts(records.size());
        if(wall){
            terminalWallFailure=true;
            auto config=p.wallModel;config.enableSst=p.enableSst;
            if(!wall->prepareGeometry(state.gasMesh,state.surface.gasFace,error,config,p.diagnostics))return false;
            ugkwp::GasMechanismView<PhysicalReal,Ns> view;
            if(p.gasMode==ugkwp::GasMode::MixtureChemistry)view=mechanism.mechanismView<Ns>();
            for(std::size_t i=0;i<records.size();++i)
                if(!wall->contexts(records[i].gasWallMatching,view,p.wallModel,contexts[i],error,&surfaceNumericalFailure)){
                    terminalWallFailure=!surfaceNumericalFailure;return false;
                }
        }
        std::vector<MaterialSurfaceRequest> requests;
        for(std::size_t i=0;i<records.size();++i)
            requests.push_back({&records[i].gasTrace,&records[i].gasGradient,wall?&contexts[i]:nullptr,&records[i].gasAcceptedWall.values()});
        terminalWallFailure=false;
        return evaluateResidentMaterialSurfaces(*gpu,state,requests,dt,true,false,output,error,
            surfaceNumericalFailure,sideVolume,&terminalWallFailure,localAudit);
    }

    bool appendAuditKnots(HostState& state,const PhysicsConfig& physics,const IntervalHistory& history,
        const std::vector<MaterialLocalAudit>& observations,WallProgram& program,std::string& error){
        const auto limit=historyBatchSize(state.surface.area.size());
        for(std::size_t begin=0;begin<observations.size();){
            const auto& first=observations[begin];
            std::size_t end=begin+1;
            while(end<observations.size()&&end-begin<limit&&observations[end].heldBegin==first.heldBegin
                &&observations[end].heldEnd==first.heldEnd&&observations[end].dt==first.dt)++end;
            std::vector<GasIntervalRecord> records;
            records.reserve(end-begin);
            for(std::size_t i=begin;i<end;++i)records.push_back(history.readRecord(observations[i].record));
            std::vector<CpuSurfaceResult> response;
            state.time=ugkp::absoluteTime(program.interval.begin,first.time);
            if(!evaluateHistory(state,physics,records,first.dt,nullptr,response,error,int(begin)))return false;
            for(std::size_t i=begin;i<end;++i){
                WallKnot knot;
                knot.time=ugkp::absoluteTime(program.interval.begin,observations[i].time);
                knot.localTime=records[i-begin].localBegin;
                knot.faces=std::move(response[i-begin].wall);
                knot.gasPoints=state.gasMesh.points;
                knot.solidPoints=state.solidMesh.points;
                program.knots.push_back(std::move(knot));
            }
            begin=end;
        }
        return true;
    }
