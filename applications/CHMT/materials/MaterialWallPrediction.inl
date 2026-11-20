bool CpuMaterialDriver::predictWindowWall(const HostState& base,const CouplingInterval& interval,WallProgram& output,std::string& error,CpuMaterialReport* diagnostics,bool holdRecession)const{
    ugkp::diagnostics::DeviceContext context{};
    ugkp::diagnostics::reset(context,base.commitSequence+1,reinterpret_cast<std::uintptr_t>(this),base.acceptedSteps,base.time);
    PhysicsConfig physics=data_->model.physics;physics.diagnostics=&context;physics.tolerances.diagnostics=&context;
    const bool result=predictWallImpl(base,interval,output,error,diagnostics,physics,holdRecession);
    auto snapshot=ugkp::diagnostics::takeSnapshot(context,ugkp::diagnostics::sourceCatalogue());
    if(!result&&!snapshot.hasFatal){
        snapshot.hasFatal=true;
        snapshot.fatal=ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::InvalidPhysicalState,4102,4);
        snapshot.fatal.step=context.step;snapshot.fatal.simulationTime=context.simulationTime;snapshot.fatal.instanceId=context.instanceId;
        snapshot.fatal.operationSequence=context.operationSequence;
        snapshot.sources.push_back({4102,"applications/CHMT/materials/CpuMaterialDriver.C",__func__,__LINE__,error});
    }
    snapshot.sources.push_back({4103,"applications/CHMT/materials/CpuMaterialDriver.C","nativeConduction",0,"computed implicit material temperature"});
    snapshot.sources.push_back({4104,"applications/CHMT/materials/CpuMaterialDriver.C","nativeConduction",0,"computed implicit material residual"});
    ugkp::diagnostics::dispatch(snapshot,ugkp::diagnostics::makeFoamSink());
    return result;
}

bool CpuMaterialDriver::predictWall(const HostState& base,const CouplingInterval& interval,WallProgram& output,
    std::string& error,CpuMaterialReport* report)const{
    return predictWindowWall(base,interval,output,error,report,false);
}
