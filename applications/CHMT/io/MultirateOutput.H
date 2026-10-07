#ifndef CHMT_IO_MULTIRATEOUTPUT_H
#define CHMT_IO_MULTIRATEOUTPUT_H
#include "io/Output.H"
#include "io/MultirateEvolution.H"
#include "coupling/ExchangeLedger.H"
#include "materials/Reaction.H"

namespace chmt { namespace io {
// This function accepts the frontend's committed owner only. Provisional GPU
// downloads and CPU candidates never enter this output/checkpoint path.
inline void requireSynchronizedState(const HostState& state) {
    require(finite(state.time) && state.time>=0, "invalid synchronized time");
    for (const auto& packet:state.ledger)
        require(packet.consumerMask == requiredConsumers(packet.kind), "cannot output a partially consumed exchange");
    for (const auto& storage:state.filmStorage)
        require(storage.consumed, "cannot output unconsumed film storage");
}

inline void publishSolidMesh(Foam::fvMesh& native, const HostMesh& accepted, const Tolerances& tolerance) {
    require(accepted.points.size() == std::size_t(native.nPoints())
        && accepted.volumes.size() == std::size_t(native.nCells())
        && accepted.owner.size() == std::size_t(native.nFaces())
        && accepted.neighbour.size() == accepted.owner.size()
        && accepted.faceOffsets.size() == accepted.owner.size()+1
        && !accepted.faceOffsets.empty() && accepted.faceOffsets.front() == 0
        && accepted.faceOffsets.back() == static_cast<int>(accepted.facePoints.size()),
        "accepted solid mesh does not match the actual OpenFOAM region");
    const auto& faces = native.faces();
    const auto& owner = native.faceOwner();
    const auto& neighbour = native.faceNeighbour();
    for (Foam::label f=0; f<native.nFaces(); ++f) {
        require(accepted.owner[f] == owner[f]
            && accepted.neighbour[f] == (f<neighbour.size()?neighbour[f]:-1)
            && accepted.faceOffsets[f+1]-accepted.faceOffsets[f] == faces[f].size(),
            "solid topology change is unsupported at macro publication");
        forAll(faces[f], vertex)
            require(accepted.facePoints[accepted.faceOffsets[f]+vertex] == faces[f][vertex],
                "solid vertex topology differs from actual OpenFOAM region");
    }
    Foam::pointField points(native.points());
    bool changed = false;
    forAll(points, pointIndex) {
        const auto& p = accepted.points[pointIndex];
        require(finite(p), "nonfinite accepted solid point");
        const Foam::point next(p.x,p.y,p.z);
        changed = changed || points[pointIndex] != next;
        points[pointIndex] = next;
    }
    if (changed) native.movePoints(points);
    forAll(native.V(), cell)
        require(closeEnough(native.V()[cell],accepted.volumes[cell],tolerance.absoluteGeometry,tolerance.relativeGeometry),
            "published OpenFOAM solid volume disagrees with accepted geometry");
}

inline void startWindowOutput(const Output& output, const CouplingControls& controls, const CpuMaterialControls& material, bool writeGeometry) {
    fileText(output.directory()/"windows.csv",
        "window,time,coupling_interval,next_gas_dt,next_coupling_interval,gas_microsteps,accepted_gas_microsteps,material_solves,material_substeps,reaction_substeps,film_substeps,linear_solves,boundary_drive_samples,material_rhs_evaluations,window_iterations,window_retries,gas_rejected_trials,temperature_prediction_error,mass_prediction_error,geometry_error,energy_residual,numerical_mass_roundoff\n");
    std::ostringstream metadata;
    metadata<<std::setprecision(17)<<"{\"execution_mode\":\"Multirate\",\"coupling_method\":\""
        <<(controls.enforcePredictionAccuracy?"iterated_waveform":"first_order_lagged")
        <<"\",\"coupling_interval\":"<<controls.interval<<",\"minimum_coupling_interval\":"<<controls.minimumInterval
        <<",\"gas_max_dt\":"<<controls.gasMaxDt<<",\"material_max_substep\":"<<controls.materialMaxSubstep
        <<",\"prediction_tolerance\":"<<controls.predictionTolerance
        <<",\"max_window_iterations\":"<<controls.maxWindowIterations
        <<",\"max_window_retries\":"<<controls.maxWindowRetries
        <<",\"drive_relative_tolerance\":"<<material.driveRelativeTolerance
        <<",\"drive_source_fraction\":"<<material.driveSourceFraction
        <<",\"drive_traction_scale_pa\":"<<material.driveTractionScale
        <<",\"checkpoint_kind\":\"synchronized_only\",\"restart_predictor\":\"rebuild_from_accepted_state\"}\n";
    fileText(output.directory()/"multirate.json",metadata.str());
    if (writeGeometry) fileText(output.directory()/"gcl.csv",
        "window,micro,region,stage,cell,V_old,V_new,signed_swept_volume,volume_residual\n");
}

inline void writeWindowOutput(const Output& output, const HostState& state, const WindowEvolutionReport& r) {
    requireSynchronizedState(state);
    std::ofstream history((output.directory()/"windows.csv").c_str(),std::ios::app);
    history<<std::setprecision(17)<<state.commitSequence<<','<<state.time<<','<<r.interval.end-r.interval.begin
        <<','<<r.nextGasDt<<','<<r.nextCouplingInterval<<','<<r.gasMicrosteps<<','<<r.acceptedGasMicrosteps
        <<','<<r.materialSolves<<','<<r.materialSubsteps
        <<','<<r.reactionSubsteps<<','<<r.filmSubsteps<<','<<r.linearSolves
        <<','<<r.boundaryDriveSamples<<','<<r.materialRhsEvaluations<<','<<r.windowIterations
        <<','<<r.windowRetries<<','<<r.gasRejectedTrials<<','<<r.temperaturePredictionError
        <<','<<r.massPredictionError<<','<<r.geometryError<<','<<r.energyResidual
        <<','<<r.numericalMassRoundoff<<'\n';
    history.flush();
    require(bool(history),"macro window report write failed");
    if (r.geometry.empty()) return;
    std::ofstream gcl((output.directory()/"gcl.csv").c_str(),std::ios::app);
    gcl<<std::setprecision(17);
    for (const auto& micro:r.geometry) for (int region=0; region<2; ++region) {
        const HostMesh& mesh = region==0?state.gasMesh:state.solidMesh;
        if (mesh.volumes.empty()) continue;
        const auto& stages = region==0?micro.gas:micro.solid;
        for (int stage=0; stage<2; ++stage) {
            const auto& s = stages[stage];
            require(s.interval>0 && s.oldVolume.size()==mesh.volumes.size()
                && s.newVolume.size()==mesh.volumes.size() && s.sweptVolume.size()==mesh.owner.size()
                && s.areaVector.size()==mesh.owner.size(),"missing actual gas-microstep stage geometry");
            std::ostringstream faces;
            faces<<std::setprecision(17)<<"face,persistent_id,owner,neighbour,swept_volume,area_x,area_y,area_z\n";
            for (std::size_t f=0; f<mesh.owner.size(); ++f)
                faces<<f<<','<<mesh.faceIds.at(f)<<','<<mesh.owner[f]<<','<<mesh.neighbour[f]<<','<<s.sweptVolume[f]
                    <<','<<s.areaVector[f].x<<','<<s.areaVector[f].y<<','<<s.areaVector[f].z<<'\n';
            const std::string name = "raw-window-"+std::to_string(state.commitSequence)+"-micro-"
                +std::to_string(micro.microSequence)+(region==0?"-gas-":"-solid-")+std::to_string(stage)+".csv";
            fileText(output.directory()/Foam::fileName(name),faces.str());
            for (std::size_t cell=0; cell<mesh.volumes.size(); ++cell) {
                Real sweep = 0;
                for (int at=mesh.cellFaceOffsets.at(cell); at<mesh.cellFaceOffsets.at(cell+1); ++at)
                    sweep += mesh.cellFaceSigns[at]*s.sweptVolume.at(mesh.cellFaces[at]);
                gcl<<state.commitSequence<<','<<micro.microSequence<<','<<(region==0?"gas":"solid")<<','<<stage
                    <<','<<cell<<','<<s.oldVolume[cell]<<','<<s.newVolume[cell]<<','<<sweep
                    <<','<<s.newVolume[cell]-s.oldVolume[cell]-sweep<<'\n';
            }
        }
    }
    gcl.flush();
    require(bool(gcl),"macro gas-stage geometry write failed");
}

inline void writeSolidOutput(const Output& output, const HostState& state, const PhysicsConfig& physics) {
    requireSynchronizedState(state);
    if (state.solid.empty()) return;
    std::ostringstream raw;
    raw<<std::setprecision(17)<<"cell,volume,energy,temperature,porosity,pore_pressure";
    for (int c=0; c<Nc; ++c) raw<<",condensed_"<<c;
    for (int s=0; s<Ns; ++s) raw<<",pore_"<<s;
    for (int reaction=0; reaction<Nr; ++reaction) raw<<",progress_"<<reaction;
    raw<<'\n';
    for (std::size_t cell=0; cell<state.solid.size(); ++cell) {
        const auto& q = state.solid[cell];
        MaterialPrimitive primitive;
        require(recoverMaterial(q,state.solidMesh.volumes[cell],physics,primitive),"accepted CPU solid diagnostic recovery failed");
        raw<<cell<<','<<state.solidMesh.volumes[cell]<<','<<q.energy<<','<<primitive.temperature
            <<','<<q.porosity<<','<<primitive.pressure;
        for (int c=0; c<Nc; ++c) raw<<','<<q.condensed[c];
        for (int s=0; s<Ns; ++s) raw<<','<<q.pore[s];
        for (int reaction=0; reaction<Nr; ++reaction) raw<<','<<q.progress[reaction];
        raw<<'\n';
    }
    fileText(output.directory()/Foam::fileName("raw-solid-"+std::to_string(state.acceptedSteps)+".csv"),raw.str());
}
}}
#endif
