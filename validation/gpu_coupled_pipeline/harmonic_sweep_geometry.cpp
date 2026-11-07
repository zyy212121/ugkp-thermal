// Reuse the independent geometric-volume/GCL checks. The historical combined
// executable also expects finite residuals to reject motion, a policy removed
// before this change; it is retained separately, not relabelled as passing.
#define main historicalSweepConstraintsMain
#include "../solvers/CHMT/devtools/multirate/test_sweep_constraints.cpp"
#undef main
int main(){
    hardSweepCeilingsValidateAndRejectUnrepresentableMotion();
    denseSmallCellSweepAccuracy();
    stationaryCurvedOffsetsPreserveTranslatedPoints();
    translatedRotatedNonuniformDryMotion();
    actualNonuniformDrySweep();
    explicitIntegratedTargets();
    stationaryNeedsNoDenseSolve();
    nonlinearThreeDimensionalRecession();
    prescribedTrajectoryGclAndComposition();
    rebuildActualTrajectorySurface();
    scaledRotatedDryMotion();
    planarWetMotionStillSupported();
    std::cout<<"PASS: 12 production geometric-motion groups, independent swept volumes and GCL\n";
}
