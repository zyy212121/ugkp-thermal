#!/usr/bin/env python3
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def block(source: str, marker: str) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise RuntimeError(f"unterminated block: {marker}")


def compact(value: str) -> str:
    return "".join(value.split())


def main() -> None:
    require(len(sys.argv) == 2, "usage: test_staged_coupling_contract.py CORE")
    source = Path(sys.argv[1]).read_text(encoding="utf-8")
    active = compact(block(source, "gasDragModelActive"))
    volume = compact(block(source, "applyGasVolumeFractionSourceKernel"))
    drag = compact(block(source, "applyEulerianGasSolidDragKernel"))
    heat = compact(block(source, "applyEulerianParticleMaterialHeatKernel"))
    initializer = compact(block(source, "initialiseEpsGPrevKernel"))
    advance = compact(block(source, 'extern "C" int ugkwpGpuResidentStrictAdvance'))
    drag_launch = compact(block(source, "int launchEulerianGasSolidDrag"))
    particle_launch = compact(block(source, "int launchParticleDragRelaxation"))
    particle_relax = compact(block(source, "relaxOneParticleToResidentGas"))

    require("returntrue;" in active and "!=2" not in active, "constant drag is inactive")
    require(
        "enerGOld*massScale+dt*cepsG*pressureOld" in volume,
        "pressure-volume work is missing",
    )
    require("constdoublemg=epsG*rhoG;" in drag, "apparent gas mass is missing")
    require("constdoublemomGX0=epsG*s.rhoUx[c];" in drag, "apparent gas momentum is missing")
    require("constdoubleenerG0=epsG*s.rhoE[c];" in drag, "apparent gas energy is missing")
    require("s.rhoE[c]=(kgNew+igAfter+diss)/epsGsafe;" in drag, "intrinsic gas energy writeback is missing")
    require("constdoublegasCapacity=epsG*rhoG*gasHeatCapacity;" in heat, "apparent gas heat capacity is missing")
    require("gasCv/clampMin(gasHeatCapacity,OfSmall)" in heat, "gas energy-capacity conversion is missing")
    require("rhoEold-gasEnergyScale*dHp/epsGsafe" in heat, "intrinsic heat writeback is missing")
    require("dragModel" not in initializer, "volume history still depends on drag")
    require("s.epsGPrev[c]=1.0-eps;" in initializer, "volume history initialization is missing")

    volume_launch = advance.index("applyGasVolumeFractionSourceKernel<<<grid,block>>>")
    drag_gate = advance.index("if(dragActive)")
    drag_call = advance.index("launchEulerianGasSolidDrag", drag_gate)
    require(volume_launch < drag_gate < drag_call, "volume source remains inside the drag gate")
    require(advance.count("applyGasVolumeFractionSourceKernel<<<grid,block>>>") == 1, "volume source launch count is invalid")
    require("case2:" in drag_launch and "ConstantResponseTimeDeviceDrag" in drag_launch, "Eulerian constant drag launch is missing")
    require("case2:" in particle_launch and "ConstantResponseTimeDeviceDrag" in particle_launch, "particle constant drag launch is missing")
    require("gasDragModelActive(s.dragModel)" in particle_relax, "particle drag activity is not centralized")
    require("s->hostDragModel!=2" not in compact(source), "direct host drag gate remains")
    require("s.dragModel!=2" not in compact(source), "direct device drag gate remains")
    print("staged wind-sand coupling contract: PASS")


if __name__ == "__main__":
    main()
