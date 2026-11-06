#!/usr/bin/env python3
"""Build wedge and symmetry meshes for the native/resident owner comparison."""
import argparse
import math
from pathlib import Path
import subprocess
import sys


def header(kind, name):
    return f"FoamFile {{version 2.0; format ascii; class {kind}; object {name};}}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    seeds = args.work / "meshes"
    seeds.mkdir(parents=True, exist_ok=False)
    for geometry in ("wedge", "symmetry", "symmetryPlane"):
        for skew in (0, .35):
            case = seeds / (geometry + ("-orthogonal" if not skew else "-skew"))
            for folder in ("system/solid", "constant/solid", "0/solid"):
                (case / folder).mkdir(parents=True)
            angle = math.radians(2.5)
            vertices = []
            for side in (-1, 1):
                for x, r in ((0, 1), (1, 1), (1, 2), (0, 2)):
                    vertices.append((x + skew * (r - 1), r * math.cos(angle), side * r * math.sin(angle)))
            block = header("dictionary", "blockMeshDict") + "convertToMeters 1;\nvertices\n(\n"
            block += "\n".join("(" + " ".join(format(v, ".17g") for v in point) + ")" for point in vertices)
            block += "\n);\nblocks (hex (0 1 2 3 4 5 6 7) (8 5 1) simpleGrading (1 1 1));\nedges ();\nboundary (\n"
            faces = {"left": ("wall", "0 1 5 4"), "right": ("wall", "3 7 6 2"),
                     "inlet": ("wall", "0 4 7 3"), "outlet": ("wall", "1 2 6 5"),
                     "back": (geometry, "0 3 2 1"), "front": (geometry, "4 5 6 7")}
            for name, (kind, face) in faces.items():
                block += f"{name} {{type {kind}; faces (({face}));}}\n"
            block += ");\nmergePatchPairs ();\n"
            (case / "system/solid/blockMeshDict").write_text(block)
            (case / "system/controlDict").write_text(header("dictionary", "controlDict") +
                "application blockMesh; startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT .25; writeControl timeStep; writeInterval 1;\n")
            (case / "system/solid/fvSchemes").write_text(header("dictionary", "fvSchemes") +
                "ddtSchemes {default Euler;} gradSchemes {default Gauss linear;} divSchemes {default none;} laplacianSchemes {default Gauss linear corrected;} interpolationSchemes {default linear;} snGradSchemes {default corrected;} fluxRequired {default no;}\n")
            temperature = header("volScalarField", "T") + "dimensions [0 0 0 1 0 0 0]; internalField uniform 300; boundaryField {\n"
            for name in faces:
                specification = ("type fixedGradient; gradient uniform 0; value uniform 300;" if name in ("left", "right")
                                 else f"type {geometry};" if name in ("back", "front") else "type zeroGradient;")
                temperature += f"{name} {{{specification}}}\n"
            (case / "0/solid/T").write_text(temperature + "}\n")
            with (case / "blockMesh.log").open("w") as output:
                subprocess.run(["blockMesh", "-region", "solid", "-case", str(case)], stdout=output, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("solid_owner_windows.py")),
                    "--binaries", str(args.binaries), "--meshes", str(seeds), "--work", str(args.work / "comparison"),
                    "--varying-initial-state"], check=True)


if __name__ == "__main__":
    main()
