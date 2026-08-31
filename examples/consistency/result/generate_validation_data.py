#!/usr/bin/env python3
from pathlib import Path
import csv
import math
import re
import struct
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def latest(case):
    values = []
    for path in case.iterdir():
        try:
            value = float(path.name)
        except ValueError:
            continue
        if path.is_dir() and value > 0:
            values.append((value, path))
    return max(values)


def scalar_field(path, size):
    text = path.read_text(encoding="utf-8")
    uniform = re.search(rf"internalField\s+uniform\s+({NUMBER})\s*;", text)
    if uniform:
        return np.full(size, float(uniform.group(1)))
    match = re.search(r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S)
    if not match:
        raise RuntimeError(path)
    values = np.fromstring(match.group(2), sep=" ")
    if values.size != size or int(match.group(1)) != size:
        raise RuntimeError(path)
    return values


def vector_field(path, size):
    text = path.read_text(encoding="utf-8")
    uniform = re.search(rf"internalField\s+uniform\s*\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)\s*;", text)
    if uniform:
        value = np.array([float(uniform.group(i)) for i in range(1, 4)])
        return np.repeat(value[None, :], size, axis=0)
    match = re.search(r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S)
    if not match:
        raise RuntimeError(path)
    rows = re.findall(rf"\(\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*\)", match.group(2))
    values = np.asarray(rows, dtype=float)
    if values.shape != (size, 3) or int(match.group(1)) != size:
        raise RuntimeError(path)
    return values


GAMMA = 1.4
LEFT = (1.0, 0.0, 100000.0)
RIGHT = (0.125, 0.0, 10000.0)


def pressure_function(pressure, state):
    rho, _, p0 = state
    sound = math.sqrt(GAMMA*p0/rho)
    if pressure > p0:
        a = 2.0/((GAMMA + 1.0)*rho)
        b = p0*(GAMMA - 1.0)/(GAMMA + 1.0)
        root = math.sqrt(a/(pressure + b))
        return (pressure - p0)*root, root*(1.0 - 0.5*(pressure - p0)/(pressure + b))
    exponent = (GAMMA - 1.0)/(2.0*GAMMA)
    ratio = pressure/p0
    return 2.0*sound/(GAMMA - 1.0)*(ratio**exponent - 1.0), ratio**(-(GAMMA + 1.0)/(2.0*GAMMA))/(rho*sound)


def sod_star():
    p = 0.5*(LEFT[2] + RIGHT[2])
    for _ in range(80):
        fl, dl = pressure_function(p, LEFT)
        fr, dr = pressure_function(p, RIGHT)
        updated = max(p - (fl + fr)/(dl + dr), 1e-12)
        if abs(updated - p) <= 1e-12*max(updated, p):
            p = updated
            break
        p = updated
    fl, _ = pressure_function(p, LEFT)
    fr, _ = pressure_function(p, RIGHT)
    return p, 0.5*(fr - fl)


P_STAR, U_STAR = sod_star()


def sod_exact(x, time_value):
    xi = (x - 0.5)/time_value
    rho_l, u_l, p_l = LEFT
    rho_r, u_r, p_r = RIGHT
    a_l = math.sqrt(GAMMA*p_l/rho_l)
    a_r = math.sqrt(GAMMA*p_r/rho_r)
    if xi <= U_STAR:
        a_star = a_l*(P_STAR/p_l)**((GAMMA - 1)/(2*GAMMA))
        head = u_l - a_l
        tail = U_STAR - a_star
        if xi <= head:
            return LEFT
        if xi >= tail:
            return rho_l*(P_STAR/p_l)**(1/GAMMA), U_STAR, P_STAR
        velocity = 2*(a_l + 0.5*(GAMMA - 1)*u_l + xi)/(GAMMA + 1)
        sound = 2*(a_l + 0.5*(GAMMA - 1)*(u_l - xi))/(GAMMA + 1)
        return rho_l*(sound/a_l)**(2/(GAMMA - 1)), velocity, p_l*(sound/a_l)**(2*GAMMA/(GAMMA - 1))
    shock = u_r + a_r*math.sqrt((GAMMA + 1)*P_STAR/(2*GAMMA*p_r) + (GAMMA - 1)/(2*GAMMA))
    if xi >= shock:
        return RIGHT
    ratio = P_STAR/p_r
    rho = rho_r*((ratio + (GAMMA - 1)/(GAMMA + 1))/((GAMMA - 1)*ratio/(GAMMA + 1) + 1))
    return rho, U_STAR, P_STAR


def write_sod():
    case = ROOT/"sodShockTube"
    time_value, directory = latest(case)
    n = 200
    x = (np.arange(n) + 0.5)/n
    rho = scalar_field(directory/"rho", n)
    velocity = vector_field(directory/"U", n)[:, 0]
    pressure = scalar_field(directory/"p", n)
    exact = np.asarray([sod_exact(value, time_value) for value in x])
    with (OUT/"sod_shock_tube.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x_m", "rho_exact_kg_m3", "rho_GPU_GKS_GKP_kg_m3", "Ux_exact_m_s", "Ux_GPU_GKS_GKP_m_s", "p_exact_Pa", "p_GPU_GKS_GKP_Pa"))
        writer.writerows(zip(x, exact[:, 0], rho, exact[:, 1], velocity, exact[:, 2], pressure))


def couette_exact(y, time_value):
    value = y
    for mode in range(1, 401):
        value += 2*((-1.0)**mode)*math.sin(mode*math.pi*y)*math.exp(-(mode*math.pi)**2*0.1*time_value)/(mode*math.pi)
    return value


def write_couette():
    case = ROOT/"planarCouette"
    time_value, directory = latest(case)
    n = 64
    y = (np.arange(n) + 0.5)/n
    numerical = vector_field(directory/"U", n)[:, 0]
    exact = np.asarray([couette_exact(value, time_value) for value in y])
    with (OUT/"planar_couette.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("y_m", "Ux_analytic_m_s", "Ux_GPU_GKS_GKP_m_s"))
        writer.writerows(zip(y, exact, numerical))


def dusty_exact(time):
    rho_g, rho_p, rho_s, diameter, viscosity = 1.0, 0.1, 1000.0, 1e-4, 0.01
    exponent = 0.687
    stokes = 18*viscosity/(rho_s*diameter**2)
    factor = 0.15*(rho_g*diameter/viscosity)**exponent
    coupled = (1 + rho_p/rho_g)*stokes
    relative = np.exp(-coupled*time)/(1 + factor*(1 - np.exp(-exponent*coupled*time)))**(1/exponent)
    mixture = rho_g/(rho_g + rho_p)
    return mixture + rho_p/(rho_g + rho_p)*relative, mixture - rho_g/(rho_g + rho_p)*relative


def write_dusty():
    case = ROOT/"dustyBox"
    rows = []
    for time_value, directory in sorted((float(p.name), p) for p in case.iterdir() if p.is_dir() and re.fullmatch(NUMBER, p.name)):
        if not (directory/"U").is_file() or not (directory/"Us").is_file():
            continue
        gas = np.mean(vector_field(directory/"U", 100)[20:80, 0])
        particle = np.mean(vector_field(directory/"Us", 100)[20:80, 0])
        exact_gas, exact_particle = dusty_exact(time_value)
        rows.append((time_value, exact_gas, gas, exact_particle, particle))
    with (OUT/"dusty_box.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "Ug_analytic_m_s", "Ug_GPU_GKS_GKP_m_s", "Up_analytic_m_s", "Up_GPU_GKS_GKP_m_s"))
        writer.writerows(rows)


def particle_temperature_by_cell(path, n_cells):
    sums = np.zeros(n_cells)
    masses = np.zeros(n_cells)
    with path.open("rb") as stream:
        header = stream.readline().decode("ascii").split()
        if header[0] != "UGKP_PARTICLES_SCHEMA1_BIN":
            raise RuntimeError(header[0])
        total = int(header[1])
        consumed = 0
        while consumed < total:
            count = struct.unpack("<I", stream.read(4))[0]
            arrays = [np.frombuffer(stream.read(8*count), dtype="<f8") for _ in range(10)]
            cell = np.frombuffer(stream.read(4*count), dtype="<i4")
            status = np.frombuffer(stream.read(4*count), dtype="<i4")
            stream.read(8*count)
            stream.read(8*count)
            temperature = arrays[6]
            mass = arrays[9]
            valid = (status == 1) & (cell >= 0) & (cell < n_cells)
            np.add.at(sums, cell[valid], mass[valid]*temperature[valid])
            np.add.at(masses, cell[valid], mass[valid])
            consumed += count
    return np.divide(sums, masses, out=np.full(n_cells, np.nan), where=masses > 0)


def write_wind():
    case = ROOT/"windSandShockTube"
    directory = case/"0.2"
    n = 200
    x = (np.arange(n) + 0.5)/n
    source = OUT/"wind_sand_shock_tube_reference.csv"
    published = np.genfromtxt(source, delimiter=",", names=True)
    epsilon_s = scalar_field(directory/"epsilonS", n)
    fields = {
        "rho_g": (1.0 - epsilon_s)*scalar_field(directory/"rho", n),
        "rho_p": epsilon_s*1000.0,
        "u_g": vector_field(directory/"U", n)[:, 0],
        "u_p": vector_field(directory/"Us", n)[:, 0],
        "p_g": scalar_field(directory/"p", n),
        "T_p": particle_temperature_by_cell(directory/"gpuResidentStrictParticles.dat", n),
    }
    references = {name: np.interp(x, published["x"], published[name]) for name in fields}
    with (OUT/"wind_sand_shock_tube.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        names = list(fields)
        writer.writerow(["x_m"] + [f"{name}_published" for name in names] + [f"{name}_GPU_GKS_GKP" for name in names])
        for index, value in enumerate(x):
            writer.writerow([value] + [references[name][index] for name in names] + [fields[name][index] for name in names])


write_sod()
write_couette()
write_dusty()
write_wind()
