#!/usr/bin/env python3
"""Read one existing macro run. Never starts a solver or changes case data.

Spatial distributions and interval-weighted observations are computed within
this run. Inventory changes are not labelled global conservation residuals:
the open-domain boundary integrals are not present in these output formats.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re
import struct
import numpy as np


def clean(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)


def dictionary(path):
    result = {}
    for key, value in re.findall(r'(?m)^\s*(\w+)\s+([^;{}]+);', clean(path.read_text())):
        value = value.strip().strip('"')
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def raw_list(path):
    match = re.search(r'\n\s*(\d+)\s*\n\s*\((.*)\)\s*$', clean(path.read_text()), re.S)
    if not match:
        raise ValueError('unsupported ASCII list: ' + str(path))
    return int(match[1]), match[2]


def mesh(directory):
    n, text = raw_list(directory/'points')
    points = np.fromstring(text.replace('(', ' ').replace(')', ' '), sep=' ').reshape(n, 3)
    n, text = raw_list(directory/'faces')
    faces = [np.fromstring(value, sep=' ', dtype=int) for value in re.findall(r'\d+\s*\(([^)]*)\)', text)]
    if len(faces) != n:
        raise ValueError('face count mismatch')
    n, text = raw_list(directory/'owner')
    owner = np.fromstring(text, sep=' ', dtype=int)
    if len(owner) != n or n != len(faces):
        raise ValueError('owner count mismatch')
    n, text = raw_list(directory/'neighbour')
    neighbour = np.fromstring(text, sep=' ', dtype=int)
    if len(neighbour) != n:
        raise ValueError('neighbour count mismatch')
    cells = int(max(owner.max(), neighbour.max(initial=-1))) + 1
    volume = np.zeros(cells)
    moment = np.zeros((cells, 3))
    area = np.zeros(len(faces))
    centers = np.zeros((len(faces), 3))
    for face, ids in enumerate(faces):
        vertices = points[ids]
        centers[face] = vertices.mean(axis=0)
        area[face] = np.linalg.norm(np.cross(vertices, np.roll(vertices, -1, axis=0)).sum(axis=0))/2
        for k in range(1, len(vertices)-1):
            signed = np.dot(vertices[0], np.cross(vertices[k], vertices[k+1]))/6
            centroid = (vertices[0]+vertices[k]+vertices[k+1])/4
            volume[owner[face]] += signed
            moment[owner[face]] += signed*centroid
            if face < len(neighbour):
                volume[neighbour[face]] -= signed
                moment[neighbour[face]] -= signed*centroid
    if not np.isfinite(volume).all() or (volume <= 0).any():
        raise ValueError('mesh has invalid signed volume')
    patches = {}
    for name, body in re.findall(r'(\w+)\s*\{(.*?)\}', clean((directory/'boundary').read_text()), re.S):
        count = re.search(r'nFaces\s+(\d+)', body)
        start = re.search(r'startFace\s+(\d+)', body)
        if count and start:
            patches[name] = (int(start[1]), int(count[1]))
    return dict(volume=volume, area=area, centers=moment/volume[:, None],
                face_centers=centers, patches=patches)


def field(path, count, patch=None):
    text = clean(path.read_text())
    vector = 'volVectorField' in text
    if patch:
        match = re.search(r'\b'+re.escape(patch)+r'\s*\{(.*?)\}', text, re.S)
        if not match:
            raise ValueError('missing patch '+patch)
        text = match[1]
    key = 'value' if patch else 'internalField'
    match = re.search(r'\b'+key+r'\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;', text, re.S)
    if match:
        values = np.fromstring(match[2].replace('(', ' ').replace(')', ' '), sep=' ')
        if int(match[1]) != count:
            raise ValueError('field count mismatch: '+str(path))
        values = values.reshape(count, 3) if vector else values
    else:
        match = re.search(r'\b'+key+r'\s+uniform\s+(\([^)]*\)|[^;]+);', text)
        if not match:
            raise ValueError('missing field data: '+str(path))
        value = np.fromstring(match[1].replace('(', ' ').replace(')', ' '), sep=' ')
        values = np.tile(value, (count, 1)) if vector else np.full(count, value[0])
    if len(values) != count or not np.isfinite(values).all():
        raise ValueError('nonfinite or malformed field: '+str(path))
    return values


def stats(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('invalid statistical input')
    if not len(values) or weights.sum() == 0:
        return dict(count=len(values), weight=0., mean=None)
    values = np.linalg.norm(values, axis=1) if values.ndim == 2 else values
    order = np.argsort(values)
    positive = weights[order] > 0
    x, w = values[order][positive], weights[order][positive]
    quantiles = np.interp(np.array([0, .1, .5, .9, 1])*w.sum(), np.cumsum(w), x)
    mean = float(np.dot(values, weights)/weights.sum())
    return dict(count=len(values), weight=float(weights.sum()), mean=mean,
                std=float(np.sqrt(np.dot((values-mean)**2, weights)/weights.sum())),
                minimum=float(values.min()), maximum=float(values.max()),
                weighted_quantiles_0_10_50_90_100=quantiles.tolist())


def spatial_bins(values, weights, coordinates, edges):
    ids = np.clip(np.searchsorted(edges, coordinates, side='right')-1, 0, len(edges)-2)
    return [dict(lower=float(edges[i]), upper=float(edges[i+1]),
                 statistics=stats(values[ids == i], weights[ids == i])) for i in range(len(edges)-1)]


def particles(path):

    with path.open('rb') as stream:
        marker, total, bound = stream.readline(256).decode('ascii').split()
        total, bound = int(total), int(bound)
        if marker not in ('UGKP_CHT_PARTICLES_SCHEMA9_FLOAT_COLD_PROFILES_BIN',
                          'UGKP_CHT_PARTICLES_SCHEMA11_NATIVE64_RAW_COLD_BIN',
                          'UGKP_CHT_PARTICLES_SCHEMA12_NATIVE32_RAW_COLD_BIN'):
            raise ValueError('unsupported particle schema: '+marker)
        specs = [(name, '<f8', 1) for name in 'px py pz pux puy puz pT pTheta pd pm'.split()]
        specs += [('pCellId','<i4',1),('pStatus','<i4',1),('pRng','<u8',1),
                  ('pOrigId','<u8',1),('pStuck','u1',1),('pStuckFaceId','<i4',1)]
        specs += [(name, '<f8', 1) for name in 'pDepositionArea pContactDuration pContactMaximumArea pContactPeakFraction'.split()]
        specs += [('pColdNodeSpecificEnthalpy','<f8' if 'NATIVE64' in marker else '<f4',8),
                  ('pColdRingSolidMass','<f8',8),('pColdFrozenArea','<f8',1),('pColdContactAge','<f8',1),
                  ('pCold2DNodeSpecificEnthalpy','<f4',64),('pCold2DRingContactAge','<f8',8),
                  ('pCold2DFrozenArea','<f8',1)]
        specs += [(name, '<f8', 1) for name in 'puxOld puyOld puzOld'.split()]
        if 'NATIVE32' not in marker:
            specs += [('pThermalEnthalpyResidual','<f8',1)]
        arrays = {name: [] for name, _, _ in specs}
        nonfinite_auxiliary = {}
        done = 0
        while done < total:
            raw = stream.read(4)
            if len(raw) != 4:
                raise ValueError('truncated particle count')
            count = struct.unpack('<I', raw)[0]
            if count <= 0 or count > bound or count > total-done:
                raise ValueError('invalid particle count')
            for name, dtype, width in specs:
                values = np.fromfile(stream, dtype=dtype, count=count*width)
                if len(values) != count*width:
                    raise ValueError('truncated particle field: '+name)
                nonfinite = int(np.count_nonzero(~np.isfinite(values)))
                if nonfinite:
                    if name in 'px py pz pux puy puz pT pTheta pd pm'.split():
                        raise ValueError('nonfinite primary particle field: '+name)



                    nonfinite_auxiliary[name] = nonfinite_auxiliary.get(name,0)+nonfinite
                if width == 1:
                    arrays[name].append(values)
            done += count
        if stream.read(1):
            raise ValueError('trailing particle restart data')
    arrays = {name: np.concatenate(parts) for name, parts in arrays.items() if parts}
    if not total:
        return dict(schema=marker, count=0, mass=0.)
    mass = arrays['pm']
    velocity = np.column_stack([arrays[name] for name in ('pux','puy','puz')])
    positions = np.column_stack([arrays[name] for name in ('px','py','pz')])
    axis = int(np.argmax(np.ptp(positions, axis=0)))
    extent = positions[:,axis]
    edges = np.linspace(extent.min(), extent.max()+np.finfo(float).eps, 17)
    status, counts = np.unique(arrays['pStatus'], return_counts=True)
    return dict(schema=marker, count=total, mass=float(mass.sum()),
                nonfinite_auxiliary_fields=nonfinite_auxiliary,
                status_counts={str(s):int(n) for s,n in zip(status,counts)},
                stuck_count=int(np.count_nonzero(arrays['pStuck'])),
                momentum=(mass[:,None]*velocity).sum(axis=0).tolist(),
                kinetic_energy=float(.5*np.dot(mass, (velocity*velocity).sum(axis=1))),
                temperature=stats(arrays['pT'], mass), diameter=stats(arrays['pd'], mass),
                spatial_axis=axis, temperature_bins=spatial_bins(arrays['pT'],mass,extent,edges))


def log_records(path):
    text = path.read_text(errors='replace') if path else ''
    result = dict(warnings=[line for line in text.splitlines() if any(key in line.lower()
                  for key in ('warning','budget exhausted','fatal','chmt failed'))][:100])
    for tag in ('THERMAL_EVENT','THERMAL_RUN_COMPLETE','SOLID_BALANCE','CHMT_REPRO'):
        records = []
        for line in text.splitlines():

            line = line.strip().strip('"')
            if not line.startswith(tag+' '):
                continue
            record = {}
            for key, value in re.findall(r'(\w+)=([^ ]+)', line):
                try:
                    record[key] = float(value)
                except ValueError:
                    record[key] = value
            records.append(record)
        result[tag] = records
    return result


def analyze_cht(case, logs):
    meshes = {region:mesh(case/'constant'/region/'polyMesh') for region in ('fluid','graphite')}
    snapshots = []
    directories = sorted((p for p in case.iterdir() if p.is_dir() and re.fullmatch(r'[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?',p.name)),key=lambda p:float(p.name))
    wall_energy = {}
    prior_radiation = float(directories[0].name)
    for directory in directories:
        event = dict(time=float(directory.name), fields={}, inventory={}, wall={})
        path = directory/'thermalExchangeState'
        if path.exists():
            event['state'] = dictionary(path)
        for region, geometry in meshes.items():
            volume = geometry['volume']
            axis = int(np.argmax(np.ptp(geometry['centers'],axis=0)))
            coordinate = geometry['centers'][:,axis]
            edges = np.linspace(coordinate.min(),coordinate.max(),17)
            names = 'T' if region == 'graphite' else 'rho rhoE rhoU rhoEs rhoHp epsilonS p T Tp theta'
            values = {}
            for name in names.split():
                path = directory/region/name
                if not path.exists():
                    continue
                data = field(path,len(volume));values[name] = data
                event['fields'][region+'/'+name] = stats(data,volume)
                if name in ('T','Tp','epsilonS'):
                    event['fields'][region+'/'+name]['spatial_axis'] = axis
                    event['fields'][region+'/'+name]['bins'] = spatial_bins(data,volume,coordinate,edges)
            if region == 'fluid':
                epsilon = 1-values.get('epsilonS',np.zeros(len(volume)))
                for name, key in (('rho','gas_mass'),('rhoE','gas_energy'),('rhoU','gas_momentum')):
                    if name in values:
                        event['inventory'][key] = np.asarray(np.tensordot(epsilon*volume,values[name],axes=1)).tolist()
                for name in ('rhoEs','rhoHp'):
                    if name in values:
                        event['inventory'][name] = float(np.dot(volume,values[name]))
                patch = 'fluid_to_graphite'
                first, count = geometry['patches'][patch]
                area = geometry['area'][first:first+count]
                state = event.get('state',{})
                heat_dt = state.get('completedSimulationTimeS',event['time'])-state.get('previousExchangeSimulationTimeS',event['time'])
                radiation_time = state.get('completedRadiationSimulationTimeS',prior_radiation)
                for name in 'T gasConvectiveWallHeatFlux particleRadiationWallHeatFlux particleReflectedWallHeatFlux particleStuckWallHeatFlux'.split():
                    path = directory/region/name
                    if not path.exists():
                        continue
                    data = field(path,count,patch)
                    entry = stats(data,area)
                    if 'HeatFlux' in name:
                        entry['power'] = float(np.dot(data,area))
                        interval = max(0.,radiation_time-prior_radiation) if 'Radiation' in name else heat_dt
                        entry['interval'] = interval
                        entry['energy'] = entry['power']*interval
                        if directory != directories[0]:
                            wall_energy[name] = wall_energy.get(name,0.)+entry['energy']
                    event['wall'][name] = entry
                prior_radiation = radiation_time
        path = directory/'gpuResidentStrictParticles.dat'
        if path.exists():
            event['particles'] = particles(path)
        snapshots.append(event)
    completed = logs['THERMAL_RUN_COMPLETE']
    return dict(snapshots=snapshots, cumulative_wall_energy=wall_energy,
                heat_events=int(completed[-1]['heatEvents']) if completed else None,
                radiation_events=int(completed[-1]['radiationEvents']) if completed else None,
                conservation=dict(solid_reports=logs['SOLID_BALANCE'],
                  scope='Solid interval energy residuals and consumed exchange flags; gas inventory changes include unrecorded open-boundary flow.'))


def csv_table(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        rows = list(reader)
    columns = {}
    for name in names:
        try:
            columns[name] = np.array([float(row[name]) for row in rows])
        except ValueError:
            continue
        if not np.isfinite(columns[name]).all():
            raise ValueError('nonfinite CSV column: '+str(path)+':'+name)
    return rows, columns


def analyze_chmt(case, logs, output=None):
    output = output or case/'checkpointOutput'
    snapshots = []
    for path in sorted(output.glob('accepted-*.csv'),key=lambda p:int(p.stem.split('-')[1])):
        rows, _ = csv_table(path)
        if len(rows) != 1:
            raise ValueError('invalid accepted summary')
        suffix = path.stem.split('-')[1]
        event = dict(time=float(rows[0]['time']), step=int(rows[0]['accepted_steps']), summary=rows[0], fields={}, inventory={})
        for kind in ('gas','material','particles'):
            path = output/(kind+'-'+suffix+'.csv')
            if not path.exists():
                continue
            _, columns = csv_table(path)
            weight = columns.get('volume',columns.get('mass',np.array([])))
            event['fields'][kind] = {name:stats(values,weight) for name,values in columns.items()
                if name in ('temperature','pressure','porosity','diameter','theta')}
            event['inventory'][kind] = {name:float(values.sum()) for name,values in columns.items()
                if name.startswith(('mass','momentum')) or name in ('volume','total_energy','thermal_energy')}
            if kind == 'particles':
                event['inventory'][kind]['count'] = len(weight)
        path = output/('exchange-'+suffix+'.csv')
        if path.exists():
            _, columns = csv_table(path)
            required = np.array([3,5,6,3,9,10],dtype=int)
            kinds = columns['kind'].astype(int)
            if (kinds < 0).any() or (kinds >= len(required)).any():
                raise ValueError('invalid exchange kind')
            components = sum((columns[name] for name in ('conductive','advective','pressure_work','viscous_work','radiation')),np.zeros(len(kinds)))
            remainder = columns['energy']-components
            scale = np.maximum(np.abs(columns['energy']),sum((np.abs(columns[name]) for name in ('conductive','advective','pressure_work','viscous_work','radiation')),np.zeros(len(kinds))))
            event['exchange'] = dict(packet_count=len(kinds),
                unconsumed=int(np.count_nonzero(columns['consumer_mask'].astype(int)!=required[kinds])),
                maximum_energy_decomposition_residual=float(np.abs(remainder).max(initial=0.)),
                relative_energy_decomposition_residual=float(np.abs(remainder).sum()/max(float(scale.sum()),1e-300)),
                transferred_mass=float(columns['mass'].sum()), transferred_energy=float(columns['energy'].sum()),
                radiative_energy=float(columns['radiation'].sum()))
        snapshots.append(event)
    records = logs['CHMT_REPRO']
    radiation_times = [record['radiationTime'] for record in records if 'radiationTime' in record]
    initial_time = snapshots[0]['time'] if snapshots else 0.
    radiation_events = len(set(t for t in radiation_times if t > initial_time)) if radiation_times else None
    return dict(snapshots=snapshots, heat_events=len(records), radiation_events=radiation_events,
                conservation=dict(exchange_reports=[event.get('exchange') for event in snapshots],
                  scope='Exchange consumption and energy decomposition. Legacy audit fields marked NOT_COLLECTED are unavailable, never zero closure residuals.'))


def temporal_observations(snapshots):
    """Right-endpoint accepted-state quadrature, not an ensemble average."""
    result = {}
    for before, after in zip(snapshots,snapshots[1:]):
        interval = after['time']-before['time']
        if interval <= 0:
            raise ValueError('accepted output times are not increasing')
        def visit(prefix, value):
            if not isinstance(value,dict):
                return
            if isinstance(value.get('mean'),(int,float)):
                entry = result.setdefault(prefix,dict(duration=0.,integral=0.))
                entry['duration'] += interval
                entry['integral'] += interval*value['mean']
            else:
                for key, child in value.items():
                    visit(prefix+'/'+key,child)
        visit('fields',after['fields'])
    for entry in result.values():
        entry['interval_weighted_mean'] = entry['integral']/entry['duration']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('CHT','CHMT'),required=True)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--log',type=Path)
    parser.add_argument('--data-directory',type=Path,help='Existing CHMT CSV directory; defaults to case/checkpointOutput')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    logs = log_records(args.log)
    result = analyze_cht(args.case,logs) if args.kind == 'CHT' else analyze_chmt(args.case,logs,args.data_directory)
    result.update(case=str(args.case.resolve()),kind=args.kind,logs=logs,
        temporal_statistics=temporal_observations(result['snapshots']),
        global_conservation_closure='unavailable: full open-boundary time integrals are not exported',
        comparison='single-run analysis only; acceptance requires manual physical review')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),snapshots=len(result['snapshots']),
                         heat_events=result['heat_events'],radiation_events=result['radiation_events'])))


if __name__ == '__main__':
    main()
