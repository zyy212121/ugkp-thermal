"""Read committed native gas-budget segments without inventing missing history."""
import argparse
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import sys

NUMBER=(int,Decimal)
IDENTITY=('species','elements','species_order_hash','thermo_hash','mechanism_hash','element_mass_factors_mol_per_kg')


def _vector(row,key,count):
    value=row.get(key)
    if not isinstance(value,list) or len(value)!=count or any(type(x) not in NUMBER or not math.isfinite(x) for x in value):
        raise ValueError('invalid accepted budget '+key)
    return value


def _close(a,b):
    a,b=Decimal(a),Decimal(b)
    return abs(a-b)<=Decimal.from_float(64*sys.float_info.epsilon)*max(abs(a),abs(b),Decimal('1e-30'))


def _same_time(a,b):
    # Producer emits round-trip binary64 times. No inventory-relative tolerance
    # can establish a missing physical interval, however small its duration.
    return float(a)==float(b)


def _read(path):
    text=Path(path).read_text()
    if not text.endswith('\n'):raise ValueError('incomplete budget record')
    lines=text.splitlines();rows=[json.loads(line,parse_float=Decimal) for line in lines]
    if (len(rows)<2 or any(not isinstance(row,dict) or type(row.get('schema')) is not int or row['schema']!=1 for row in rows)
            or rows[0].get('record')!='NEW_SEGMENT'):
        raise ValueError('missing completed gas budget segment')
    h=rows[0];species=h.get('species');elements=h.get('elements')
    if (not isinstance(species,list) or not species
            or any(not isinstance(name,str) or not name for name in species) or len(set(species))!=len(species)
            or not isinstance(elements,list) or any(not isinstance(name,str) or not name for name in elements)
            or len(set(elements))!=len(elements)):
        raise ValueError('invalid budget species/element identity')
    ns=len(species);ne=len(elements)
    for key in ('species_order_hash','thermo_hash','mechanism_hash'):
        if not isinstance(h.get(key),str) or not h[key].isdigit():raise ValueError('missing budget model hash')
    matrix=_vector(h,'element_mass_factors_mol_per_kg',ns*ne)
    if any(value<0 for value in matrix):raise ValueError('negative element composition')
    initial=_vector(h,'initial_inventory',ns+2)
    if initial[0]<=0 or any(value<0 for value in initial[2:]):raise ValueError('invalid initial inventory')
    time=h.get('accepted_initial_time')
    if type(time) not in NUMBER or not math.isfinite(time) or time<0:raise ValueError('invalid segment start')
    for sequence,row in enumerate(rows[1:],1):
        end=row.get('accepted_time');start=row.get('interval_start')
        if (row.get('record')!='ACCEPTED_INTERVAL' or type(row.get('sequence')) is not int or row['sequence']!=sequence
                or type(end) not in NUMBER or not math.isfinite(end) or end<=time
                or type(start) not in NUMBER or not math.isfinite(start) or not _same_time(start,time)):
            raise ValueError('missing or nonmonotone accepted budget interval')
        inventory=_vector(row,'inventory',ns+2)
        if inventory[0]<=0 or any(value<0 for value in inventory[2:]):raise ValueError('invalid accepted inventory')
        _vector(row,'cumulative_boundary_transport',ns+2);_vector(row,'cumulative_chemistry_species',ns)
        time=end
    return h,rows[-1],len(rows)-1,lines


def read_segments(paths,expected_end=None):
    # Form balances before rounding the producer's long-double decimal totals.
    with localcontext() as context:
        context.prec=80
        result=_read_segments(paths,expected_end)
    def public(value):
        if isinstance(value,Decimal):return float(value)
        if isinstance(value,list):return [public(item) for item in value]
        if isinstance(value,dict):return {key:public(item) for key,item in value.items()}
        return value
    return public(result)


def _read_segments(paths,expected_end):
    if not paths:raise ValueError('no gas budget segments')
    pieces=[_read(path) for path in paths];first=pieces[0][0];ns=len(first['species']);ne=len(first['elements'])
    matrix=first['element_mass_factors_mol_per_kg']
    def elements(values):
        return [sum((Decimal(values[k])*Decimal(matrix[k*ne+e]) for k in range(ns)),Decimal(0)) for e in range(ne)]
    def balances(initial,final,boundary,chemical):
        changes=[Decimal(final[k])-Decimal(initial[k])-Decimal(boundary[k]) for k in range(ns+2)]
        return {'constraint_or_roundoff_residual':[value-Decimal(chemical[k-2]) if k>=2 else value for k,value in enumerate(changes)],
                'element_conservation_residual_mol':elements(changes[2:]),'chemistry_element_drift_mol':elements(chemical)}
    segments=[];jumps=[]
    for index,(h,endpoint,count,lines) in enumerate(pieces):
        segment={'file':str(Path(paths[index]).resolve()),'accepted_initial_time':h['accepted_initial_time'],
                 'accepted_final_time':endpoint['accepted_time'],'accepted_intervals':count,
                 'initial_inventory':h['initial_inventory'],'final_inventory':endpoint['inventory'],
                 'cumulative_boundary_transport':endpoint['cumulative_boundary_transport'],
                 'cumulative_chemistry_species':endpoint['cumulative_chemistry_species'],'raw_records':lines}
        segment.update(balances(h['initial_inventory'],endpoint['inventory'],endpoint['cumulative_boundary_transport'],endpoint['cumulative_chemistry_species']))
        segments.append(segment)
        if index:
            prior,previous,_,_=pieces[index-1]
            if any(h[key]!=prior[key] for key in IDENTITY):raise ValueError('restart budget identity mismatch')
            time_matches=_same_time(h['accepted_initial_time'],previous['accepted_time'])
            inventory_jump=[Decimal(a)-Decimal(b) for a,b in zip(h['initial_inventory'],previous['inventory'])]
            if not time_matches or any(inventory_jump):
                jumps.append({'next_segment':index+1,'time_jump':Decimal(h['accepted_initial_time'])-Decimal(previous['accepted_time']),
                              'inventory_jump':inventory_jump,'status':'UNVERIFIED_RESTART_DISCONTINUITY'})
    last=pieces[-1][1];final_time=last['accepted_time']
    if expected_end is not None and (not math.isfinite(expected_end) or not _close(final_time,expected_end)):
        raise ValueError('accepted budget final endpoint mismatch')
    boundary=[sum((Decimal(piece[1]['cumulative_boundary_transport'][k]) for piece in pieces),Decimal(0)) for k in range(ns+2)]
    chemical=[sum((Decimal(piece[1]['cumulative_chemistry_species'][k]) for piece in pieces),Decimal(0)) for k in range(ns)]
    continuity=('UNVERIFIED_SEGMENT_CHAIN' if jumps else 'VERIFIED_SEGMENT_CHAIN' if len(pieces)>1
                else 'SINGLE_SEGMENT_FROM_ZERO' if first['accepted_initial_time']==0 else 'RESTART_SEGMENT_ONLY')
    result={'status':'UNVERIFIED_DISCONTINUOUS_SEGMENTS' if jumps else 'AVAILABLE','schema':1,'continuity':continuity,
        'complete_from_time_zero':not jumps and first['accepted_initial_time']==0,'segment_count':len(pieces),
        'accepted_intervals':sum(piece[2] for piece in pieces),'accepted_initial_time':first['accepted_initial_time'],'accepted_final_time':final_time,
        'identity':{key:first[key] for key in IDENTITY},'initial_inventory':first['initial_inventory'],'final_inventory':last['inventory'],
        'cumulative_boundary_transport':boundary,'cumulative_chemistry_species':chemical,
        'segments':segments,'restart_discontinuities':jumps,
        'element_status':'DECLARED' if ne else 'UNAVAILABLE_NO_ELEMENT_BASIS',
        'mass_elements_physical_energy_acceptance':'NOT_ESTABLISHED',
        'scope':'Applied limited boundary fluxes and actual published chemical species. Energy is formation-inclusive total energy; no chemistry energy source is subtracted. Residuals are reported, never repaired or absorbed into sources.',
        'files':[str(Path(path).resolve()) for path in paths]}
    # A gap/jump cannot cancel a physical residual in another segment. Preserve
    # every segment's result and raw records, without a global conservation claim.
    result.update({key:None for key in ('constraint_or_roundoff_residual','element_conservation_residual_mol','chemistry_element_drift_mol')}
                  if jumps else balances(first['initial_inventory'],last['inventory'],boundary,chemical))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('segments',nargs='+',type=Path)
    parser.add_argument('--expected-end',type=float);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('budget report output must be fresh')
    result=read_segments(args.segments,args.expected_end);args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
