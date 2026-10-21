#!/usr/bin/env python3
"""Compare chronological replay to frozen mathematical final-A.csv.

Physical values use abs(gpu-cpu)/max(1,abs(cpu)), threshold 1e-5,
matching the native directional driver's output criterion. Residual acceptance
is independently checked against 1.01e-8; moments retain the profile's 1e-9.
The checksum includes all three source components, base means and face fluxes.
Timing and nonlinear iteration counts are observations, never parity criteria.
"""
import argparse,csv,json,math
from pathlib import Path

FIELDS=['rawQ','rawTau','fitQ','fitTau','source0','source1','checksum']
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--actual',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=2000)
    p.add_argument('--shared-cpu',type=Path)
    a=p.parse_args()
    actual=list(csv.DictReader(a.actual.open()))
    frozen=list(csv.DictReader(a.reference.open()))
    required=set(FIELDS+['owner','step','stage','ok','actualOk','fitOk','finite','residual','moment','seconds',
        'wallJ0','wallJ1','matchingJ0','matchingJ1','heatBalance','kBalance','profileJ0','profileJ1'])
    if not actual or not required.issubset(actual[0]):
        raise SystemExit('Current qualification schema required; missing flux checks cannot pass')
    def key(row):return tuple(int(row[x]) for x in ['owner','step','stage'])
    expected={}
    for owner in [0,7782]:
        rows=[row for row in frozen if int(row['owner'])==owner][:a.limit]
        expected.update((key(row),row) for row in rows)
    keys=[key(row) for row in actual]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):
        raise SystemExit('Missing, duplicate or unexpected chronological records')
    if keys!=list(expected):raise SystemExit('Replay changed chronological order')
    stats={field:{'max_abs':0.,'max_normalized_floor_1':0.,'worst_record':None} for field in FIELDS}
    shared={key(row):row for row in csv.DictReader(a.shared_cpu.open())} if a.shared_cpu else {}
    direct_fields=['matchingJ0','matchingJ1','heatBalance','kBalance','profileJ0','profileJ1']
    direct={field:0. for field in direct_fields}
    nonzero_transport={0:0,7782:0};max_species_sum=0.
    failures=[];maxres=maxmoment=seconds=0.
    for row in actual:
        ident=key(row);ref=expected[ident]
        for flag in ['ok','actualOk','fitOk','finite']:
            if int(row[flag])!=1:failures.append([ident,flag])
        residual=float(row['residual']);moment=float(row['moment'])
        if not math.isfinite(residual) or residual>1.01e-8:failures.append([ident,'residual'])
        if not math.isfinite(moment) or moment>1e-9:failures.append([ident,'moment'])
        maxres=max(maxres,residual);maxmoment=max(maxmoment,moment);seconds+=float(row['seconds'])
        for field in FIELDS:
            got,want=float(row[field]),float(ref[field]);diff=abs(got-want);norm=diff/max(1.,abs(want))
            if not math.isfinite(got) or not math.isfinite(want) or norm>1e-5:failures.append([ident,field])
            z=stats[field];z['max_abs']=max(z['max_abs'],diff)
            if norm>z['max_normalized_floor_1']:z['max_normalized_floor_1']=norm;z['worst_record']=ident
        for field in ['wallJ0','wallJ1']:
            if field in row and float(row[field])!=0.:failures.append([ident,field])
        for field in direct_fields:
            if field in row:
                value=float(row[field])
                if not math.isfinite(value):failures.append([ident,field+' nonfinite'])
                if shared:
                    want=float(shared[ident][field]);norm=abs(value-want)/max(1.,abs(want))
                    if not math.isfinite(want) or norm>1e-5:failures.append([ident,field+' shared parity'])
                    direct[field]=max(direct[field],norm)
        if 'profileJ0' in row:
            nonzero_transport[ident[0]]+=abs(float(row['profileJ0']))>1e-12
            defect=abs(float(row['profileJ0'])+float(row['profileJ1']))
            max_species_sum=max(max_species_sum,defect)
            if not math.isfinite(defect) or defect>1e-12:failures.append([ident,'profile species sum'])
    if 'profileJ0' in actual[0]:
        for owner,count in nonzero_transport.items():
            if count==0:failures.append([owner,'no resolved nonzero reconstructed interface species transport'])
    result={'passed':not failures,'records':len(actual),'nodes':24,'physical_residual_limit':1.01e-8,'moment_limit':1e-9,
        'max_residual':maxres,'max_moment':maxmoment,'physical_comparison':stats,'failures':failures,
        'operator_and_launch_seconds':seconds,'nonzero_transport_records':nonzero_transport,'max_profile_species_sum':max_species_sum,'shared_cpu_output_error_floor_1':direct,'timing_scope':'single chronological wall launch, not a GPU throughput speedup'}
    for field in ['wallJ0','wallJ1','heatBalance','kBalance','profileJ0','profileJ1']:
        if field in actual[0]:result['max_abs_'+field]=max(abs(float(row[field])) for row in actual)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
