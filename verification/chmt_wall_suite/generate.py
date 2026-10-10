#!/usr/bin/env python3
"""Generate fresh native solver cases; never execute, overwrite, or fake results."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from catalog import cases
from materials import get as material_card
from foam import ROOT,header,module,copy_inputs,replace_entry,fingerprint,block_mesh,write_field

def _json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
def _text(path,text):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(text)

def wall_config(family,laminar=False,nodes=24):
    if int(nodes)!=nodes or not 4<=nodes<=128:raise ValueError("production wall nodes must be between 4 and 128")
    if family not in ('lowRe','wallFunction','boundaryLayer_reactingSst','boundaryLayer_constantTransport','boundaryLayer_finiteRate'):
        raise ValueError('unsupported wall family '+family)
    inner='wallTreatment '+('boundaryLayer' if family.startswith('boundaryLayer') else family)+';\n'
    if family.startswith('boundaryLayer'):
        model=family.split('_',1)[1]
        if model=='constantTransport' and not laminar:raise ValueError('constantTransport is a laminar verification limit')
        inner+='boundaryLayer { model '+model+'; nodes '+str(nodes)+'; maxIterations 60; relativeTolerance 1e-8; absoluteTolerance 1e-10; stretch 2; }\n'
    if laminar:return 'simulationType laminar;\n'+inner
    return 'simulationType RAS;\nRAS { model kOmegaSST; turbulence on; printCoeffs on; kOmegaSSTCoeffs { '+inner+' } }\n'

def _gas_template(dest,cells=64):
    maker=module('test/gasUGKP/mixtureTransport/make_case.py');maker.create_case(dest,cells=cells)

def _couette(dest,opts,newwall=False):
    copy_inputs(ROOT/'examples/consistency/planarCouette',dest)
    n=int(opts.get('cells',128 if newwall else 64))
    p=dest/'system/blockMeshDict';s=p.read_text().replace('(1 64 1)',f'(1 {n} 1)')
    s=s.replace('xMin { type symmetryPlane;', 'xMin { type cyclic; neighbourPatch xMax; transform translational; separationVector (-0.01 0 0);')
    s=s.replace('xMax { type symmetryPlane;', 'xMax { type cyclic; neighbourPatch xMin; transform translational; separationVector (0.01 0 0);')
    p.write_text(s)
    for p in (dest/'0').iterdir():
        if p.is_file():p.write_text(p.read_text().replace('type symmetryPlane;','type cyclic;'))
    if newwall:
        # Supply real shared mixture metadata, not an inert diagnostic array.
        temp=dest/'_gas_template';_gas_template(temp,4)
        shutil.copyfile(temp/'constant/gasModelProperties',dest/'constant/gasModelProperties')
        shutil.rmtree(temp)
        gas=(dest/'constant/gasModelProperties').read_text().replace('1040 0 0','1004.5 0 0').replace('molarMass 0.028','molarMass 0.0289702530249').replace('0.02 0.02','0 0')
        (dest/'constant/gasModelProperties').write_text(gas)
        patches={p:'type cyclic;' for p in ('xMin','xMax')};patches.update(bottom='type zeroGradient;',top='type zeroGradient;',zEmpty='type empty;')
        for name in ('Y_A','Y_B'):write_field(dest,'',name,'0 0 0 0 0 0 0',.5,patches)
        # New wall model requires explicit fixed temperatures, kept isothermal.
        p=dest/'0/T';s=p.read_text();s=re.sub(r'(bottom|top)\s*\{[^{}]*\}',lambda m:m.group(1)+' { type fixedValue; value uniform 300; }',s);p.write_text(s)
        fp=dest/'constant/fluidProperties'
        fp.write_text(re.sub(r'turbulence\s*\{[^{}]*\}', 'turbulence { '+wall_config('boundaryLayer_constantTransport',True,int(opts.get('wall_nodes',24)))+' }', fp.read_text()))
    return dict(cells=n,end_time=.05,parameters=dict(height=1.,wall_velocity=1.,kinematic_viscosity=.1,velocity_scale=1.,axis=1),metric='couette',mesh_commands=[['blockMesh']],initialization=[])

def _sod(dest,opts):
    copy_inputs(ROOT/'examples/consistency/sodShockTube',dest)
    s=(dest/'system/blockMeshDict').read_text();match=re.search(r'hex\s*\([^)]*\)\s*\((\d+)\s+1\s+1\)',s)
    if not match:raise ValueError('Sod template cell layout changed')
    n=int(opts.get('cells',int(match.group(1))));s=s[:match.start(1)]+str(n)+s[match.end(1):];(dest/'system/blockMeshDict').write_text(s)
    return dict(cells=n,end_time=.0007,parameters=dict(length=1.,interface=.5,gamma=1.4,left=[1.,0.,100000.],right=[.125,0.,10000.]),metric='sod',mesh_commands=[['blockMesh']],initialization=[['setFields']])

def _wave(dest,opts):
    n=int(opts.get('cells',64));maker=module('test/gasUGKP/mixtureTransport/make_case.py')
    maker.create_case(dest,cells=n,diffusivity=float(opts.get('diffusivity',.02)),velocity=float(opts.get('velocity',1.)),end_time=float(opts.get('end_time',.1)))
    p=json.loads((dest/'case_parameters.json').read_text())
    return dict(cells=n,end_time=p['end_time'],parameters=p,metric='wave',mesh_commands=[['blockMesh']],initialization=[])

def _reactor(dest,opts):
    maker=module('test/gasUGKP/chemistryReactor/make_case.py')
    variant=opts.get('refinement','fine');maker.create_case(dest,variant)
    contract=json.loads((dest/'case_contract.json').read_text())
    return dict(cells=contract['cells'],end_time=contract['end_time'],parameters=contract,metric='reactor',mesh_commands=[['blockMesh']],initialization=[])

def _surface_properties(card,active=True):
    s='material { gasContactResistance 0; solidContactResistance 0; enableMelting false; enablePoreOutflow false; evaporation Disabled;\n'
    s+='enableSurfaceReactions '+str(active).lower()+';\n'
    if active:
        r=card['surface_reactions'][0]
        s+='surfaceReactions { conversion { '+''.join(f'{k} {r[k]:.17g}; ' for k in ('A','temperaturePower','activationEnergy'))
        s+=' '.join(k+' ('+' '.join(format(x,'.17g') for x in r[k])+');' for k in ('condensedNu','gasNu','order','gasOrder'))+' } }\n'
    return s+'}\n'

def _material_text(card):
    out='condensedThermo {\n'
    for name,c in zip(card['condensed_names'],card['condensed']):out+=name+' { '+''.join(f'{k} {v:.17g}; ' for k,v in c.items())+'element (35.714285714285715); }\n'
    return out+'}\n'

def _chmt_base(dest,opts,contact=False):
    card=material_card(opts.get('material','controlled_v1'))
    if contact:
        if opts.get('material','controlled_v1')!='controlled_v1':
            raise ValueError('small_cht_contact uses the fixed controlled_v1 fixture; material overrides are not implemented')
        # Reuse the shipped native contact fixture including checker coefficients.
        maker=module('applications/CHMT/verification/coupled_runtime/make_case.py')
        maker.make_case(dest,'thermal',None,None,bool(opts.get('refine',False)))
        spec=json.loads((dest/'verification.json').read_text())
        card['surface_reactions']=[];card['gas_diffusivity_m2_s']=0.;card['gas_viscosity_Pa_s']=0.;card['gas_conductivity_W_m_K']=1.;card['gas_species'][1]['molar_mass']=.032
        card['source']='Fixed synthetic thermal-contact fixture in applications/CHMT/verification/coupled_runtime/make_case.py; not measured material data.'
        card['element_basis']='Not declared: the frozen, nonreacting contact fixture uses elements (); no synthetic atom mapping or elemental-conservation claim is attached.'
        card['pyrolysis_products']='Disabled: no bulk or surface conversion and no emitted gas products. S0/S1 are frozen gas species with molecular masses 0.028/0.032 kg/mol.'
        card['pore_viscosity_Pa_s']=0.;card['ambient_temperature_K']=0.
        card['applicability']='Native constant-property dry two-region contact fixture; no reaction or motion.'
        return dict(cells=8,end_time=spec['end_time'],metric='contact',parameters=spec,material_card=card,mesh_commands=[['blockMesh'],['blockMesh','-region','solid','-dict','system/blockMeshSolidDict']],initialization=[])
    subprocess.run([sys.executable,str(ROOT/'applications/CHMT/verification/coupled_input/make_case.py'),str(dest)],check=True,capture_output=True,text=True)
    p=dest/'constant/chmtProperties';s=p.read_text()
    s=re.sub(r'condensedThermo\s*\{.*?\n\}',_material_text(card).strip(),s,flags=re.S)
    s=re.sub(r'modelFingerprint\s+"[^"]+";','',s)
    s=s.replace('elements ();','elements (X);')
    s=s.replace('materialSource syntheticConstantSolid;','materialSource '+('syntheticControlledV2' if opts.get('material')=='controlled_v2' else 'syntheticControlledV1')+';').replace('mechanismSource noMaterialReaction;','mechanismSource controlledSurfaceConversion;')
    s=s.replace('gasConductivity 1;','gasConductivity '+str(card['gas_conductivity_W_m_K'])+';').replace('gasViscosity 0;','gasViscosity '+str(card['gas_viscosity_Pa_s'])+';')
    s+=_surface_properties(card)+ '\nmeshMotion { policy CoupledRecession; }\n'
    s+='emissivity 0; ambientTemperature 300; permeability 0; poreViscosity '+str(card['pore_viscosity_Pa_s'])+';\n'
    end=float(opts.get('end_time',.01));replace_entry(dest/'system/controlDict','endTime',end)
    s=s.replace('couplingInterval 0.01','couplingInterval '+str(min(end/10,.001)))
    gas=dest/'constant/gasModelProperties';t=gas.read_text().replace('molarMass 0.032','molarMass 0.028').replace('diffusion { model none; }','diffusion { model constant; coefficients (2e-5 2e-5); }');t=t.replace('species (S0 S1);','species (S0 S1); elements (X);').replace('coefficients (1040 0 0);','coefficients (1040 0 0); atoms (1);');gas.write_text(t)
    s+='modelFingerprint "'+hashlib.sha256((s+t).encode()).hexdigest()+'";\n';p.write_text(s)
    # Uniform 700 K controlled reactive slab, zero initial pore mass and dry surface.
    T=700.;c=card['condensed'][0];replace_entry(dest/'0/T','internalField','uniform '+str(T));replace_entry(dest/'0/rho','internalField','uniform '+str(101325/(8.31446261815324/.028*T)))
    replace_entry(dest/'0/solid/rho_C0','internalField','uniform '+str(c['rho']))
    E=c['rho']*(c['e0']+c['cp0']*T+.5*c['cp1']*T*T)
    replace_entry(dest/'0/solid/solidEnergyDensity','internalField','uniform '+str(E))
    return dict(cells=8,end_time=end,metric='chmt',parameters=dict(initial_temperature_K=T,closed=True,surface_area_m2=1.,surface_mass_flux_kg_m2_s=card['surface_reactions'][0]['A']*sum(card['surface_reactions'][0]['gasNu'])),material_card=card,mesh_commands=[['blockMesh'],['blockMesh','-region','solid','-dict','system/blockMeshSolidDict']],initialization=[])

def _flat_gas(dest,opts,patches=None):
    nx=int(opts.get('nx',96));ny=int(opts.get('ny',48));height=float(opts.get('height',.2));grading=float(opts.get('grading',8.))
    if nx<2 or ny<3 or height<=0 or grading<=0:raise ValueError('flat plate requires nx>=2, ny>=3, positive height/grading')
    _gas_template(dest,max(4,nx*ny))
    L=2.;T=300.;p=101325.;W=.028;U=float(opts.get('velocity',70.));R=8.31446261815324/W;rho=p/(R*T)
    (dest/'system/blockMeshDict').write_text(block_mesh(0,L,0,height,.01,nx,ny,grading=grading))
    patch_types={'inlet':'type fixedValue; value uniform VALUE;','outlet':'type zeroGradient;','wall':'type zeroGradient;','top':'type zeroGradient;','sides':'type empty;'}
    fields=[('rho','1 -3 0 0 0 0 0',rho,False),('p','1 -1 -2 0 0 0 0',p,False),('T','0 0 0 1 0 0 0',T,False),('U','0 1 -1 0 0 0 0',f'({U} 0 0)',True),('Y_A','0 0 0 0 0 0 0',.5,False),('Y_B','0 0 0 0 0 0 0',.5,False),('epsilonS','0 0 0 0 0 0 0',0,False),('Us','0 1 -1 0 0 0 0','(0 0 0)',True),('theta','0 2 -2 0 0 0 0',0,False),('rhoE','1 -1 -2 0 0 0 0',rho*((1040-R)*T+.5*U**2),False),('k','0 2 -2 0 0 0 0',1e-4,False),('omega','0 0 -1 0 0 0 0',1.,False)]
    for name,dims,v,vector in fields:
        ps={k:x.replace('VALUE',str(v)) for k,x in patch_types.items()}
        if name=='U':ps['wall']='type fixedValue; value uniform (0 0 0);';ps['top']='type slip;'
        if name=='T':ps['wall']='type fixedValue; value uniform 300;'
        if name=='k':ps['wall']='type fixedValue; value uniform 0;'
        write_field(dest,'',name,dims,v,ps,vector)
    fp=dest/'constant/fluidProperties';s=fp.read_text().replace('mu 0;','mu 1.8e-5;');fp.write_text(s)
    family=opts.get('wall','lowRe')
    # Actual gasUGKP reader loads fluidProperties/turbulence.
    fp.write_text(s.replace('turbulence { simulationType laminar; }','turbulence { '+wall_config(family,nodes=int(opts.get('wall_nodes',24)))+' }'))
    end=float(opts.get('end_time',.2));replace_entry(dest/'system/controlDict','endTime',end);replace_entry(dest/'system/controlDict','writeInterval',end/10)
    return dict(cells=nx*ny,end_time=end,metric='profiles',parameters=dict(nx=nx,ny=ny,length=L,height=height,grading=grading,width=.01,velocity=U,rho=rho,temperature=T,pressure=p),wall_family=family,mesh_commands=[['blockMesh']],initialization=[])

def _flat_chmt(dest,opts):
    # Generate actual conformal adjacent gas/material blocks, not standalone film.
    out=_chmt_base(dest,opts);card=out['material_card'];nx=int(opts.get('nx',96));ny=int(opts.get('ny',48));ns=int(opts.get('solid_ny',24));H=float(opts.get('height',.2));G=float(opts.get('grading',8.));L=2.;W=.01
    if min(nx,ny,ns)<2 or G<=0 or H<=0:raise ValueError('invalid paired grid')
    (dest/'system/blockMeshDict').write_text(block_mesh(0,L,0,H,W,nx,ny,grading=G))
    # Solid top is interface; gas lower wall is interface.
    (dest/'system/blockMeshSolidDict').write_text(block_mesh(0,L,-.05,0,W,nx,ns,('solidInlet','solidOutlet','back','interface','sides')))
    s=(dest/'constant/chmtProperties').read_text().replace('surface { patch right; gasPatch left; }','surface { patch interface; gasPatch wall; }')
    s=re.sub(r'modelFingerprint\s+"[^"]+";','',s)
    family=opts.get('wall','lowRe')
    if family=='wallFunction':raise ValueError('CHMT does not support ordinary wallFunction; choose lowRe or boundaryLayer_reactingSst')
    inner=wall_config(family,nodes=int(opts.get('wall_nodes',24)));start=inner.index('kOmegaSSTCoeffs {')+len('kOmegaSSTCoeffs {');body=inner[start:].rsplit('}',2)[0]
    s+='enableSst true;\nsst { '+body+' }\n'
    s+='boundaries { inlet { kind Inlet; } outlet { kind Outlet; } wall { kind Interface; } top { kind Slip; } }\n'
    t=(dest/'constant/gasModelProperties').read_text();s+='modelFingerprint "'+hashlib.sha256((s+t).encode()).hexdigest()+'";\n';(dest/'constant/chmtProperties').write_text(s)
    T=700.;U=float(opts.get('velocity',30.));rho=101325/(8.31446261815324/.028*T)
    bp={'inlet':'type fixedValue; value uniform VALUE;','outlet':'type zeroGradient;','wall':'type zeroGradient;','top':'type zeroGradient;','sides':'type empty;'}
    for name,dims,val,vector in [('rho','1 -3 0 0 0 0 0',rho,False),('T','0 0 0 1 0 0 0',T,False),('U','0 1 -1 0 0 0 0',f'({U} 0 0)',True),('Y_S0','0 0 0 0 0 0 0',.1,False),('Y_S1','0 0 0 0 0 0 0',.9,False),('k','0 2 -2 0 0 0 0',1e-4,False),('omega','0 0 -1 0 0 0 0',1.,False)]:
        ps={k:v.replace('VALUE',str(val)) for k,v in bp.items()}
        if name=='U':ps['wall']='type fixedValue; value uniform (0 0 0);'
        write_field(dest,'',name,dims,val,ps,vector)
    for old in list((dest/'0/solid').iterdir()):
        if old.is_file():
            match=re.search(r'internalField\s+uniform\s+([^;]+);',old.read_text());dims=re.search(r'dimensions\s+\[([^]]+)\]',old.read_text()).group(1)
            val=match.group(1);ps={p:'type zeroGradient;' for p in ('solidInlet','solidOutlet','back','interface')};ps['sides']='type empty;';write_field(dest,'solid',old.name,dims,val,ps)
    # Clean obsolete 2x2x2 gas fields; rewritten gas set already complete.
    out.update(cells=nx*ny,metric='chmt_profiles',parameters=dict(nx=nx,ny=ny,solid_ny=ns,length=L,height=H,grading=G,width=W,closed=False,velocity=U,temperature=T,pressure=101325.,rho=rho),wall_family=family)
    return out

def _mesh_patch_names(path):
    text=path.read_text();return re.findall(r'\b([A-Za-z_]\w*)\s*\{\s*type\s+\w+;\s*(?:inGroups\s+[^;]+;\s*)?nFaces',text)

def _mesh_cell_count(directory):
    addressing={}
    for name in ('owner','neighbour'):
        text=(directory/name).read_text()
        body=re.search(r'\n\s*(\d+)\s*\n\s*\((.*?)\)',text,re.S)
        if not body:raise ValueError('imported mesh '+name+' list must be ASCII')
        labels=list(map(int,body.group(2).split()))
        if len(labels)!=int(body.group(1)) or any(label<0 for label in labels):
            raise ValueError('invalid imported mesh '+name+' addressing')
        addressing[name]=labels
    if not addressing['owner'] or len(addressing['neighbour'])>len(addressing['owner']):
        raise ValueError('invalid imported mesh face counts')
    return max(addressing['owner']+addressing['neighbour'])+1

def _mss7(dest,opts,moving=False):
    # Preserve every geometric point/face; explicitly model a 3-D slip-sided sector.
    out=_flat_chmt(dest,opts) if moving else _flat_gas(dest,opts)
    src=ROOT/'examples/thermal/MSS7_laminar/constant'
    for region,target in [('fluid',dest/'constant/polyMesh')]+([('graphite',dest/'constant/solid/polyMesh')] if moving else []):
        source=Path(opts.get('gas_mesh' if region=='fluid' else 'solid_mesh',src/region/'polyMesh')).resolve()
        shutil.copytree(source,target)
        boundary=target/'boundary';s=boundary.read_text().replace('type            wedge;','type            patch;');boundary.write_text(s)
    gas_patches=_mesh_patch_names(dest/'constant/polyMesh/boundary');solid_patches=_mesh_patch_names(dest/'constant/solid/polyMesh/boundary') if moving else []
    if not gas_patches:raise ValueError('MSS7 patch parser failed')
    for region,ps in [('',gas_patches)]+([('solid',solid_patches)] if moving else []):
        for f in (dest/'0'/region).iterdir():
            if not f.is_file():continue
            text=f.read_text();val=re.search(r'internalField\s+uniform\s+([^;]+);',text)
            if not val:raise ValueError('MSS7 adapted input requires uniform initial fields')
            dims=re.search(r'dimensions\s+\[([^]]+)\]',text).group(1);v=val.group(1);types={p:'type zeroGradient;' for p in ps}
            if 'defaultFaces' in types:types['defaultFaces']='type empty;'
            if region=='':
                if 'inlet' in types:types['inlet']='type fixedValue; value uniform '+v+';'
                if f.name=='U':
                    types['fluid_to_graphite']='type fixedValue; value uniform (0 0 0);'
                    if not moving:
                        types['wedgeBack']='type slip;';types['wedgeFront']='type slip;'
                if f.name=='T' and not moving:types['fluid_to_graphite']='type fixedValue; value uniform 300;'
                if f.name=='k' and not moving:types['fluid_to_graphite']='type fixedValue; value uniform 0;'
            write_field(dest,region,f.name,dims,v,types,f.name in ('U','Us'))
    if moving:
        p=dest/'constant/chmtProperties';s=p.read_text().replace('surface { patch interface; gasPatch wall; }','surface { patch graphite_to_fluid; gasPatch fluid_to_graphite; }')
        s=s.replace('wall { kind Interface; } top { kind Slip; }','fluid_to_graphite { kind Interface; } wedgeBack { kind Slip; } wedgeFront { kind Slip; }')
        s=re.sub(r'modelFingerprint\s+"[^"]+";','',s);s+='modelFingerprint "'+hashlib.sha256((s+(dest/'constant/gasModelProperties').read_text()).encode()).hexdigest()+'";\n';p.write_text(s)
    # blockMesh is intentionally omitted for imported production meshes.
    mesh_cells=_mesh_cell_count(dest/'constant/polyMesh')
    out.update(cells=mesh_cells,metric='chmt_profiles' if moving else 'profiles',mesh_commands=[],parameters=dict({k:v for k,v in out['parameters'].items() if k in ('velocity','temperature','pressure','rho','closed')},geometry_source=str(src.relative_to(ROOT)),gas_mesh_override=str(opts.get('gas_mesh','none')),solid_mesh_override=str(opts.get('solid_mesh','none')),geometry_adaptation='Imported 3D sector with wedge patch types replaced by explicit slip-sided patches. Optional user mesh overrides are recorded separately. Not an axisymmetric equation validation.',closed=False))
    return out

def prepare(name,destination,options=None):
    options=dict(options or {});dest=Path(destination)
    if dest.exists():raise FileExistsError('Refusing to overwrite existing case '+str(dest))
    spec=cases()[name]
    if 'wall' in options and spec['wall_variants'] and options['wall'] not in spec['wall_variants']:raise ValueError('unsupported wall family for '+name)
    dest.parent.mkdir(parents=True,exist_ok=True)
    try:
        k=spec['generator']
        if k in ('couette','couette_wall'):extra=_couette(dest,options,k=='couette_wall')
        elif k=='sod':extra=_sod(dest,options)
        elif k in ('wave','reacting_wave'):
            extra=_wave(dest,options)
            if k=='reacting_wave':
                from synthetic_chemistry import enable
                enable(dest,['A','B']);extra['parameters']['reaction_rate_per_s']=2.;extra['metric']='reacting_wave'
        elif k=='reactor':extra=_reactor(dest,options)
        elif k=='contact':extra=_chmt_base(dest,options,True)
        elif k in ('receding','reacting_receding','laminar_reacting_wall'):
            extra=_chmt_base(dest,options)
            if k in ('reacting_receding','laminar_reacting_wall'):
                from synthetic_chemistry import enable
                source=enable(dest,['S0','S1']);extra['parameters']['reaction_rate_per_s']=2.;extra['material_card']['gas_phase_reactions']=[source]
            if k=='laminar_reacting_wall':
                p=dest/'constant/chmtProperties';text=p.read_text();text=re.sub(r'modelFingerprint\s+"[^"]+";','',text)
                inner=wall_config('boundaryLayer_finiteRate',True,int(options.get('wall_nodes',24))).replace('simulationType laminar;','')
                text+='enableSst false;\nsst { '+inner+' }\n'
                text+='modelFingerprint "'+hashlib.sha256((text+(dest/'constant/gasModelProperties').read_text()).encode()).hexdigest()+'";\n';p.write_text(text)
                extra['wall_family']='boundaryLayer_finiteRate'
        elif k=='flatplate':extra=_flat_gas(dest,options)
        elif k=='flatplate_moving':extra=_flat_chmt(dest,options)
        elif k=='mss7':extra=_mss7(dest,options)
        elif k=='mss7_moving':extra=_mss7(dest,options,True)
        else:raise ValueError('unsupported generator')
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
        spec.update(extra,source_commit=commit,options=options,inputs_sha256=fingerprint(dest),native_execution='NOT_RUN',cpu_reference='NOT_RUN',mesh='NOT_RUN',validation='NOT_RUN')
        control=(dest/'system/controlDict').read_text()
        spec['time_controls']={key:(re.search(r'\b'+key+r'\s+([^;]+);',control).group(1).strip() if re.search(r'\b'+key+r'\s+([^;]+);',control) else None) for key in ('deltaT','adjustTimeStep','maxCo','maxDeltaT','endTime','writeInterval')}
        spec['backend_entries']={'native_gpu':'run.py --mode execute --required','cpu_reference':'metrics.py compare() after actual native output; never a solver substitute','cpu_kernel_probes':'host_cases.py --backend cpu (separate explicitly scoped production-kernel tests)'}
        spec['evidence_boundary']='Input generation does not execute a solver. No GPU validation or experimental agreement has been demonstrated.'
        _json(dest/'suite_case.json',spec)
        if 'material_card' in spec:_json(dest/'material_card.json',spec['material_card'])
        return spec
    except Exception as e:
        _json(dest/'generation_failure.json',{'status':'FAILED','reason':str(e),'native_execution':'NOT_RUN'})
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',choices=sorted(cases()));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--options',type=Path,help='JSON override file; e.g. nx, ny, height, grading, wall, material, end_time');a=p.parse_args()
    print(json.dumps(prepare(a.case,a.output,json.loads(a.options.read_text()) if a.options else {}),sort_keys=True,indent=2))
if __name__=='__main__':main()
