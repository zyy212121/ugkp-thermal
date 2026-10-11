from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generate import prepare
from catalog import cases

def test_new_plate_keeps_box_distinct_and_preserves_material_and_endpoint(tmp_path):
    box=prepare('chmt_laminar_reacting_wall',tmp_path/'box')
    plate=prepare('chmt_laminar_reacting_plate',tmp_path/'plate')
    assert box['generator']=='laminar_reacting_wall'
    assert plate['generator']=='laminar_reacting_plate'
    assert box['end_time']==plate['end_time']==.01
    assert box['material_card']==plate['material_card']
    assert plate['cells']==16
    assert '(4 2 2)' in (tmp_path/'plate/system/blockMeshDict').read_text()
    assert 'boundaries { sides { kind Slip; } }' in (tmp_path/'plate/constant/chmtProperties').read_text()
    assert 'boundaries { sides { kind Slip; } }' not in (tmp_path/'box/constant/chmtProperties').read_text()
    assert 'wall_corner_geometry' in cases()['chmt_laminar_reacting_wall']['known_limitation']

def test_plate_default_matches_admissible_validated_wall_resolution(tmp_path):
    # The retained N24 counterexample has a negative discrete reactant profile.
    # N48 is the resolution used by the accepted native coupled windows.
    prepare('chmt_laminar_reacting_plate',tmp_path/'default')
    prepare('chmt_laminar_reacting_plate',tmp_path/'counterexample',{'wall_nodes':24})
    prepare('chmt_laminar_reacting_wall',tmp_path/'original_box')
    assert 'nodes 48;' in (tmp_path/'default/constant/chmtProperties').read_text()
    assert 'nodes 24;' in (tmp_path/'counterexample/constant/chmtProperties').read_text()
    assert 'nodes 24;' in (tmp_path/'original_box/constant/chmtProperties').read_text()
