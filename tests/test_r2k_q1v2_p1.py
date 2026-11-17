import hashlib

import cv2
import numpy as np
import pytest

from experiments.r2k_q1v2_p1.admission import ARMS, PRIMARY, SECONDARY, archive_hashes, preflight
from experiments.r2k_q1v2_p1.statistics import decision, quantities, summarize_pairs


def save_fixture(root, name, payload):
    (root/name).write_bytes(payload)
    return {'path':name,'sha256':hashlib.sha256(payload).hexdigest()}


def png(value, shape=(480,640), dtype=np.uint8):
    ok,encoded=cv2.imencode('.png',np.full(shape,value,dtype=dtype))
    assert ok
    return encoded.tobytes()


def metadata(tmp_path):
    doc={'schema':'rgbt-p1r-v1','source_url':'https://example.invalid/synthetic',
         'input_representation':'provider_native_p0_grid',
         'rights':{'research':True,'local_derivatives':True,'local_audit':True,
                   'terms_url':'https://example.invalid/terms','review_receipt':'synthetic-only'},
         'duplicate_review':{'complete':True,'receipt':'synthetic-only','method':'test edges',
                             'historical_scope':'all_available_mfnet_and_historical_rgbt',
                             'historical_matches':[],'internal_links':[]},'frames':[]}
    for i in range(40):
        row={'id':f'f{i}','group':f'g{i}','site':f's{i}','sequence':f'q{i}','sensor':'fixture',
             'capture_time':'2026-01-01T16:00:00+00:00','illumination':'day' if i<20 else 'night',
             'illumination_receipt':'synthetic-log','alignment_receipt':'synthetic-log',
             'registration':'registered','thermal_state':'intact','supports_frozen_blur':True}
        row['rgb']=save_fixture(tmp_path,f'{i}-rgb.png',png(i,(480,640,3)))
        row['thermal']=save_fixture(tmp_path,f'{i}-thermal.png',png(i+40))
        doc['frames'].append(row)
    return doc


def test_metadata_counts_and_byte_overlap(tmp_path):
    x=metadata(tmp_path)
    ready=preflight(x,archive_hashes(),tmp_path)
    assert ready['status']=='READY_FOR_INDEPENDENT_REVIEW'
    assert ready['scientific_admission']=='PENDING_INDEPENDENT_REVIEW'
    assert ready['gates']=={PRIMARY:True,SECONDARY:False}
    assert ready['support']=={'groups':40,'day':20,'night':20,'positive_blur_groups':40}
    match=x['frames'][0]['rgb']['sha256']
    out=preflight(x,{match},tmp_path)
    assert out['status']=='BLOCKED' and out['required_exclusions']==['f0']
    assert out['gates']=={PRIMARY:False,SECONDARY:False}
    assert preflight(x,set())['status']=='BLOCKED'


def test_does_not_treat_same_sequence_or_related_frames_as_distinct(tmp_path):
    x=metadata(tmp_path)
    x['frames'][2]['sequence']=x['frames'][1]['sequence']
    x['duplicate_review']['internal_links']=[['f0','f4','neighbor']]
    x['rights']['local_derivatives']=False
    result=preflight(x,set(),tmp_path)
    assert {'split_sequence:q1','cross_group_neighbor:f0:f4','unverified_right:local_derivatives'}<=set(result['issues'])
    x['frames'][1]['capture_time']='2026-01-01'
    with pytest.raises(ValueError,match='timezone'):
        preflight(x,set(),tmp_path)


def test_manifest_checks_actual_bytes_and_traversal(tmp_path):
    x=metadata(tmp_path)
    p=tmp_path/x['frames'][3]['thermal']['path']
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA256'):
        preflight(x,set(),tmp_path)
    x['frames'][0]['rgb']['path']='../secret'
    with pytest.raises(ValueError,match='Unsafe'):
        preflight(x,set(),tmp_path)


def test_physical_defocus_evidence_is_verified_separately(tmp_path):
    x=metadata(tmp_path)
    x['physical_defocus_receipt']='self-attested'
    assert not preflight(x,set(),tmp_path)['gates'][SECONDARY]
    x['physical_defocus']={
        'controlled':True,'instrument_log':save_fixture(tmp_path,'instrument.csv',b'fixture settings'),
        'pairs':[{'intact_id':'f0','sensor':'fixture','intact_label':'intact',
                  'defocus_label':'controlled_defocus','log_entry':'fixture-row',
                  'intact_setting':'A','defocus_setting':'B',
                  'defocus_thermal':save_fixture(tmp_path,'defocus.png',png(111))}],
    }
    assert preflight(x,set(),tmp_path)['gates']=={PRIMARY:True,SECONDARY:True}
    (tmp_path/'instrument.csv').write_bytes(b'changed')
    assert preflight(x,set(),tmp_path)['gates']=={PRIMARY:True,SECONDARY:False}
    x['rights'].pop('local_audit')
    assert preflight(x,set(),tmp_path)['gates']=={PRIMARY:False,SECONDARY:False}


@pytest.mark.parametrize('payload',[png(500,dtype=np.uint16),png(20,(512,640)),png(20,(480,640,3))])
def test_native_thermal_rejects_format_even_with_valid_hash(tmp_path,payload):
    x=metadata(tmp_path)
    x['frames'][0]['thermal']=save_fixture(tmp_path,'unsupported.png',payload)
    with pytest.raises(ValueError,match='Incompatible P0 thermal'):
        preflight(x,set(),tmp_path)


def test_support_and_representation_gate(tmp_path):
    x=metadata(tmp_path)
    x['frames']=x['frames'][:34]
    for row in x['frames'][14:]:
        row['supports_frozen_blur']=False
    out=preflight(x,set(),tmp_path)
    assert set(out['issues'])=={'support_below_minimum:groups','support_below_minimum:night',
                               'support_below_minimum:positive_blur_groups'}
    x['input_representation']='resize_and_requantize'
    assert 'PROTOCOL_CHANGE_REQUIRED' in preflight(x,set(),tmp_path)['issues']


def synthetic_pairs():
    rows=[]
    for i in range(40):
        base=np.array([0,0,0,0],np.uint8)
        bad=np.array([200+i%2*31,0,0,0],np.uint8)
        y=np.array([True,False,False,False])
        scores={a:{'healthy':base,'blur0':base,'blur1':base} for a in ARMS}
        scores['thermal']={'healthy':base,'blur0':bad,'blur1':bad}
        observed={a:{k:np.zeros(4,bool) for k in ('healthy','blur0','blur1')} for a in ARMS}
        rows.append({'id':f'f{i}','group':f'g{i:02d}','illumination':'day' if i<20 else 'night',
                     'scores':scores,'low_texture':np.array([False,True,False,False]),
                     'targets':{'blur0':y,'blur1':y},'observable':observed})
    return rows


def test_synthetic_full_bootstrap_and_no_observability_exclusions():
    rows=synthetic_pairs()
    result=summarize_pairs(rows,{a:20 for a in ARMS})
    assert result['decision']=='PASS' and result['draws']==10000
    assert result['thermal_minus_rgb']['ap']['point']==.75
    assert result['thermal_minus_rgb']['ap']['valid_draws']==10000
    assert result['thermal_minus_rgb']['balanced_brier']['ci95'][1]<0
    assert result['strata']['thermal']['all']['recall']==1
    assert result['operators']['thermal']['healthy_fpr']==dict.fromkeys(('all','day','night','low_texture'),0.)
    assert result['operators']['thermal']['observability_coverage']==0
    assert result['operators']['thermal']['retained_coverage']==1


def test_frozen_guards_and_undefined_support():
    enough={'groups':40,'day':20,'night':20,'blur_positive':40}
    assert decision(enough,(.1,.2),(-.3,-.2),[False],True)=='FAIL'
    assert decision(enough,(-.1,.2),(-.3,-.2),[True],True)=='INCONCLUSIVE'
    assert decision(dict(enough,night=14),(.1,.2),(-.3,-.2),[True],True)=='INCONCLUSIVE'
    assert decision(enough,(.1,.2),(-.3,-.2),[None],True)=='INCONCLUSIVE'
    assert decision(enough,(.1,.2),(-.3,-.2),[True],False)=='FAIL'
    d=np.zeros((1,2,256),int); d[0,0,0]=2
    assert np.isnan(quantities(d)['balanced_brier'][0])
