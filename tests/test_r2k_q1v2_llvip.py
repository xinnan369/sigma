import cv2
import numpy as np
from PIL import Image
import pytest

from experiments.r2k_q1v2_llvip import pilot, scan


def test_exact_crop_area_and_gray():
    a=np.zeros((1024,1280,3),np.uint8)
    a[:32]=255
    a[992:]=255
    a[32:34,:2]=[[[0,0,200],[0,100,0]],[[200,0,0],[0,0,0]]]
    a[990:992,1278:]=(0,100,0)
    v=pilot.adapt(a)
    assert v.shape==(480,640)
    assert v[0,0]==cv2.cvtColor(np.array([[[50,25,50]]],np.uint8),
                                 cv2.COLOR_BGR2GRAY)[0,0]
    assert v[-1,-1]==59
    assert np.count_nonzero(v)==2
    with pytest.raises(ValueError):
        pilot.adapt(a[:,:,0])


def test_scan_pairs_duplicates_malformed_and_annotation(tmp_path):
    for m,s,i in [('visible','train','a'),('infrared','train','a'),
                  ('visible','test','a'),('infrared','test','a'),('visible','test','b')]:
        p=tmp_path/m/s/(i+'.jpg')
        p.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(16,16)).save(p)
    bad=tmp_path/'visible/test/bad.jpg'
    bad.write_bytes(b'invalid')
    xml=tmp_path/'Annotations/a.xml'
    xml.parent.mkdir()
    xml.write_text('<annotation><filename>a.jpg</filename><size><width>16</width>'
                   '<height>16</height></size><segmented>0</segmented><object><name>person</name>'
                   '<bndbox><xmin>2</xmin><xmax>2</xmax><ymin>1</ymin><ymax>4</ymax>'
                   '</bndbox></object></annotation>')
    rows=[scan.inspect(tmp_path,p) for p in sorted(tmp_path.rglob('*')) if p.is_file()]
    s=scan.summarize(rows)
    assert s['pairing']['test']['paired_ids']==1
    assert s['pairing']['test']['visible_only']==['b']
    assert s['duplicate_split_ids']==['a']
    assert s['duplicates']['sha256']['cross_split_groups']==1
    assert s['duplicates']['pixel_sha256']['extra_files']==4
    assert s['annotation']['invalid_boxes']==1
    assert s['decode_errors'][0][0]=='visible/test/bad.jpg'
    assert s['pairing']['test']['visible']['spec_counts']=={'JPEG 16x16 RGB 3ch 8bit':2}
    assert s['perceptual_train_test']['visible']['pairs_at_hamming_le_6']==2
    scan.review_material(tmp_path,s['perceptual_train_test']['visible']['candidate_paths'],tmp_path/'review')
    assert all(r['decision']=='UNKNOWN' for r in pilot.read(tmp_path/'review/review.json'))


def test_roster_deterministic_and_excludes_unpaired_and_wrong_dimensions():
    rows=[]
    for i in ('a','b','c','d'):
        for m in ('visible','infrared'):
            if (i,m)==('c','infrared'):
                continue
            rows.append(dict(id=i,split='test',modality=m,width=(1279 if i=='d' else 1280),
                             height=1024,mode='RGB',bits_per_sample=8,format='JPEG',
                             channels=3,bytes=10,capture_exif=False,
                             path=f'{m}/test/{i}.jpg',sha256=f'{m}-{i}'))
    r=pilot.chosen_roster(rows,3407,2)
    assert r==pilot.chosen_roster(rows[::-1],3407,2)
    assert {v['id'] for v in r}=={'a','b'}
    assert all(v['rgb']['path'].endswith(v['id']+'.jpg') and
               v['thermal']['path'].endswith(v['id']+'.jpg') for v in r)
    with pytest.raises(ValueError):
        pilot.chosen_roster(rows,3407,3)

def test_history_scope_refuses_old_holdout_and_validation():
    from experiments.r2k_q1v2_llvip.history import fit_calibration_ids
    p={'partitions':{k:{'images':{ident:'group'}} for k,ident in
                     [('fit','a'),('calibration','b'),('holdout','h')]}}
    assert fit_calibration_ids(p,['v'])==['a','b']
    with pytest.raises(ValueError):
        fit_calibration_ids(p,['b'])
    p['partitions']['holdout']['images']['a']='group'
    with pytest.raises(ValueError):
        fit_calibration_ids(p,[])

def test_archived_clean_input_flags_are_counts_not_health_labels():
    import hashlib
    base=pilot.ROOT/'reports/evidence/llvip-pilot'
    for folder in (base,base/'p4'):
        stats=pilot.read(folder/'results.json')
        ledger=(folder/'per-image.jsonl').read_bytes()
        assert hashlib.sha256(ledger).hexdigest()==stats['per_image_sha256']
        rows=[__import__('json').loads(line) for line in ledger.splitlines()]
        assert len(rows)==64 and sum(len(r['blur']) for r in rows)==128
        for arm, result in stats['metrics'].items():
            flagged=sum(r['clean_input_flagged'][arm] for r in rows)
            points=sum(r['clean_input_points'] for r in rows)
            assert points==64*4800
            assert result['clean_input_flag_rate']==flagged/points
            assert 'healthy_fpr' not in result
