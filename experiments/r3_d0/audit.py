"""Recheck local release bytes and freeze a train-only detection split."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np

from .llvip import parse_voc

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'reports/evidence'
SEED=3407


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dump(path,obj):
    path.write_bytes((json.dumps(obj,indent=2,allow_nan=False)+'\n').encode())


def group_split(groups, seed=SEED):
    """Deterministic subset sum, nearest to 20% by retained image count."""
    order=sorted(groups,key=lambda g:(sha(f'{seed}/{g}'.encode()),g))
    n=sum(map(len,groups.values()))
    possible={0:()}
    for g in order:
        for count, chosen in tuple(possible.items()):
            possible.setdefault(count+len(groups[g]),chosen+(g,))
    valid=set(possible)-{0,n}
    if not valid:
        raise ValueError('Need at least two components')
    target=min(valid,key=lambda k:(abs(5*k-n),k))
    return sorted(possible[target])


def run(root,out):
    archive=E/'llvip-precheck'
    manifest=(archive/'file_manifest.json').read_bytes()
    prior=json.loads((archive/'data_audit.json').read_bytes())
    if sha(manifest)!=prior['manifest_sha256']:
        raise ValueError('Historical manifest digest mismatch')
    rows={r['path']:r for r in json.loads(manifest)['files']}
    seen=set()
    ids={}
    for split,expected in [('train',12025),('test',3463)]:
        sides=[]
        for m in ('visible','infrared'):
            files=list((root/m/split).iterdir())
            side={p.stem for p in files if p.is_file() and p.suffix=='.jpg'}
            if len(files)!=len(side) or len(side)!=expected:
                raise ValueError('Unexpected image inventory')
            sides.append(side)
        if sides[0]!=sides[1] or sides[0]&seen:
            raise ValueError('Missing or cross-split ID')
        ids[split]=sorted(sides[0]); seen.update(sides[0])
    annotations=list((root/'Annotations').iterdir())
    if len(annotations)!=len(seen) or {p.name for p in annotations}!={i+'.xml' for i in seen}:
        raise ValueError('Unexpected annotation inventory')

    def verify(rel):
        p=root/rel
        with p.open('rb') as f:
            h=hashlib.file_digest(f,'sha256').hexdigest()
        if h!=rows[rel]['sha256']:
            raise ValueError('Changed local file: '+rel)

    rels=[f'{m}/{s}/{i}.jpg' for s in ids for i in ids[s] for m in ('visible','infrared')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(verify,rels):
            pass
    invalid=[]
    totals={}
    annotation_counts={}
    for s, roster in ids.items():
        totals[s]=Counter()
        for i in roster:
            rel=f'Annotations/{i}.xml'
            b=(root/rel).read_bytes()
            h=sha(b)
            if h!=rows[rel]['sha256']:
                raise ValueError('Changed local annotation: '+rel)
            boxes,flags,bad=parse_voc(b,i)
            annotation_counts[i]=len(boxes)
            totals[s].update(person=len(boxes)+len(bad),invalid=len(bad),difficult=sum(flags))
            invalid.extend(dict(id=i,official_split=s,xml_sha256=h,**r) for r in bad)

    def img(m,s,i):
        return rows[f'{m}/{s}/{i}.jpg']

    def pval(m,s,i):
        return int(img(m,s,i)['phash63'],16)

    candidates=[]
    for left,right,d in prior['perceptual_train_test']['visible']['candidate_paths']:
        ti,ri=Path(left).stem,Path(right).stem
        if (pval('visible','test',ti)^pval('visible','train',ri)).bit_count()!=d:
            raise ValueError('RGB candidate changed')
        t=(pval('infrared','test',ti)^pval('infrared','train',ri)).bit_count()
        candidates.append(dict(official_test=ti,official_train=ri,rgb_hamming=d,
                               thermal_hamming=t,
                               exact_in_either_modality=any(img(m,'test',ti)['pixel_sha256']==
                                   img(m,'train',ri)['pixel_sha256'] for m in ('visible','infrared')),
                               verdict='UNKNOWN'))
    pilot=[]
    for path in [E/'llvip-pilot/roster.json', E/'llvip-pilot/p4/roster.json']:
        roster=[p['id'] for p in json.loads(path.read_bytes())['pairs']]
        pilot.append(roster)
    if pilot[0]!=pilot[1] or len(set(pilot[0]))!=64 or not set(pilot[0])<=set(ids['test']):
        raise ValueError('Unexpected prior test exposure')

    train=ids['train']
    prefixes={i:i[:2] for i in train}
    parent={g:g for g in prefixes.values()}

    def find(g):
        while parent[g]!=g:
            g=parent[g]
        return g

    def join(a,b):
        a,b=sorted((find(prefixes[a]),find(prefixes[b])))
        parent[b]=a

    # Prefix is a conservative proxy, not a verified capture session.
    exact=defaultdict(list)
    for i in train:
        for m in ('visible','infrared'):
            exact[(m,img(m,'train',i)['pixel_sha256'])].append(i)
    for cluster in exact.values():
        for i in cluster[1:]:
            join(cluster[0],i)
    rgb,thermal=(np.array([pval(m,'train',i) for i in train],dtype=np.uint64)
                 for m in ('visible','infrared'))
    edge_counts=Counter()
    for idx,i in enumerate(train):
        dr=np.bitwise_count(rgb[idx+1:]^rgb[idx]); dt=np.bitwise_count(thermal[idx+1:]^thermal[idx])
        for j in np.flatnonzero((dr<=6)&(dt<=6)):
            other=train[idx+1+int(j)]
            if prefixes[i]!=prefixes[other]:
                edge_counts['-'.join(sorted((prefixes[i],prefixes[other])))]+=1
                join(i,other)
    quarantine={r['id']:'invalid_annotation_frame' for r in invalid}
    for r in candidates:
        i=r['official_train']
        quarantine[i]=';'.join(filter(None,(quarantine.get(i),'cross_official_split_candidate_UNKNOWN')))
    groups=defaultdict(list)
    for i in train:
        if i not in quarantine:
            groups[find(prefixes[i])].append(i)
    val_groups=group_split(groups)
    val=sorted(i for g in val_groups for i in groups[g])
    fit=sorted(set(train)-set(val)-set(quarantine))
    if len(fit)+len(val)+len(quarantine)!=12025:
        raise ValueError('Nonexclusive split')
    file_hashes={i:{p:rows[p]['sha256'] for p in (f'visible/train/{i}.jpg',f'infrared/train/{i}.jpg',
                   f'Annotations/{i}.xml')} for i in sorted(fit+val)}
    split=dict(seed=SEED,fit=fit,validation=val,quarantine=quarantine,
               proxy_components=dict(sorted(groups.items())),validation_components=val_groups,
               development_sha256=file_hashes,official_test_used_for_selection=False)
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'split.json',split)
    dump(out/'audit.json',dict(pairs={s:len(v) for s,v in ids.items()},jpeg_sha256_rechecked=len(rels),
                              xml_sha256_rechecked=len(seen),annotation_totals=totals,
                              invalid_annotations=invalid,cross_split_candidates=candidates,
                              historic_pilot_test_ids=pilot[0],
                              official_test_exposure=dict(xml_read=len(ids['test']),
                                  person_annotations_counted=totals['test']['person'],
                                  images_pixel_decoded=0, detection_inferences=0,
                                  detection_metric_computations=0, model_selections=0,
                                  historical_thermal_pilot_pairs=len(pilot[0])),
                              candidate_test_id_previously_seen=any(r['official_test'] in pilot[0]
                                                                   for r in candidates),
                              cross_prefix_dual_phash_edges=dict(edge_counts),
                              development={s:{'pairs':len(v),'boxes':sum(annotation_counts[i] for i in v)}
                                           for s,v in [('fit',fit),('validation',val)]},
                              manifest_sha256=sha(manifest),split_sha256=sha((out/'split.json').read_bytes()),
                              environment=dict(python=platform.python_version(),numpy=np.__version__,
                                               opencv=cv2.__version__,hash_workers=4)))
    print('PASS inventory',totals,'development',len(fit),len(val),'groups',val_groups,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    run(a.root,a.output)
