"""Scan LLVIP release bytes without creating or altering images."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import io
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zipfile
import zlib

import numpy as np
from PIL import Image


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False)+'\n').encode())


def phash(im):
    from scipy.fft import dctn

    gray = np.asarray(im.convert('L').resize((32, 32), Image.Resampling.LANCZOS),
                      dtype=np.float32)
    bands = dctn(gray, norm='ortho')[:8, :8].ravel()[1:]
    return np.packbits(bands > np.median(bands)).tobytes().hex()


def inspect(root, path, zip_index=None):
    rel = path.relative_to(root).as_posix()
    parts = rel.split('/')
    data = path.read_bytes()
    row = dict(path=rel, bytes=len(data), sha256=sha(data))
    if zip_index is not None:
        row['zip_crc32_match'] = (
            zip_index.get('LLVIP/'+rel) == (len(data), zlib.crc32(data)))
    try:
        if parts[0] in ('visible', 'infrared'):
            if len(parts) != 3 or parts[1] not in ('train', 'test'):
                raise ValueError('Unexpected image layout')
            row.update(modality=parts[0], split=parts[1], id=path.stem)
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                row.update(format=im.format, width=im.width, height=im.height,
                           mode=im.mode, channels=len(im.getbands()),
                           bits_per_sample=getattr(im, 'bits', None),
                           pixel_sha256=sha(im.mode.encode()+struct.pack('>II', *im.size)+im.tobytes()),
                           phash63=phash(im),
                           capture_exif=any(im.getexif().get(k) for k in (306, 36867, 36868)))
        elif parts[0] == 'Annotations':
            x = ET.fromstring(data)
            w, h = (int(x.findtext('size/'+k)) for k in ('width', 'height'))
            objects = x.findall('object')
            bad = 0
            for obj in objects:
                a, b, c, d = (int(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax'))
                bad += not (0 <= a < c <= w and 0 <= b < d <= h)
            row.update(kind='voc', id=path.stem, annotation_filename=x.findtext('filename'),
                       width=w, height=h, classes=dict(Counter(o.findtext('name') for o in objects)),
                       segmented=x.findtext('segmented'), invalid_boxes=bad)
    except (OSError, ValueError, TypeError, ET.ParseError) as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
    return row


def pair_inventory(rows, split):
    sides = {m: defaultdict(list) for m in ('visible', 'infrared')}
    for r in rows:
        if r.get('split') == split and 'error' not in r:
            sides[r['modality']][r['id']].append(r)
    v, t = sides.values()
    shared = sorted(v.keys() & t.keys())
    ambiguous = sorted({i for side in (v,t) for i, rr in side.items() if len(rr) != 1})
    mismatches = [i for i in shared if any(v[i][0][k] != t[i][0][k] for k in ('width','height'))]
    stats = dict(visible_ids=len(v), thermal_ids=len(t), paired_ids=len(shared),
                 visible_only=sorted(v.keys()-t.keys()), thermal_only=sorted(t.keys()-v.keys()),
                 ambiguous_ids=ambiguous, dimension_mismatch_ids=mismatches,
                 id_prefix2_counts=dict(Counter(i[:2] for i in shared)))
    for m, index in sides.items():
        rr = [r for group in index.values() for r in group]
        specs = Counter(f"{r['format']} {r['width']}x{r['height']} {r['mode']} "
                        f"{r['channels']}ch {r['bits_per_sample']}bit" for r in rr)
        stats[m] = dict(files=len(rr), spec_counts=dict(specs),
                        bytes_min_p05_median_p95_max=np.percentile(
                            [r['bytes'] for r in rr], [0,5,50,95,100]).tolist() if rr else [],
                        exif_capture_timestamp_present=sum(r['capture_exif'] for r in rr))
    rejected = set(ambiguous+mismatches)
    return stats, {i:(v[i][0], t[i][0]) for i in shared if i not in rejected}


def _popcount(values):
    """Population count over uint64 Hamming distances, including NumPy 1.x."""
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(values)
    # A byte lookup is endian independent because all eight counts are summed.
    table = np.array([b.bit_count() for b in range(256)], dtype=np.uint8)
    octets = np.ascontiguousarray(values).view(np.uint8).reshape(-1, 8)
    return table[octets].sum(axis=1, dtype=np.uint8).reshape(values.shape)


def summarize(rows):
    images = [r for r in rows if 'modality' in r and 'error' not in r]
    annotations = [r for r in rows if r.get('kind') == 'voc']
    partitions = {s:pair_inventory(rows,s) for s in ('train','test')}
    ids = set.union(*(set(p) for _,p in partitions.values()))
    ann_ids = {r['id'] for r in annotations}
    duplicates = {}
    for key in ('sha256','pixel_sha256'):
        index = defaultdict(list)
        for r in images:
            index[r[key]].append(r)
        groups = [g for g in index.values() if len(g)>1]
        cross = [g for g in groups if len({r['split'] for r in g})>1]
        duplicates[key] = dict(groups=len(groups), extra_files=sum(len(g)-1 for g in groups),
                               affected_files=sum(map(len,groups)), cross_split_groups=len(cross),
                               groups_paths=[[r['path'] for r in g] for g in groups],
                               cross_split_paths=[[r['path'] for r in g] for g in cross])
    perceptual = {}
    for m in ('visible','infrared'):
        train, test = ([r for r in images if (r['modality'],r['split']) == (m,s)]
                       for s in ('train','test'))
        h = np.array([int(r['phash63'],16) for r in train], dtype=np.uint64)
        hits = []
        minimum = None
        for r in test:
            dist = _popcount(h ^ np.uint64(int(r['phash63'],16)))
            if len(dist):
                minimum = int(dist.min()) if minimum is None else min(minimum,int(dist.min()))
            hits.extend([r['path'],train[j]['path'],int(dist[j])] for j in np.flatnonzero(dist<=6))
        perceptual[m] = dict(pairs_at_hamming_le_6=len(hits), minimum_distance=minimum,
                             left_images_with_hit=len({hit[0] for hit in hits}),
                             candidate_paths=sorted(hits,key=lambda z:(z[2],z[0],z[1])))
    classes = Counter()
    for r in annotations:
        classes.update(r['classes'])
    return dict(
        files=len(rows), decode_errors=[[r['path'],r['error']] for r in rows if 'error' in r],
        zip_crc32_mismatches=[r['path'] for r in rows if r.get('zip_crc32_match') is False],
        pairing={s:stats for s,(stats,_) in partitions.items()},
        duplicate_split_ids=sorted(partitions['train'][1].keys() & partitions['test'][1].keys()),
        duplicates=duplicates, perceptual_train_test=perceptual,
        annotation=dict(files=len(annotations), classes=dict(classes),
                        segmented_counts=dict(Counter(r['segmented'] for r in annotations)),
                        raw_dims=dict(Counter(f"{r['width']}x{r['height']}" for r in annotations)),
                        filename_mismatch_count=sum(Path(r['annotation_filename'] or '').stem!=r['id']
                                                    for r in annotations),
                        invalid_boxes=sum(r['invalid_boxes'] for r in annotations),
                        files_with_invalid_boxes=[[r['path'],r['invalid_boxes']]
                                                  for r in annotations if r['invalid_boxes']],
                        images_without_annotations=sorted(ids-ann_ids),
                        annotations_without_images=sorted(ann_ids-ids)))


def review_material(root, candidates, output):
    output.mkdir(parents=True, exist_ok=False)
    lines = ['<!doctype html><meta charset="utf-8"><title>LLVIP local review</title>',
             '<h1>LLVIP: all candidates UNKNOWN</h1><p>Compare each RGB pair and its Thermal '
             'counterparts. Record same frame, related scene, unrelated, or insufficient evidence '
             'with provenance separately. Hash proximity alone cannot determine leakage.</p>']
    records = []
    for test, train, distance in candidates:
        records.append(dict(test=test,train=train,distance=distance,decision='UNKNOWN'))
        lines.append(f'<h2>{html.escape(test)} / {html.escape(train)} (Hamming {distance})</h2>')
        for m in ('visible','infrared'):
            for rel in (test,train):
                path = root / m / Path(*Path(rel).parts[1:])
                lines.append(f'<figure style="display:inline-block"><img width="430" '
                             f'src="{html.escape(path.resolve().as_uri())}"><figcaption>'
                             f'{html.escape(path.relative_to(root).as_posix())}</figcaption></figure>')
            lines.append('<br>')
    dump(output/'review.json',records)
    (output/'review.html').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--zip',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review-local',type=Path)
    p.add_argument('--workers',type=int,default=4)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    paths = sorted((f for f in a.root.rglob('*') if f.is_file()),
                   key=lambda f:f.relative_to(a.root).as_posix())
    if not paths:
        raise SystemExit('No files found')
    index = None
    if a.zip:
        with zipfile.ZipFile(a.zip) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
        index = {i.filename:(i.file_size,i.CRC) for i in infos}
        if len(index)!=len(infos) or set(index)!={'LLVIP/'+f.relative_to(a.root).as_posix()
                                                 for f in paths}:
            raise SystemExit('ZIP and extracted inventory differ')
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(lambda f:inspect(a.root,f,index),paths))
    result = summarize(rows)
    dump(a.output/'data_audit.json',result)
    with (a.output/'file_manifest.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row,separators=(',',':'))+'\n')
    if a.review_local:
        review_material(a.root,result['perceptual_train_test']['visible']['candidate_paths'],
                        a.review_local)


if __name__ == '__main__':
    main()
