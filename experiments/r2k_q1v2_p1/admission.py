"""Validate sealed metadata, group closure and byte manifests without inference."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[2]
PINS = {
    'docs/R2k_Q1v2_INDEPENDENT_EVALUATION.md': 'ddf5c95603285ad18babb1e575515c8237f7f22492ec342e0d239e7603859ca5',
    'reports/evidence/r2k-q1v2-p0/fit-calibration.json': 'f22ff63524ce69d43d218a8434141fb8c03b2b3585af53778455ecb0ed565c9c',
    'reports/evidence/r1c/manifest.json': '36410fb1f3c5e452613b2520d6540e6b9f080be64ada850a3ce1c60bce8cf543',
    'configs/R2k_Q1v2_protocol.json': '7207e245def001e4f7ad5ef31e4e138a6704b5f682bb2b630c64533e641d92bd',
}
ARMS = ('thermal', 'rgb', 'q1', 'ablated', 'shuffle', 'constant')
HASH = re.compile(r'[0-9a-f]{64}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def frozen_parameters(path):
    for name, digest in PINS.items():
        require(sha(ROOT/name)==digest, f'Accepted artifact differs: {name}')
    original=read(ROOT/'reports/evidence/r2k-q1v2-p0/fit-calibration.json')
    require(sha(path)==original['frozen_parameters_sha256'], 'Frozen P0 parameter bytes differ')
    params=read(path)
    require(params['source_sha256']==original['source_sha256']
            and params['protocol_sha256']==original['protocol_sha256']
            and all(sha(ROOT/name)==digest for name,digest in params['source_sha256'].items()),
            'Frozen P0 implementation differs')
    thresholds={a: params['calibration'][a]['threshold_bin'] for a in ARMS}
    require(thresholds=={a:original['calibration'][a]['threshold_bin'] for a in ARMS}, 'Threshold drift')
    return {'status':'PASS','parameter_sha256':sha(path),'pinned_files':PINS,
            'verified_source_sha256':params['source_sha256'],'thresholds':thresholds}


def archive_hashes():
    p=ROOT/'reports/evidence/r1c/manifest.json'
    require(sha(p)==PINS[p.relative_to(ROOT).as_posix()], 'Frozen MFNet byte manifest differs')
    return {r['sha256'] for r in read(p)['files'] if r['path'].startswith(('RGB/','Modal/'))}


def timestamp(s):
    parsed=datetime.fromisoformat(s.replace('Z','+00:00'))
    require(parsed.utcoffset() is not None, 'Capture timestamp requires timezone')
    return parsed


def file_record(file, root):
    name=file['path']
    rel=PurePosixPath(name)
    require(bool(name) and not rel.is_absolute() and '.'!=name and '..' not in rel.parts
            and ':' not in name and '\\' not in name and bool(HASH.fullmatch(file['sha256'])),
            'Unsafe or malformed file manifest')
    if root is not None:
        target=(root/rel).resolve(strict=True)
        require(target.is_relative_to(root.resolve()) and target.is_file() and sha(target)==file['sha256'],
                'File missing, outside archive, or different SHA256')
    return file['sha256']


def preflight(metadata, historical, root=None):
    """Return evidence readiness; a human reviewer still approves any acquisition."""
    require(metadata['schema']=='rgbt-p1-v1' and metadata['source_url'].startswith('https://'), 'Source metadata differs')
    rights=metadata['rights']
    review=metadata['duplicate_review']
    records=metadata['frames']
    require(records, 'Empty frame roster')
    for row in records:
        require(all(isinstance(row[k],str) and row[k].strip() for k in
                    ('id','group','site','sequence','sensor','alignment_receipt','illumination_receipt')),
                'Frame identifiers or attestations are missing')
        require(row['illumination'] in ('day','night') and row['thermal_state']=='intact'
                and type(row['supports_frozen_blur']) is bool
                and row['registration'] in ('registered','calibrated'),
                'Invalid capture lighting, state, alignment or blur eligibility')
        timestamp(row['capture_time'])
    by_id={r['id']:r for r in records}
    require(len(by_id)==len(records), 'Duplicate frame ID')
    issues=[]
    for name in ('research','local_derivatives','local_audit'):
        if rights.get(name) is not True:
            issues.append('unverified_right:'+name)
    if not rights.get('terms_url') or not rights.get('review_receipt'):
        issues.append('missing_terms_or_authorized_review')
    if root is None:
        issues.append('original_bytes_not_verified')
    if review.get('complete') is not True or not review.get('receipt') or not review.get('method'):
        issues.append('near_duplicate_review_pending')
    if review.get('historical_scope')!='all_available_mfnet_and_historical_rgbt':
        issues.append('historical_scope_pending')
    if not metadata.get('physical_defocus_receipt'):
        issues.append('physical_defocus_secondary_pending')
    grouped={}
    file_groups={}
    overlap=set(review.get('historical_matches',[]))
    require(overlap<=set(by_id), 'Unknown historical-match frame')
    for r in records:
        g=r['group']
        for key in ('site','sequence'):
            # Use global canonical site/sequence IDs, not export-local names.
            identity=key,r[key]
            if grouped.setdefault(identity,g)!=g:
                issues.append('split_'+key+':'+r[key])
        for arm in ('rgb','thermal'):
            h=file_record(r[arm],root)
            if h in historical:
                overlap.add(r['id'])
            if file_groups.setdefault(h,g)!=g:
                issues.append('cross_group_byte_duplicate:'+r['id'])
    for link in review['internal_links']:
        a,b,reason=link
        require(a in by_id and b in by_id and reason in ('near_duplicate','neighbor','derivative','repeat'),
                'Invalid reviewed link')
        if by_id[a]['group']!=by_id[b]['group']:
            issues.append('cross_group_'+reason+':'+a+':'+b)
    if overlap:
        issues.append('historical_overlap_exclusion_required')
    counts={'groups':len({r['group'] for r in records}),
            **{k:len({r['group'] for r in records if r['illumination']==k}) for k in ('day','night')},
            'positive_blur_groups':len({r['group'] for r in records if r['supports_frozen_blur']})}
    minima={'groups':40,'day':15,'night':15,'positive_blur_groups':15}
    issues.extend('support_below_minimum:'+k for k,v in counts.items() if v<minima[k])
    return {'status':'BLOCKED' if issues else 'READY_FOR_INDEPENDENT_REVIEW','support':counts,
            'required_exclusions':sorted(overlap),'issues':sorted(set(issues)),
            'real_quality_evaluation':'NOT_RUN'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--frozen-parameters',type=Path,required=True)
    p.add_argument('--metadata',type=Path)
    p.add_argument('--data-root',type=Path)
    a=p.parse_args()
    require(a.metadata is not None or a.data_root is None, 'Data root requires a metadata manifest')
    result={'freeze':frozen_parameters(a.frozen_parameters)}
    if a.metadata is not None:
        result['admission']=preflight(read(a.metadata),archive_hashes(),a.data_root)
    print(json.dumps(result,indent=2))
