"""Bind the numerical gate to the committed protocol and all R2h code."""
import subprocess

from experiments.r1.assets import sha256
from experiments.r1g.partition import ROOT, read, require
from experiments.r2g.journal import checksum, encode
from experiments.r2h.donors import FILE

CONFIG = ROOT / 'configs/R2h_protocol.json'


def current():
    paths = [CONFIG, FILE, *sorted((ROOT / 'experiments/r2h').rglob('*.py'))]
    names = [p.relative_to(ROOT).as_posix() for p in paths]
    git = ['git', '-C', str(ROOT)]
    head = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    for p, name in zip(paths, names, strict=True):
        blob = subprocess.check_output(git + ['show', f'{head}:{name}'])
        require(p.read_bytes() == blob, f'Execution asset differs from HEAD: {name}')
    return {'execution_head': head, 'protocol_sha256': sha256(CONFIG), 'donor_file_sha256': sha256(FILE),
            'r2h_sources': {n: sha256(p) for n, p in zip(names[2:], paths[2:], strict=True)}}


def validate_gate(folder, fingerprint, ids, seed, arms):
    gate = read(folder / 'certificate.json')
    require(gate['phase'] == 'r2h-pilot' and gate['status'] == 'PASS' and gate['fingerprints'] == fingerprint
            and gate['ids'] == ids and {(r['seed'], r['arm']) for r in gate['summary']} == {(seed, a) for a in arms}
            and len(gate['summary']) == len(arms), 'Independent R2h pilot gate failed')
    for item in gate['summary']:
        require(item['count'] == len(ids) and sha256(folder / item['records']) == item['sha256'],
                'Independent R2h pilot evidence differs')
    return sha256(folder / 'certificate.json')


def bind(fingerprint, live, donors):
    sources = {**fingerprint['sources'], **live['r2h_sources']}
    return {**fingerprint, **live, 'sources': sources, 'source_sha256': checksum(encode(sources)),
            'donor_mapping_sha256': donors['mapping_sha256']}
