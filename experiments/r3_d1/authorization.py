"""Reviewed authorization only. Empty trust registry intentionally blocks all arms."""
import hashlib
from pathlib import Path
import json

from .checkpoint import digest
from .data import SPLIT_SHA256
from .model import SEEDS

from .review_registry import TRUSTED_RECEIPTS
REQUIRED=frozenset(('A','B','C','D','data_integrity','protocol_signature'))


def require_authorization(path,arm,seed,signature=None):
    if path is None:
        raise ValueError('Formal training blocked for all arms: no reviewed authorization')
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() not in TRUSTED_RECEIPTS:
        raise ValueError('Formal training blocked for all arms: unreviewed authorization')
    receipt=json.loads(raw)
    keys={f'{a}/{s}' for a in 'ABCD' for s in SEEDS}
    if (arm not in 'ABCD' or seed not in SEEDS or receipt.get('schema')!='r3-d1b-r1-authorization-v1'
            or receipt.get('split_sha256')!=SPLIT_SHA256
            or receipt.get('gates')!=dict.fromkeys(REQUIRED,'PASS')
            or set(receipt.get('run_signatures',{}))!=keys):
        raise ValueError('Formal training blocked for all arms: incomplete review gates')
    if signature is not None and receipt['run_signatures'][f'{arm}/{seed}']!=digest(signature):
        raise ValueError('Formal training blocked for all arms: signature mismatch')
