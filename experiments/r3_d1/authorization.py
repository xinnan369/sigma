"""Authorize only runs whose receipt and runtime signatures were approved."""
import hashlib
import json
from pathlib import Path

from .checkpoint import digest
from .data import SPLIT_SHA256
from .model import SEEDS
from .review_registry import TRUSTED_RECEIPTS

ARMS = ('A', 'B', 'C', 'D')
REQUIRED = frozenset((*ARMS, 'data_integrity', 'protocol_signature'))
CANDIDATE_RUNS = frozenset(f'{arm}/{seed}' for arm in ARMS for seed in SEEDS)


def require_authorization(path, arm, seed, signature=None):
    """Check scope before any run I/O and the exact runtime signature when available."""
    if path is None:
        raise ValueError('Formal training blocked for all arms: no reviewed authorization')
    raw = Path(path).read_bytes()
    approved_runs = TRUSTED_RECEIPTS.get(hashlib.sha256(raw).hexdigest())
    if approved_runs is None:
        raise ValueError('Formal training blocked for all arms: unreviewed authorization')
    receipt = json.loads(raw)
    run_signatures = receipt.get('run_signatures')
    if (arm not in ARMS or seed not in SEEDS or
            receipt.get('schema') != 'r3-d1b-r1-authorization-v1' or
            receipt.get('split_sha256') != SPLIT_SHA256 or
            receipt.get('gates') != dict.fromkeys(REQUIRED, 'PASS') or
            not isinstance(run_signatures, dict) or run_signatures.keys() != CANDIDATE_RUNS):
        raise ValueError('Formal training blocked for all arms: incomplete review gates')

    run_key = f'{arm}/{seed}'
    approved = approved_runs.get(run_key)
    if approved is None:
        raise ValueError(f'Formal training blocked: {run_key} is outside the approved run scope')
    if (run_signatures[run_key] != approved.receipt_signature_sha256 or
            (signature is not None and digest(signature) != approved.runtime_signature_sha256)):
        raise ValueError('Formal training blocked for all arms: signature mismatch')
