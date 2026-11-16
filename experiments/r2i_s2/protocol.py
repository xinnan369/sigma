"""Fail closed on unapproved seeds and changes to the inherited protocol."""
from experiments.r1.assets import sha256
from experiments.r1g.partition import ROOT, read, require

CONFIG = ROOT / 'configs/R2i_protocol.json'
STAGE = ROOT / 'configs/R2i_S2_protocol.json'
SEEDS = (3408, 3409)
PARENT_SHA = '60c3aefb80343ff95834f3fa2724f9b73d221026fb1b24bbdedd527a8488983d'


def require_seed(seed):
    stage = read(STAGE)
    require(stage == {'version': 'r2i-s2-v1', 'stage': 'S2', 'training_seeds': list(SEEDS),
                      'parent': 'configs/R2i_protocol.json', 'parent_sha256': PARENT_SHA}
            and sha256(CONFIG) == PARENT_SHA, 'R2i-S2 protocol differs')
    require(seed in SEEDS, 'Training seed is not approved for S2')
    return seed
