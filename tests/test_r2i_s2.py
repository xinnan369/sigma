import subprocess
import sys

import pytest
import torch

from experiments.r1d.state import EpochSampler
from experiments.r2i.augment import train_occlusion
from experiments.r2i_s2.driver import verify_history
from experiments.r2i_s2.protocol import require_seed


@pytest.mark.parametrize('seed', [3408, 3409])
def test_five_epochs_deterministic_and_paired(seed):
    ids = [str(i) for i in range(1254)]
    sampler = EpochSampler(1254, require_seed(seed))
    rows = []
    for epoch in range(1, 6):
        for i in range(157):
            names = [ids[i] for i in sampler.next_window(8)]
            rows.append(dict(epoch=epoch, update=len(rows)+1, cursor=sampler.cursor,
                             sample_ids=names, microbatches=len(names), valid_pixel_counts=[1]*len(names)))
        assert len(names) == 6
        if epoch < 5:
            sampler.advance_epoch()
    for _arm in ('baseline', 'gated', 'control'):
        assert verify_history(rows, ids, seed, complete=True).order == sampler.order
    wrong = 3409 if seed == 3408 else 3408
    with pytest.raises(ValueError, match='ordering'):
        verify_history(rows, ids, wrong, complete=True)


def test_seed_and_epoch_aware_augmentation():
    x = torch.ones((1, 3, 480, 640))
    args = x, (.5,)*3, (.25,)*3
    def position(seed, epoch):
        return train_occlusion(*args, seed, '00057D', 1., 64, epoch=epoch)[1]
    assert position(3408, 2) == position(3408, 2)
    assert len({tuple(position(s,e)) for s in (3408,3409) for e in (1,2)}) == 4


@pytest.mark.parametrize('seed', [0,3407,3410,3411])
def test_refuse_unapproved_seed(seed):
    with pytest.raises(ValueError, match='not approved'):
        require_seed(seed)


def test_restore_in_fresh_cpu_process(tmp_path):
    code = """
import sys, torch
from pathlib import Path
from experiments.r1d.state import EpochSampler, save
from experiments.r2i_s2.driver import restart
seed=3409
torch.manual_seed(seed)
model=torch.nn.Linear(2,1)
opt=torch.optim.AdamW(model.parameters(),lr=6e-5)
lr=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
sampler=EpochSampler(1254,seed)
path=Path(sys.argv[1])
signature={'seed':seed}
if sys.argv[2]=='save':
    sampler.next_window(8)
    model(torch.ones(1,2)).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    lr.step()
    save(path,model,opt,lr,sampler,{'updates':1},signature)
else:
    torch.manual_seed(15)
    progress=restart(path,model,opt,lr,sampler,signature)
    assert progress['updates']==1 and sampler.cursor==8 and lr.last_epoch==1
    assert all(p['step'].item()==1 for p in opt.state.values())
"""
    for command in ('save', 'load'):
        subprocess.run([sys.executable, '-c', code, str(tmp_path/'state.pt'), command], check=True)
