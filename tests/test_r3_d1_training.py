import hashlib
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from experiments.r3_d1.checkpoint import Checkpoints, Events, rng_state, run_lock
from experiments.r3_d1.data import flipped
from experiments.r3_d1.evaluate import improves, validate
from experiments.r3_d1.model import Detector, SEEDS
from experiments.r3_d1.plan import WIDTHS, optimizer
from experiments.r3_d1.resume_audit import compare
from experiments.r3_d1.train import (batches, new_progress, optimizer_cursor,
                                    reproducible, verify_progress)


def roster():
    return tuple(f'{i:06d}' for i in range(9926))


def test_complete_paired_orders_all_seed_epochs():
    ids=roster()
    for seed in SEEDS:
        for e in range(10):
            plan=batches(seed,e,ids)
            assert plan==batches(seed,e,ids)
            assert [len(i) for i in plan]==[4]*2481+[2]
            assert sorted(i for b in plan for i in b)==list(ids)
            expected=[hashlib.sha256(f'{seed}/{e}/{i}/flip'.encode()).digest()[0]<128 for i in ids]
            assert [flipped(i,'fit',seed,e) for i in ids]==expected
    assert not flipped(ids[0],'validation',3407,0)
    assert any(flipped(i,'fit',3407,0)!=flipped(i,'fit',3407,1) for i in ids[:50])


def test_cursor_rejects_changed_order_and_handles_epoch_boundary():
    ids=roster()
    p=new_progress(3407,'D',0,ids)
    p.update(cursor=2482,update=2482,images=9926)
    verify_progress(p,3407,'D',ids)
    p['order'][0],p['order'][1]=p['order'][1],p['order'][0]
    with pytest.raises(ValueError,match='Sample order'):
        verify_progress(p,3407,'D',ids)
    assert improves(dict(ap50=.7,ap50_95=.5),0,dict(epoch=1,ap50=.7,ap50_95=.5))
    assert not improves(dict(ap50=.7,ap50_95=.5),2,dict(epoch=1,ap50=.7,ap50_95=.5))


class FakeUpstream:
    # CPU-only construction placeholder for optimizer grouping; no scan claims.
    class CrossMambaFusionBlock(nn.Conv2d):
        def __init__(self,**kwargs):
            super().__init__(64,64,1)
    ConcatMambaFusionBlock=CrossMambaFusionBlock


@pytest.mark.parametrize('arm',list('ABCD'))
def test_optimizer_covers_every_tensor_once(arm):
    m=Detector(arm,widths=WIDTHS,upstream=FakeUpstream)
    opt=optimizer(m)
    actual=[id(p) for g in opt.param_groups for p in g['params']]
    assert sorted(actual)==sorted(id(p) for p in m.parameters())
    optimizer_cursor(opt,m,0)


SIG={'cuda':False,'arm':'synthetic','seed':3407}


def build():
    torch.manual_seed(3407)
    m=nn.Sequential(nn.Conv2d(2,2,1),nn.BatchNorm2d(2))
    return m,optimizer(m)


def step(m,opt):
    x=torch.rand(2,2,4,4)+np.random.random()+random.random()
    m(x).square().mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_checkpoint_restores_two_step_trajectory_and_truncates_tail(tmp_path):
    reproducible(3407,False)
    control,copt=build()
    step(control,copt)
    step(control,copt)
    expected=rng_state(False)
    reproducible(3407,False)
    m,opt=build()
    ck=Checkpoints(tmp_path)
    with Events(tmp_path/'events.jsonl') as log:
        step(m,opt)
        log.append({'update':1})
        ck.commit(m,opt,SIG,{'update':1},log,best=True)
        log.append({'uncommitted':True})
    m,opt=build()
    with Events(tmp_path/'events.jsonl',resume=True) as log:
        p=ck.restore(m,opt,SIG,log,lambda p: p['update']==1)
        optimizer_cursor(opt,m,p['update'])
        step(m,opt)
        log.append({'update':2})
        ck.commit(m,opt,SIG,{'update':2},log)
    assert (tmp_path/'events.jsonl').read_bytes()==b'{"update":1}\n{"update":2}\n'
    compare(control.state_dict(),m.state_dict())
    compare(copt.state_dict(),opt.state_dict())
    compare(expected,rng_state(False))
    optimizer_cursor(opt,m,2)
    ledger=json.loads((tmp_path/'checkpoint-ledger.json').read_bytes())
    assert ledger['best']!=ledger['last']


@pytest.mark.parametrize('fault', ['signature','checkpoint','log'])
def test_invalid_restore_rejected_without_truncating(fault,tmp_path):
    m,opt=build()
    ck=Checkpoints(tmp_path)
    p=tmp_path/'events.jsonl'
    with Events(p) as log:
        step(m,opt)
        log.append({'update':1})
        ck.commit(m,opt,SIG,{'update':1},log)
        log.append({'tail':True})
    sig=dict(SIG)
    if fault=='signature':
        sig['seed']=3408
    if fault=='checkpoint':
        obj=ck._path(ck._read()['last'])
        obj.write_bytes(obj.read_bytes()+b'!')
    if fault=='log':
        p.write_bytes(b'x'+p.read_bytes()[1:])
    before=p.read_bytes()
    with Events(p,resume=True) as log,pytest.raises(ValueError,match='differs'):
        ck.restore(m,opt,sig,log,lambda _:None)
    assert p.read_bytes()==before


def test_lock_forbids_concurrent_writer(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError,match='locked'),run_lock(tmp_path):
            pass
    with run_lock(tmp_path):
        pass


def test_validation_reads_only_val_and_keeps_unmatched_gt(monkeypatch):
    from experiments.r3_d1 import evaluate
    calls=[]
    class Data:
        def ids(self,role):
            assert role=='validation'
            return ['000001']
        def read_pair(self,i,role):
            calls.append(('pair',i,role))
            return torch.zeros(3,1,1),torch.ones(3,1,1),None
        def read_boxes(self,i,role):
            calls.append(('boxes',i,role))
            return np.array([[0,0,2,2],[4,4,6,6]],np.float32)
    class Model(nn.Module):
        arm='A'
        def forward(self,rgb,thermal):
            assert rgb.shape==(1,3,1,1) and thermal is None
            return [torch.zeros(1,6,1,1)]
    monkeypatch.setattr(evaluate,'decode',lambda _: [dict(boxes=[[0,0,2,2]],scores=[.9])])
    r=validate(Model(),Data(),'cpu')
    assert r['gt']==2 and r['images']==r['predictions']==1
    assert 0<r['ap50_95']<1 and len(r['predictions_sha256'])==64
    assert calls==[('pair','000001','validation'),('boxes','000001','validation')]


def test_ledger_recovers_aliases_after_interrupted_publication(tmp_path,monkeypatch):
    m,opt=build()
    ck=Checkpoints(tmp_path)
    with Events(tmp_path/'events.jsonl') as log:
        step(m,opt)
        log.append({'update':1})
        ck.commit(m,opt,SIG,{'update':1},log,best=True)
        originals=ck._aliases
        def interrupted(*args):
            raise RuntimeError('simulated process loss after ledger rename')
        with monkeypatch.context() as patch:
            patch.setattr(ck,'_aliases',interrupted)
            step(m,opt)
            log.append({'update':2})
            with pytest.raises(RuntimeError,match='simulated'):
                ck.commit(m,opt,SIG,{'update':2},log)
    entry=ck._read()
    from experiments.r3_d1.checkpoint import hashfile
    assert hashfile(tmp_path/'last.pt')==entry['previous']['last']
    assert hashfile(tmp_path/'best.pt')==entry['best']
    with Events(tmp_path/'events.jsonl',resume=True) as log:
        p=ck.restore(m,opt,SIG,log,lambda _:None)
        optimizer_cursor(opt,m,p['update'])
    assert hashfile(tmp_path/'last.pt')==entry['last']
    assert hashfile(tmp_path/'best.pt')==entry['best']
