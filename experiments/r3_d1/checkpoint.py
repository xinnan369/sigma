"""Durable checkpoint transactions. Only load locally produced trusted artifacts."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import torch


def json_bytes(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def hashfile(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def sync_dir(path):
    if os.name!='nt':
        fd=os.open(path,os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic(path,writer):
    fd,name=tempfile.mkstemp(prefix='.writing-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name,path)
        sync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path,value):
    atomic(path,lambda f:f.write(json_bytes(value)))


@contextmanager
def run_lock(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a+b') as handle:
        if os.name=='nt':
            import msvcrt
            if handle.tell()==0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc:
                raise RuntimeError('Experiment already locked') from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Experiment already locked') from exc
        yield  # Descriptor close releases the lock, including after process death.


def rng_state(cuda):
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if cuda else [])


def restore_rng(state,cuda):
    if len(state['cuda'])!=(torch.cuda.device_count() if cuda else 0):
        raise ValueError('CUDA RNG topology differs')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if cuda:
        torch.cuda.set_rng_state_all(state['cuda'])


class Events:
    def __init__(self,path,*,resume=False):
        self.path=path
        self.hasher=hashlib.sha256()
        self.offset=0
        self.stream=path.open('r+b' if resume else 'x+b')

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.stream.close()

    def append(self,event):
        raw=json_bytes(event)
        self.stream.write(raw)
        self.stream.flush()
        self.hasher.update(raw)
        self.offset+=len(raw)

    def commit_receipt(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return dict(offset=self.offset,sha256=self.hasher.hexdigest())

    def restore(self,receipt):
        self.stream.seek(0)
        remaining=receipt['offset']
        if not isinstance(remaining,int) or remaining<0:
            raise ValueError('Invalid log offset')
        h=hashlib.sha256()
        while remaining:
            b=self.stream.read(min(remaining,1024*1024))
            if not b:
                raise ValueError('Committed log prefix missing')
            remaining-=len(b)
            h.update(b)
        if h.hexdigest()!=receipt['sha256']:
            raise ValueError('Committed log prefix digest differs')
        self.offset=receipt['offset']
        self.hasher=h
        self.stream.truncate(self.offset)
        self.stream.flush()
        os.fsync(self.stream.fileno())


class Checkpoints:
    """Ledger atomically commits all refs; public aliases are repairable projections."""
    def __init__(self,root):
        self.root=root
        self.ledger=root/'checkpoint-ledger.json'
        self.objects=root/'checkpoint-objects'
        self.objects.mkdir(exist_ok=True)

    def _path(self,d):
        if not isinstance(d,str) or len(d)!=64 or any(c not in '0123456789abcdef' for c in d):
            raise ValueError('Invalid checkpoint digest')
        return self.objects/(d+'.pt')

    def _read(self):
        return json.loads(self.ledger.read_bytes())

    def _aliases(self,entry):
        for key in ('last','best'):
            dst=self.root/(key+'.pt')
            want=entry[key]
            actual=hashfile(dst) if dst.exists() else None
            if actual==want:
                continue
            # A crash after ledger publication can leave only the prior alias.
            if actual is not None and actual!=entry['previous'][key]:
                raise ValueError(f'Corrupt {key} checkpoint alias')
            if want is not None:
                src=self._path(want)
                with src.open('rb') as f:
                    atomic(dst,lambda out:shutil.copyfileobj(f,out))

    def commit(self,model,optimizer,signature,progress,events,*,best=False):
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError('Checkpoint requires cleared gradients')
        prior=self._read() if self.ledger.exists() else dict(last=None,best=None)
        pending=self.objects/'.pending.pt'
        state=dict(signature=signature,progress=progress,model=model.state_dict(),
                   optimizer=optimizer.state_dict(),rng=rng_state(signature['cuda']),
                   event_log=events.commit_receipt())
        atomic(pending,lambda f:torch.save(state,f))
        h=hashfile(pending)
        os.replace(pending,self._path(h))
        sync_dir(self.objects)
        entry=dict(last=h,best=h if best else prior['best'],
                   previous={k:prior[k] for k in ('last','best')},
                   signature_sha256=digest(signature),update=progress['update'],
                   event_log=state['event_log'])
        write_json(self.ledger,entry)
        self._aliases(entry)
        # Keep the currently committed states. Unreferenced objects are never resumed.
        for p in self.objects.glob('*.pt'):
            if p.name.endswith('.pt') and p.stem not in (entry['last'],entry['best']):
                p.unlink()

    def restore(self,model,optimizer,signature,events,check_progress,check_optimizer=None):
        entry=self._read()
        if entry['signature_sha256']!=digest(signature):
            raise ValueError('Experiment signature differs')
        for h in {entry['last'],entry['best']}-{None}:
            if hashfile(self._path(h))!=h:
                raise ValueError('Checkpoint SHA256 differs')
        # Validate aliases without mutating the log; aliases are never deserialized.
        for name in ('last','best'):
            alias=self.root/(name+'.pt')
            if alias.exists() and hashfile(alias) not in (entry[name],entry['previous'][name]):
                raise ValueError(f'Corrupt {name} checkpoint alias')
        state=torch.load(self._path(entry['last']),map_location='cpu',weights_only=False)
        if (digest(state['signature'])!=digest(signature) or
                state['event_log']!=entry['event_log'] or state['progress']['update']!=entry['update']):
            raise ValueError('Checkpoint metadata differs from ledger')
        check_progress(state['progress'])
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer'])
        if check_optimizer is not None:
            check_optimizer(state['progress'])
        events.restore(state['event_log'])
        self._aliases(entry)
        restore_rng(state['rng'],signature['cuda'])
        return state['progress']
