"""Atomic training state, including the shuffled sampler and CPU noise RNG."""
from pathlib import Path
import hashlib
import json
import torch


def identity(ds, cfg, mean=None, std=None, static=None):
    # Epoch count may be increased when resuming, but data/model/loss must match.
    settings = {k: v for k, v in cfg['training'].items()
                if k not in ('epochs', 'resume', 'checkpoint_every_steps', 'patience', 'min_epochs')}
    digest = hashlib.sha256(json.dumps({**cfg, 'training': settings}, sort_keys=True).encode())
    for name in ('time', 'lat', 'lon', 'variable'):
        digest.update(str(ds[name].values.tolist()).encode())
    for name in ('predictors', 'target'):
        for start in range(0, len(ds.time), 24):
            digest.update(ds[name].isel(time=slice(start, start + 24)).values.tobytes())
    for array in (mean, std, static):
        if array is not None:
            digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def save_resume(path, net, opt, rng, fingerprint, epoch, cursor, windows,
                losses, history, best, stale):
    state = dict(fingerprint=fingerprint, epoch=epoch, cursor=cursor,
                 windows=list(map(int, windows)), losses=losses, history=history,
                 best=best, stale=stale, rng=rng.bit_generator.state,
                 torch_rng=torch.get_rng_state(),
                 state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
                 optimizer=opt.state_dict())
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)
