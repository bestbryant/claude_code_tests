"""torch_scatter subset (own implementation on torch.scatter_reduce): scatter / scatter_add / scatter_sum /
scatter_mean / scatter_max / scatter_min / scatter_softmax. Positions that receive nothing are 0 (and their arg
index is src.size(dim)), as in torch_scatter."""
import torch


def _broadcast(index, src, dim):
    dim = dim % src.dim()
    if index.dim() == 1:
        shape = [1] * src.dim(); shape[dim] = -1
        index = index.view(shape)
    for _ in range(index.dim(), src.dim()):
        index = index.unsqueeze(-1)
    return index.expand_as(src), dim


def _out(src, index, dim, dim_size):
    size = list(src.shape)
    size[dim] = int(dim_size) if dim_size is not None else (int(index.max().item()) + 1 if index.numel() else 0)
    return src.new_zeros(size)


def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    index, dim = _broadcast(index, src, dim)
    o = _out(src, index, dim, dim_size) if out is None else out
    return o.scatter_add_(dim, index, src)


scatter_add = scatter_sum


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    s = scatter_sum(src, index, dim, out, dim_size)
    cnt = scatter_sum(torch.ones_like(src), index, dim, None, s.shape[dim % src.dim()])
    return s / cnt.clamp(min=1) if s.is_floating_point() else torch.div(s, cnt.clamp(min=1), rounding_mode='floor')


def _scatter_ext(src, index, dim, out, dim_size, red):
    index, dim = _broadcast(index, src, dim)
    o = _out(src, index, dim, dim_size) if out is None else out
    o = o.scatter_reduce(dim, index, src, red, include_self=False)
    # arg: position along dim of the element that attains the extreme (largest such position)
    pos = torch.arange(src.shape[dim], device=src.device).view([-1 if d == dim else 1 for d in range(src.dim())])
    pos = pos.expand_as(src)
    hit = src == o.gather(dim, index)
    cand = torch.where(hit, pos, torch.full_like(pos, -1))
    arg = torch.full(o.shape, -1, dtype=torch.long, device=src.device).scatter_reduce(dim, index, cand, 'amax',
                                                                                     include_self=True)
    arg[arg < 0] = src.shape[dim]
    return o, arg


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    return _scatter_ext(src, index, dim, out, dim_size, 'amax')


def scatter_min(src, index, dim=-1, out=None, dim_size=None):
    return _scatter_ext(src, index, dim, out, dim_size, 'amin')


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce='sum'):
    if reduce in ('sum', 'add'):
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    if reduce == 'max':
        return scatter_max(src, index, dim, out, dim_size)[0]
    if reduce == 'min':
        return scatter_min(src, index, dim, out, dim_size)[0]
    raise ValueError(reduce)


def scatter_softmax(src, index, dim=-1, eps=0.0, dim_size=None):
    mx = scatter_max(src, index, dim, dim_size=dim_size)[0]
    idx, d = _broadcast(index, src, dim)
    e = (src - mx.gather(d, idx)).exp()
    s = scatter_sum(e, index, dim, dim_size=dim_size) + eps
    return e / s.gather(d, idx)
