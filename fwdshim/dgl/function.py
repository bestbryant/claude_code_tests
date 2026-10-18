"""dgl.function built-ins: copy_u / copy_e / u_mul_e / u_add_e messages and sum / mean / max / min reducers."""
import torch


class _Message:
    def __init__(self, kind, u=None, e=None, out='m'):
        self.kind, self.u, self.e, self.out = kind, u, e, out

    def compute(self, g, eids):
        if self.kind == 'copy_u':
            return g.ndata[self.u][g._src[eids]]
        if self.kind == 'copy_e':
            return g.edata[self.e][eids]
        a, b = g.ndata[self.u][g._src[eids]], g.edata[self.e][eids]
        # DGL broadcasts the feature dimensions (all but the first) numpy-style, right-aligned
        if a.dim() < b.dim():
            a = a.view(a.shape[0], *([1] * (b.dim() - a.dim())), *a.shape[1:])
        elif b.dim() < a.dim():
            b = b.view(b.shape[0], *([1] * (a.dim() - b.dim())), *b.shape[1:])
        if self.kind == 'u_mul_e':
            return a * b
        if self.kind == 'u_add_e':
            return a + b
        raise ValueError(self.kind)


class _Reduce:
    def __init__(self, op, msg, out):
        self.op, self.msg, self.out = op, msg, out

    def compute(self, m, dst, n):
        shape = (n,) + tuple(m.shape[1:])
        if self.op in ('sum', 'mean'):
            out = m.new_zeros(shape).index_add_(0, dst, m)
            if self.op == 'mean':
                deg = torch.bincount(dst, minlength=n).clamp(min=1).to(m.dtype)
                out = out / deg.view(-1, *([1] * (m.dim() - 1)))
            return out
        idx = dst.view(-1, *([1] * (m.dim() - 1))).expand_as(m)
        red = 'amax' if self.op == 'max' else 'amin'
        out = m.new_zeros(shape).scatter_reduce(0, idx, m, red, include_self=False)
        return out             # nodes without messages stay 0, as in DGL


def copy_u(u, out='m'):
    return _Message('copy_u', u=u, out=out)


copy_src = copy_u


def copy_e(e, out='m'):
    return _Message('copy_e', e=e, out=out)


copy_edge = copy_e


def u_mul_e(lhs_field, rhs_field, out='m'):
    return _Message('u_mul_e', u=lhs_field, e=rhs_field, out=out)


def u_add_e(lhs_field, rhs_field, out='m'):
    return _Message('u_add_e', u=lhs_field, e=rhs_field, out=out)


def sum(msg, out):            # noqa: A001  (DGL's name)
    return _Reduce('sum', msg, out)


def mean(msg, out):
    return _Reduce('mean', msg, out)


def max(msg, out):            # noqa: A001
    return _Reduce('max', msg, out)


def min(msg, out):            # noqa: A001
    return _Reduce('min', msg, out)
