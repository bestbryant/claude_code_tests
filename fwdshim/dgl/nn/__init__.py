"""dgl.nn readouts used by ms-pred: AvgPooling, SumPooling, GlobalAttentionPooling (softmax gate per graph),
expand_as_pair, and a GraphConv (norm='both') for completeness."""
import torch
import torch.nn as _nn


def expand_as_pair(input_, g=None):
    return input_ if isinstance(input_, tuple) else (input_, input_)


def _segments(graph, device):
    bnn = graph.batch_num_nodes().to(device)
    return bnn, torch.repeat_interleave(torch.arange(bnn.numel(), device=device), bnn)


def _segment_sum(x, seg, n):
    return x.new_zeros((n,) + tuple(x.shape[1:])).index_add_(0, seg, x)


class SumPooling(_nn.Module):
    def forward(self, graph, feat):
        bnn, seg = _segments(graph, feat.device)
        return _segment_sum(feat, seg, bnn.numel())


class AvgPooling(_nn.Module):
    def forward(self, graph, feat):
        bnn, seg = _segments(graph, feat.device)
        s = _segment_sum(feat, seg, bnn.numel())
        return s / bnn.clamp(min=1).to(feat.dtype).view(-1, *([1] * (feat.dim() - 1)))


class GlobalAttentionPooling(_nn.Module):
    def __init__(self, gate_nn, feat_nn=None):
        super().__init__()
        self.gate_nn, self.feat_nn = gate_nn, feat_nn

    def forward(self, graph, feat, get_attention=False):
        bnn, seg = _segments(graph, feat.device)
        gate = self.gate_nn(feat)
        idx = seg.view(-1, *([1] * (gate.dim() - 1))).expand_as(gate)
        gmax = gate.new_zeros((bnn.numel(),) + tuple(gate.shape[1:])).scatter_reduce(0, idx, gate, 'amax',
                                                                                    include_self=False)
        e = torch.exp(gate - gmax[seg])
        gate = e / _segment_sum(e, seg, bnn.numel())[seg]
        x = self.feat_nn(feat) if self.feat_nn is not None else feat
        out = _segment_sum(gate * x, seg, bnn.numel())
        return (out, gate) if get_attention else out


class GraphConv(_nn.Module):
    """Kipf & Welling convolution with symmetric degree normalisation (DGL default norm='both')."""

    def __init__(self, in_feats, out_feats, norm='both', weight=True, bias=True, activation=None,
                 allow_zero_in_degree=False):
        super().__init__()
        self.norm, self.activation = norm, activation
        self.weight = _nn.Parameter(torch.empty(in_feats, out_feats)) if weight else None
        self.bias = _nn.Parameter(torch.zeros(out_feats)) if bias else None
        if self.weight is not None:
            _nn.init.xavier_uniform_(self.weight)

    def forward(self, graph, feat, weight=None, edge_weight=None):
        src, dst = graph.edges()
        n = graph.num_nodes()
        h = feat
        if self.norm in ('both', 'left'):
            dout = torch.bincount(src, minlength=n).clamp(min=1).to(h.dtype)
            h = h * (dout.pow(-0.5) if self.norm == 'both' else 1.0 / dout).view(-1, *([1] * (h.dim() - 1)))
        w = weight if weight is not None else self.weight
        if w is not None:
            h = h @ w
        m = h[src] if edge_weight is None else h[src] * edge_weight.view(-1, *([1] * (h.dim() - 1)))
        out = h.new_zeros((n,) + tuple(h.shape[1:])).index_add_(0, dst, m)
        if self.norm in ('both', 'right'):
            din = torch.bincount(dst, minlength=n).clamp(min=1).to(h.dtype)
            out = out * (din.pow(-0.5) if self.norm == 'both' else 1.0 / din).view(-1, *([1] * (out.dim() - 1)))
        if self.bias is not None:
            out = out + self.bias
        return self.activation(out) if self.activation is not None else out
