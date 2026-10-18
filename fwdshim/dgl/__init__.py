"""Minimal pure-PyTorch stand-in for the parts of DGL that ms-pred ICEBERG / GLACIER inference uses.

Own implementation (this repository, MIT) following DGL's documented semantics for homogeneous graphs: graph
construction, node / edge feature frames, batching, induced node subgraphs, custom node reordering, message passing
with the built-in copy_u / copy_e / u_mul_e messages and sum / mean / max reducers (plus user-defined message and
mailbox reduce functions), apply_edges, local_scope and per-graph node readouts.
"""
import contextlib

import torch

from . import function          # noqa: F401  (dgl.function)
from . import backend           # noqa: F401  (dgl.backend.pytorch)
from . import nn                # noqa: F401  (dgl.nn)

NID = '_ID'
EID = '_ID'
ALL = slice(None)


def _long(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.long, device=device if device is not None else x.device)
    return torch.as_tensor(x, dtype=torch.long, device=device)


class _View:
    """Read access to node features gathered at edge endpoints (edges.src / edges.dst)."""

    def __init__(self, frame, idx):
        self._f, self._i = frame, idx

    def __getitem__(self, k):
        return self._f[k][self._i]

    def __contains__(self, k):
        return k in self._f


class EdgeBatch:
    def __init__(self, g, eids):
        self._g, self._eids = g, eids
        src, dst = g._src[eids], g._dst[eids]
        self.src, self.dst = _View(g.ndata, src), _View(g.ndata, dst)
        self.data = _View(g.edata, eids)

    def edges(self):
        return self._g._src[self._eids], self._g._dst[self._eids], self._eids

    def batch_size(self):
        return int(self._eids.numel())


class NodeBatch:
    def __init__(self, g, nodes, mailbox):
        self._g, self._nodes = g, nodes
        self.data = _View(g.ndata, nodes)
        self.mailbox = mailbox

    def nodes(self):
        return self._nodes

    def batch_size(self):
        return int(self._nodes.numel())


class DGLGraph:
    def __init__(self, src, dst, num_nodes, device=None):
        self._src, self._dst = _long(src, device), _long(dst, device)
        self._n = int(num_nodes)
        self.ndata, self.edata = {}, {}
        self._bnn = torch.tensor([self._n], dtype=torch.long, device=self._src.device)
        self._bne = torch.tensor([self._src.numel()], dtype=torch.long, device=self._src.device)

    # --- structure
    @property
    def device(self):
        return self._src.device

    @property
    def idtype(self):
        return torch.long

    @property
    def is_homogeneous(self):
        return True

    @property
    def batch_size(self):
        return int(self._bnn.numel())

    @property
    def srcdata(self):
        return self.ndata

    @property
    def dstdata(self):
        return self.ndata

    def num_nodes(self, ntype=None):
        return self._n

    def num_edges(self, etype=None):
        return int(self._src.numel())

    number_of_nodes = num_nodes
    number_of_edges = num_edges

    def num_src_nodes(self, ntype=None):
        return self._n

    num_dst_nodes = num_src_nodes

    def nodes(self):
        return torch.arange(self._n, device=self.device)

    def edges(self, form='uv', order='eid'):
        if form == 'all':
            return self._src, self._dst, torch.arange(self.num_edges(), device=self.device)
        if form == 'eid':
            return torch.arange(self.num_edges(), device=self.device)
        return self._src, self._dst

    def in_degrees(self, v=None):
        d = torch.bincount(self._dst, minlength=self._n)
        return d if v is None else d[_long(v, self.device)]

    def out_degrees(self, v=None):
        d = torch.bincount(self._src, minlength=self._n)
        return d if v is None else d[_long(v, self.device)]

    def batch_num_nodes(self, ntype=None):
        return self._bnn

    def batch_num_edges(self, etype=None):
        return self._bne

    def set_batch_num_nodes(self, val):
        self._bnn = _long(val, self.device)

    def set_batch_num_edges(self, val):
        self._bne = _long(val, self.device)

    def to(self, device, **kw):
        g = DGLGraph(self._src.to(device), self._dst.to(device), self._n)
        g.ndata = {k: v.to(device) for k, v in self.ndata.items()}
        g.edata = {k: v.to(device) for k, v in self.edata.items()}
        g._bnn, g._bne = self._bnn.to(device), self._bne.to(device)
        return g

    def subgraph(self, nodes, relabel_nodes=True, store_ids=True, output_device=None):
        return node_subgraph(self, nodes, relabel_nodes, store_ids, output_device)

    def int(self):
        return self

    def long(self):
        return self

    @contextlib.contextmanager
    def local_scope(self):
        nd, ed = dict(self.ndata), dict(self.edata)
        try:
            yield
        finally:
            self.ndata, self.edata = nd, ed

    # --- computation
    def _eids(self, edges):
        if edges is None or (isinstance(edges, slice) and edges == ALL):
            return torch.arange(self.num_edges(), device=self.device)
        return _long(edges, self.device).view(-1)

    def apply_edges(self, func, edges=ALL, etype=None):
        eids = self._eids(edges)
        if isinstance(func, function._Message):
            out = {func.out: func.compute(self, eids)}
        else:
            out = func(EdgeBatch(self, eids))
        full = eids.numel() == self.num_edges()
        for k, v in out.items():
            if full:
                self.edata[k] = v
            else:
                if k not in self.edata:
                    self.edata[k] = v.new_zeros((self.num_edges(),) + tuple(v.shape[1:]))
                self.edata[k] = self.edata[k].clone()
                self.edata[k][eids] = v

    def update_all(self, message_func, reduce_func, apply_node_func=None, etype=None):
        eids = torch.arange(self.num_edges(), device=self.device)
        if isinstance(message_func, function._Message):
            msgs = {message_func.out: message_func.compute(self, eids)}
        else:
            msgs = message_func(EdgeBatch(self, eids))
        if isinstance(reduce_func, function._Reduce):
            m = msgs[reduce_func.msg]
            self.ndata[reduce_func.out] = reduce_func.compute(m, self._dst, self._n)
        else:
            self._mailbox_reduce(msgs, reduce_func)
        if apply_node_func is not None:
            self.ndata.update(apply_node_func(NodeBatch(self, torch.arange(self._n, device=self.device), None)))

    def _mailbox_reduce(self, msgs, reduce_func):
        """User-defined reduce: nodes are bucketed by in-degree, each mailbox holds that node's messages in edge-ID
        order; nodes without incoming edges get zeros."""
        deg = self.in_degrees()
        order = torch.argsort(self._dst * (self.num_edges() + 1) + torch.arange(self.num_edges(), device=self.device))
        dst_sorted = self._dst[order]
        start = torch.zeros(self._n + 1, dtype=torch.long, device=self.device)
        start[1:] = torch.cumsum(deg, 0)
        results = {}
        for d in torch.unique(deg).tolist():
            if d == 0:
                continue
            nodes = torch.nonzero(deg == d, as_tuple=True)[0]
            pos = start[nodes].unsqueeze(1) + torch.arange(d, device=self.device).unsqueeze(0)
            e = order[pos]
            mailbox = {k: v[e] for k, v in msgs.items()}
            out = reduce_func(NodeBatch(self, nodes, mailbox))
            for k, v in out.items():
                if k not in results:
                    results[k] = v.new_zeros((self._n,) + tuple(v.shape[1:]))
                results[k][nodes] = v
        for k, v in results.items():
            self.ndata[k] = v


DGLHeteroGraph = DGLGraph


def graph(data, num_nodes=None, idtype=None, device=None, **kw):
    src, dst = data
    src, dst = _long(src, device), _long(dst, device)
    if num_nodes is None:
        num_nodes = int(max(src.max().item() if src.numel() else -1, dst.max().item() if dst.numel() else -1) + 1)
    return DGLGraph(src, dst, num_nodes, device)


def batch(graphs, ndata=ALL, edata=ALL):
    graphs = list(graphs)
    dev = graphs[0].device
    n_off, srcs, dsts, bnn, bne = 0, [], [], [], []
    for g in graphs:
        srcs.append(g._src + n_off); dsts.append(g._dst + n_off)
        n_off += g.num_nodes()
        bnn.append(g._bnn); bne.append(g._bne)
    out = DGLGraph(torch.cat(srcs) if srcs else torch.zeros(0, dtype=torch.long, device=dev),
                   torch.cat(dsts) if dsts else torch.zeros(0, dtype=torch.long, device=dev), n_off)
    out._bnn, out._bne = torch.cat(bnn), torch.cat(bne)
    for k in graphs[0].ndata:
        if all(k in g.ndata for g in graphs):
            out.ndata[k] = torch.cat([g.ndata[k] for g in graphs], 0)
    for k in graphs[0].edata:
        if all(k in g.edata for g in graphs):
            out.edata[k] = torch.cat([g.edata[k] for g in graphs], 0)
    return out


def unbatch(g, node_split=None, edge_split=None):
    bnn = g.batch_num_nodes() if node_split is None else _long(node_split, g.device)
    bne = g.batch_num_edges() if edge_split is None else _long(edge_split, g.device)
    out, n0, e0 = [], 0, 0
    for nn_, ne_ in zip(bnn.tolist(), bne.tolist()):
        sub = DGLGraph(g._src[e0:e0 + ne_] - n0, g._dst[e0:e0 + ne_] - n0, nn_)
        sub.ndata = {k: v[n0:n0 + nn_] for k, v in g.ndata.items()}
        sub.edata = {k: v[e0:e0 + ne_] for k, v in g.edata.items()}
        out.append(sub)
        n0 += nn_; e0 += ne_
    return out


def node_subgraph(g, nodes, relabel_nodes=True, store_ids=True, output_device=None):
    """Induced subgraph on `nodes` (bool mask or ids); nodes keep the given order (ascending for a mask), edges keep
    their original order; batch sizes are recomputed per input graph."""
    if isinstance(nodes, torch.Tensor) and nodes.dtype == torch.bool:
        nid = torch.nonzero(nodes, as_tuple=True)[0]
    else:
        nid = _long(nodes, g.device).view(-1)
    new_id = torch.full((g.num_nodes(),), -1, dtype=torch.long, device=g.device)
    new_id[nid] = torch.arange(nid.numel(), device=g.device)
    keep = (new_id[g._src] >= 0) & (new_id[g._dst] >= 0)
    eid = torch.nonzero(keep, as_tuple=True)[0]
    sub = DGLGraph(new_id[g._src[eid]], new_id[g._dst[eid]], nid.numel())
    sub.ndata = {k: v[nid] for k, v in g.ndata.items()}
    sub.edata = {k: v[eid] for k, v in g.edata.items()}
    if store_ids:
        sub.ndata[NID] = nid
        sub.edata[EID] = eid
    gid_n = torch.repeat_interleave(torch.arange(g.batch_size, device=g.device), g.batch_num_nodes())
    gid_e = torch.repeat_interleave(torch.arange(g.batch_size, device=g.device), g.batch_num_edges())
    sub._bnn = torch.bincount(gid_n[nid], minlength=g.batch_size)
    sub._bne = torch.bincount(gid_e[eid], minlength=g.batch_size)
    return sub


def reorder_graph(g, node_permute_algo=None, edge_permute_algo='src', store_ids=True, permute_config=None):
    """Node order from permute_config['nodes_perm'] (new node i = old node perm[i]) when node_permute_algo is
    'custom'; edges then sorted (stably) by new source ID ('src'), new destination ID ('dst') or kept."""
    n_perm = torch.arange(g.num_nodes(), device=g.device)
    if node_permute_algo == 'custom':
        n_perm = _long(permute_config['nodes_perm'], g.device)
    inv = torch.empty_like(n_perm)
    inv[n_perm] = torch.arange(n_perm.numel(), device=g.device)
    src, dst = inv[g._src], inv[g._dst]
    if edge_permute_algo == 'src':
        e_perm = torch.sort(src, stable=True)[1]
    elif edge_permute_algo == 'dst':
        e_perm = torch.sort(dst, stable=True)[1]
    elif edge_permute_algo == 'custom':
        e_perm = _long(permute_config['edges_perm'], g.device)
    else:
        e_perm = torch.arange(g.num_edges(), device=g.device)
    out = DGLGraph(src[e_perm], dst[e_perm], g.num_nodes())
    out.ndata = {k: v[n_perm] for k, v in g.ndata.items()}
    out.edata = {k: v[e_perm] for k, v in g.edata.items()}
    if store_ids:
        out.ndata[NID] = n_perm
        out.edata[EID] = e_perm
    return out


def _readout(g, feat, weight, op):
    x = g.ndata[feat]
    if weight is not None:
        x = x * g.ndata[weight]
    seg = torch.repeat_interleave(torch.arange(g.batch_size, device=x.device), g.batch_num_nodes())
    out = x.new_zeros((g.batch_size,) + tuple(x.shape[1:]))
    out.index_add_(0, seg, x)
    if op == 'mean':
        out = out / g.batch_num_nodes().clamp(min=1).to(x.dtype).view(-1, *([1] * (x.dim() - 1)))
    return out


def sum_nodes(g, feat, weight=None, op='sum', ntype=None):
    return _readout(g, feat, weight, 'sum')


def mean_nodes(g, feat, weight=None, ntype=None):
    return _readout(g, feat, weight, 'mean')
