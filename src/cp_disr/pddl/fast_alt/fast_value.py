"""FAST: an implementation-level speed-up of the frozen state value V of the Depots models (card C1-DEPOTS-FROZEN-SCORE-FAST-ALT-V3, plan section 7). The scoring function is NOT changed.

The reference path is ``PddlSerialModel.state_values`` -> ``zv`` -> ``B2CachedPolicy._encode`` -> ``_features`` -> phi / goal attention / rho. This module reproduces the same arithmetic with

  A. batch state packing: Python big-int states -> little-endian bytes -> unpacked bit matrix -> proposition TRUE/FALSE tensors (same one-hot codes as ``codes_for_state_general``), built on the device from a bool matrix;
  B. caches that are fixed once the model is in eval mode: the static node embedding (N, 128), the basis-composed RGCN weights of the four layers, and the per-relation batched edge index tensors for a given number of graphs;
  C. an encoder path that returns only the node states H (the reference also computes the old goal / global readout and discards it);
  D. no redundant device synchronisation (the host read-back already waits for the device).
Nothing is approximated: the same graphs, the same message passing (PyG ``RGCNConv`` ``propagate`` per relation), the same layer norms, the same heads, the same physical chunking (48 states per call, then the encoder edge budget).
"""
from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np
import torch
from torch_geometric.nn.conv.rgcn_conv import masked_edge_index

from ..task import atom_id


class FastValue:
    """Cached, batched evaluator of V for one task / template and one frozen model."""

    OUTER_CHUNK = 48                                   # states per physical call, as NeuralEval
    HOST_CACHE_BYTES = 512 * 2 ** 20
    GPU_CACHE_BYTES = 256 * 2 ** 20

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.base = model.mg.base
        self.enc = self.base.encoder
        self.cache_bytes = 0
        self.timers = {}

    # ------------------------------------------------------------------ per-task preparation (inside the problem's clock)
    def prepare(self, task, template):
        t0 = time.perf_counter()
        self.task, self.template = task, template
        m = self.model
        st, gprop = m.mg.goal_free_static(template)
        self.st, self.gprop = st, gprop
        feats = self.enc.features
        with torch.no_grad():
            static = feats.kind.weight[st.kind_idx] + torch.where(st.is_action.unsqueeze(-1), feats.action.weight[st.action_idx], feats.predicate.weight[st.pred_idx])
            static = static + (feats.arg.weight[st.arg_idx] * st.arg_w.unsqueeze(-1)).sum(1)
            self.static = static.contiguous()
            self.weights = [(layer.comp @ layer.weight.view(layer.num_bases, -1)).view(layer.num_relations, layer.in_channels_l, layer.out_channels).contiguous() for layer in self.enc.layers]
        self.R = self.enc.layers[0].num_relations
        # A. packing tables
        id2bit = {atom_id(p, a): i for i, (p, a) in enumerate(task.dyn_atoms)}
        static_true = {atom_id(p, a) for p, a in task.static}
        P = len(st.prop_ids)
        bit = np.full(P, -1, dtype=np.int64)
        is_static = np.zeros(P, dtype=bool)
        for j, pid in enumerate(st.prop_ids):
            if pid in id2bit:
                bit[j] = id2bit[pid]
            elif pid in static_true:
                is_static[j] = True
        self.has_bit = bit >= 0
        self.bit_idx = np.where(self.has_bit, bit, 0)
        self.is_static = is_static
        self.has_bit_t = torch.from_numpy(self.has_bit).to(self.device)
        self.nbytes = max(1, (len(task.dyn_atoms) + 7) // 8)
        self.edge_budget = type(self.model).EDGE_BUDGET if hasattr(type(self.model), "EDGE_BUDGET") else 6_000_000
        self.N, self.E = st.N, int(st.ei.shape[1])
        self._edges = OrderedDict()
        self._edge_bytes = {}
        self.cache_bytes = self.static.numel() * 4 + sum(w.numel() * 4 for w in self.weights)
        self.prepare_seconds = time.perf_counter() - t0
        return self.prepare_seconds

    # ------------------------------------------------------------------ A. packing
    def pack(self, states):
        """(G, P, 3) float codes on the device, equal to ``torch.stack([codes_for_state_general(...)])``."""
        G = len(states)
        buf = b"".join(s.to_bytes(self.nbytes, "little") for s in states)
        bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8).reshape(G, self.nbytes), axis=1, bitorder="little")
        true = (bits[:, self.bit_idx].astype(bool) & self.has_bit) | self.is_static
        t = torch.from_numpy(true).to(self.device)
        codes = torch.zeros((G, true.shape[1], 3), device=self.device)
        codes[..., 0] = t
        codes[..., 1] = ~t
        return codes

    # ------------------------------------------------------------------ B. batched edge cache
    def edges_for(self, G):
        hit = self._edges.get(G)
        if hit is not None:
            self._edges.move_to_end(G)
            return hit
        st = self.st
        offsets = (torch.arange(G, device=self.device) * self.N).repeat_interleave(self.E)
        ei = st.ei.repeat(1, G) + offsets.unsqueeze(0)
        et = st.et.repeat(G)
        per_rel = [masked_edge_index(ei, et == r) for r in range(self.R)]
        nbytes = sum(e.numel() * 8 for e in per_rel)
        while self._edges and self.cache_bytes + nbytes > self.GPU_CACHE_BYTES:
            k, _ = self._edges.popitem(last=False)
            self.cache_bytes -= self._edge_bytes.pop(k)
        self._edges[G] = per_rel
        self._edge_bytes[G] = nbytes
        self.cache_bytes += nbytes
        return per_rel

    # ------------------------------------------------------------------ C. node-state encoder (no readout)
    def encode_nodes(self, codes):
        st, feats = self.st, self.enc.features
        G, N = codes.shape[0], self.N
        dyn = feats.fact(torch.cat((codes, st.prop_goal_sign.unsqueeze(0).expand(G, -1, -1)), dim=-1))
        h = self.static.unsqueeze(0).repeat(G, 1, 1)
        h[:, st.prop_pos] = h[:, st.prop_pos] + dyn
        edges = self.edges_for(G)
        hflat = h.reshape(G * N, -1)
        size = (G * N, G * N)
        for li, (layer, norm) in enumerate(zip(self.enc.layers, self.enc.norms)):
            W = self.weights[li]
            out = torch.zeros(G * N, layer.out_channels, device=self.device)
            for r in range(self.R):
                hh = layer.propagate(edges[r], x=hflat, edge_type_ptr=None, size=size)
                out = out + (hh @ W[r])
            out = out + hflat @ layer.root
            out = out + layer.bias
            hflat = norm(torch.relu(out))
        return hflat.reshape(G, N, -1)

    def value_from_codes(self, codes):
        m = self.model
        H = self.encode_nodes(codes)
        x = m._features(self.st, self.gprop, codes, H)
        z = m.mg.heads.phi(x)
        if m.attn is not None:
            gs = m.attn._goal_static(self.st, self.gprop, self.template)
            f, a = m.attn.pair_features(gs, codes, self.gprop)
            z = z + m.attn(z, f, a)
        return m.mg.heads.rho(z.sum(1)).squeeze(-1)

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def values(self, states):
        """V of arbitrary state bit masks; physical chunking identical to ``NeuralEval`` (48 states) and ``PddlSerialModel.zv`` (encoder edge budget)."""
        out = []
        ch = max(1, self.edge_budget // max(self.E, 1))
        for i in range(0, len(states), self.OUTER_CHUNK):
            chunk = states[i:i + self.OUTER_CHUNK]
            codes = self.pack(chunk)
            if codes.shape[0] <= ch:
                v = self.value_from_codes(codes)
            else:
                v = torch.cat([self.value_from_codes(codes[a:a + ch]) for a in range(0, codes.shape[0], ch)])
            out.append(v)
        v = torch.cat(out) if len(out) > 1 else out[0]
        return v.double().cpu().tolist()                      # the read-back waits for the device: no separate synchronize
