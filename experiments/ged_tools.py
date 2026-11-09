"""
GED / NED computation for CI2N netlists.

Uses the dataset's own graph construction (validation/utils_ged.py) and exact GED search
(validation/my_networkx.py), and adds two cheap bounds so every image gets a usable value:

  lower bound   max(label-count bound, "branch" bound). Valid for unit-cost GED: no edit
                path can be cheaper.
  upper bound   cost of the best node mapping found by alternating Hungarian assignment
                (devices <-> nets). This is a real edit path, so GED cannot be higher.

The dataset search is then seeded with the upper bound and only looks for strictly
cheaper paths. GED is exact when the bounds meet or the search finishes before the
timeout; otherwise the true GED lies in [lower, upper].

Cost model (same as utils_ged / networkx): substituting a node or edge with a different
type costs 1, deleting or inserting a node or edge costs 1. Graphs are bipartite
multigraphs: device nodes (ntype "PMOS", "R", ...) and net nodes (ntype "net"), one edge
per device pin, labelled with the pin type (etype "gp2n", "dp2n", ...).

Call load(validation_dir) once (per process) before anything else.
"""

import math
import sys
import time
import types
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

_utils = None  # validation/utils_ged.py, set by load()
_search = None  # validation/my_networkx.my_optimize_edit_paths
_BIG = 1e9


# ----------------------------------------------------------------------------- setup


def load(validation_dir) -> None:
    """Import the dataset's GED code (needs the validation .venv: torch, networkx, ...)."""
    global _utils, _search
    if _utils is not None:
        return
    _install_dgl_shim()
    sys.path.insert(0, str(Path(validation_dir).resolve()))
    import my_networkx
    import utils_ged

    utils_ged._init_logger = lambda *a, **k: utils_ged.FakeLogger()  # silence step logs
    _utils, _search = utils_ged, my_networkx.my_optimize_edit_paths


def _install_dgl_shim() -> None:
    """Stand-in for dgl.heterograph(...).to_networkx() as used by utils_ged.py.

    dgl is not installable on recent macOS. utils_ged only needs a MultiDiGraph with node
    attr 'ntype' and edge attr 'etype' = (src_type, edge_type, dst_type); node counts per
    type are max index + 1, as dgl infers them without num_nodes_dict.
    """

    class _HeteroGraph:
        def __init__(self, graph_data):
            self.edges = {k: (s.tolist(), d.tolist()) for k, (s, d) in graph_data.items()}

        def to_networkx(self):
            n_nodes = Counter()
            for (st, _, dt), (src, dst) in self.edges.items():
                if src:
                    n_nodes[st] = max(n_nodes[st], max(src) + 1)
                if dst:
                    n_nodes[dt] = max(n_nodes[dt], max(dst) + 1)
            g = nx.MultiDiGraph()
            for t, n in n_nodes.items():
                g.add_nodes_from(((t, i) for i in range(n)), ntype=t)
            for canon, (src, dst) in self.edges.items():
                for s, d in zip(src, dst):
                    g.add_edge((canon[0], s), (canon[2], d), etype=canon)
            return g

    sys.modules.setdefault("dgl", types.SimpleNamespace(heterograph=_HeteroGraph))


# ----------------------------------------------------------------------------- graphs


def to_nx(netlist: dict) -> nx.MultiGraph:
    """CI2N JSON netlist -> the undirected multigraph the dataset's GED compares."""
    # utils_ged reads json['ckt_type'] (category label, unused in the graph). Golden files
    # have it; model predictions don't, so default it or every prediction would crash.
    netlist = {"ckt_type": "", **netlist}
    _, het, _ = _utils.HeteroGraph().generate_all_from_json(netlist)
    return _utils.to_MG(het.to_networkx().to_undirected())


class _Graph:
    """Precomputed views of a multigraph used by the bounds."""

    def __init__(self, g: nx.MultiGraph):
        self.nodes = list(g.nodes)
        self.ntype = {n: d["ntype"] for n, d in g.nodes(data=True)}
        self.incident = {n: Counter() for n in self.nodes}  # node -> Counter(edge labels)
        self.adj = {n: {} for n in self.nodes}  # node -> {neighbour: Counter(labels)}
        self.groups = {}  # frozenset({u, v}) -> Counter(labels of parallel edges)
        for u, v, d in g.edges(data=True):
            label = d["etype"]
            self.incident[u][label] += 1
            self.incident[v][label] += 1
            self.adj[u].setdefault(v, Counter())[label] += 1
            self.adj[v].setdefault(u, Counter())[label] += 1
            self.groups.setdefault(frozenset((u, v)), Counter())[label] += 1
        self.n_edges = g.number_of_edges()
        self.nets = [n for n in self.nodes if self.ntype[n] == "net"]
        self.devices = [n for n in self.nodes if self.ntype[n] != "net"]

    def degree(self, n) -> int:
        return sum(self.incident[n].values())


def _multiset_cost(a: Counter, b: Counter) -> int:
    """Min edits turning label multiset a into b (substitute/delete/insert cost 1 each)."""
    return max(sum(a.values()), sum(b.values())) - sum((a & b).values())


def _assign(rows, cols, sub, delete, insert):
    """Optimal assignment with deletions/insertions; returns ({row: col}, total cost)."""
    m, n = len(rows), len(cols)
    if m + n == 0:
        return {}, 0.0
    C = np.zeros((m + n, m + n))
    C[:m, :n] = [[sub(r, c) for c in cols] for r in rows]
    C[:m, n:] = _BIG
    C[m:, :n] = _BIG
    for i, r in enumerate(rows):
        C[i, n + i] = delete(r)
    for j, c in enumerate(cols):
        C[m + j, j] = insert(c)
    ri, ci = linear_sum_assignment(C)
    mapping = {rows[i]: cols[j] for i, j in zip(ri, ci) if i < m and j < n}
    return mapping, float(C[ri, ci].sum())


# ----------------------------------------------------------------------------- bounds


def lower_bound(A: _Graph, B: _Graph) -> int:
    """Valid GED lower bound: max of the label-count bound and the branch bound."""
    # Label counts: node types and edge labels must be fixed independently.
    node_lb = _multiset_cost(Counter(A.ntype.values()), Counter(B.ntype.values()))
    edge_lb = _multiset_cost(sum(A.groups.values(), Counter()), sum(B.groups.values(), Counter()))

    # Branch bound: each node plus half of each incident edge, matched optimally.
    _, branch = _assign(
        A.nodes, B.nodes,
        sub=lambda u, v: (A.ntype[u] != B.ntype[v]) + 0.5 * _multiset_cost(A.incident[u], B.incident[v]),
        delete=lambda u: 1 + 0.5 * A.degree(u),
        insert=lambda v: 1 + 0.5 * B.degree(v),
    )
    return max(node_lb + edge_lb, math.ceil(branch - 1e-9))


def mapping_cost(A: _Graph, B: _Graph, phi: dict) -> int:
    """Exact cost of the cheapest edit path that uses node mapping phi (A -> B, injective).

    Unmapped nodes are deleted/inserted. Parallel edges between mapped node pairs are
    matched optimally by label, exactly as the networkx search does.
    """
    cost = sum(A.ntype[u] != B.ntype[v] for u, v in phi.items())
    cost += (len(A.nodes) - len(phi)) + (len(B.nodes) - len(phi))
    pairs = {}
    for key, labels in A.groups.items():
        u, v = tuple(key)
        target = frozenset((phi[u], phi[v])) if u in phi and v in phi else ("deleted", key)
        pairs.setdefault(target, [Counter(), Counter()])[0].update(labels)
    for key, labels in B.groups.items():
        pairs.setdefault(key, [Counter(), Counter()])[1].update(labels)
    return cost + sum(_multiset_cost(a, b) for a, b in pairs.values())


def _side_step(A, B, rows, cols, other_map):
    """Best mapping of one node class (devices or nets) given the mapping of the other.

    Because the graph is bipartite, every edge has exactly one endpoint in each class, so
    for a fixed mapping of the other class the edge cost decomposes per node pair and this
    assignment is optimal for that side.
    """

    def profile(node):
        return {other_map.get(nb, ("unmapped", nb)): labels for nb, labels in A.adj[node].items()}

    profiles = {r: profile(r) for r in rows}

    def sub(r, c):
        pa, pb = profiles[r], B.adj[c]
        return (A.ntype[r] != B.ntype[c]) + sum(
            _multiset_cost(pa.get(k, Counter()), pb.get(k, Counter())) for k in pa.keys() | pb.keys()
        )

    mapping, _ = _assign(rows, cols, sub,
                         delete=lambda r: 1 + A.degree(r), insert=lambda c: 1 + B.degree(c))
    return mapping


def upper_bound(A: _Graph, B: _Graph, max_rounds: int = 20) -> int:
    """GED upper bound from alternating device/net assignment, two starting points."""
    by_labels = lambda X, Y: (lambda r, c: (X.ntype[r] != Y.ntype[c])
                              + _multiset_cost(X.incident[r], Y.incident[c]))
    best = mapping_cost(A, B, {})  # delete everything, insert everything

    for start in ("nets", "devices"):
        side_rows, side_cols = (A.nets, B.nets) if start == "nets" else (A.devices, B.devices)
        first, _ = _assign(side_rows, side_cols, by_labels(A, B),
                           delete=lambda r: 1 + A.degree(r), insert=lambda c: 1 + B.degree(c))
        net_map = first if start == "nets" else {}
        dev_map = first if start == "devices" else {}
        if start == "devices":
            net_map = _side_step(A, B, A.nets, B.nets, dev_map)
        prev = math.inf
        for _ in range(max_rounds):
            dev_map = _side_step(A, B, A.devices, B.devices, net_map)
            net_map = _side_step(A, B, A.nets, B.nets, dev_map)
            cost = mapping_cost(A, B, {**dev_map, **net_map})
            best = min(best, cost)
            if cost >= prev:
                break
            prev = cost
    return best


# ----------------------------------------------------------------------------- GED


def _strict_edge_match(a: dict, b: dict) -> bool:
    """Multigraph-correct edge match: all parallel edges' labels must agree."""
    return sorted(d["etype"] for d in a.values()) == sorted(d["etype"] for d in b.values())


def compute_ged(pred: dict, gold: dict, timeout: float) -> dict:
    """GED between a predicted and a golden netlist, with bounds.

    Returns dict(ged, ged_lb, exact, method, denom, error, iso_check_quirk, seconds):
      ged        best known GED (upper bound; exact if `exact`)
      ged_lb     proven lower bound
      method     how the value was settled: isomorphic | bounds_meet | search | search_timeout
      denom      devices + nets + ports of golden (NED denominator)
      iso_check_quirk  dataset's own exact-match check says "identical" but the graphs are
                 not (it compares only the first of parallel edges), i.e. calc_ged.py
                 would have counted a false success here
    """
    t0 = time.time()
    error = None
    try:
        g_pred = to_nx(pred)
    except Exception as e:  # malformed prediction the dataset parser can't handle
        error = f"{type(e).__name__}: {e}"
        g_pred = to_nx({"ckt_netlist": []})
    g_gold = to_nx(gold)
    A, B = _Graph(g_pred), _Graph(g_gold)
    denom = len(B.nodes) + B.n_edges
    result = dict(denom=denom, error=error, iso_check_quirk=False)

    if nx.is_isomorphic(g_pred, g_gold, _utils.node_match, _strict_edge_match):
        return {**result, "ged": 0, "ged_lb": 0, "exact": True, "method": "isomorphic",
                "seconds": time.time() - t0}
    result["iso_check_quirk"] = nx.is_isomorphic(g_pred, g_gold, _utils.node_match,
                                                 _utils.edge_match4iso)

    lb, ub = lower_bound(A, B), upper_bound(A, B)
    method = "bounds_meet"
    if lb < ub:
        # Dataset search, pruned to paths strictly cheaper than the heuristic bound.
        s0 = time.time()
        for _, _, cost in _search(g_pred, g_gold, _utils.node_match, _utils.edge_match,
                                  upper_bound=ub - 1, timeout=timeout):
            ub = min(ub, int(cost))
        if time.time() - s0 < timeout * 0.95:  # search completed -> ub is optimal
            lb, method = ub, "search"
        else:
            method = "search_timeout"
    return {**result, "ged": ub, "ged_lb": lb, "exact": lb == ub, "method": method,
            "seconds": time.time() - t0}
