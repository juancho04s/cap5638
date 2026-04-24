"""
Graph / cascade features.

Each post in FakeNewsNet has a retweet tree we read from a JSON file
(node = user, edge = retweet). For each cascade we compute:

  - cascade size            : total nodes
  - cascade depth           : longest path from root
  - max breadth             : max nodes at any depth
  - structural virality     : Wiener index / n(n-1) (Goel et al., 2016)
  - mean out-degree         : average #retweets per user
  - mean clustering coef    : nx.average_clustering
  - propagation speed       : nodes / first-hour duration
  - root degree             : retweets directly from origin

When the cascade JSON is absent (LIAR or missing files), all features
default to zero — graph features then contribute nothing for that row,
which is the correct behavior.
"""

import json
import os

import numpy as np
import networkx as nx

from ..utils import log, Timer


GRAPH_FEATURE_NAMES = [
    "cascade_size", "cascade_depth", "max_breadth", "structural_virality",
    "mean_out_degree", "mean_clustering", "propagation_speed", "root_degree",
]


def _build_graph(cascade_json):
    """cascade_json = list of {user, parent, t}  (parent=None for root)."""
    G = nx.DiGraph()
    root = None
    for node in cascade_json:
        u = node["user"]; p = node.get("parent"); t = node.get("t", 0.0)
        G.add_node(u, t=t)
        if p is None:
            root = u
        else:
            G.add_edge(p, u)
    return G, root


def _features_from_graph(G, root):
    n = G.number_of_nodes()
    if n <= 1 or root is None:
        return np.zeros(len(GRAPH_FEATURE_NAMES), dtype=np.float32)

    # Depth / breadth
    depths = nx.single_source_shortest_path_length(G, root)
    depth = max(depths.values())
    breadth_at = {}
    for d in depths.values():
        breadth_at[d] = breadth_at.get(d, 0) + 1
    max_breadth = max(breadth_at.values())

    # Structural virality (Wiener index / n*(n-1))
    UG = G.to_undirected()
    try:
        wiener = nx.wiener_index(UG)
        struct_vir = wiener / (n * (n - 1))
    except Exception:
        struct_vir = 0.0

    out_deg = np.mean([d for _, d in G.out_degree()])
    clustering = nx.average_clustering(UG)

    # Propagation speed: nodes per hour in the first hour.
    times = [G.nodes[u].get("t", 0.0) for u in G.nodes]
    first_hour = sum(1 for t in times if t <= 3600)
    speed = first_hour / 1.0

    root_deg = G.out_degree(root)

    return np.array([n, depth, max_breadth, struct_vir, out_deg,
                     clustering, speed, root_deg], dtype=np.float32)


def extract_graph_features(data):
    df = data["df"]
    n_rows = len(df)
    X = np.zeros((n_rows, len(GRAPH_FEATURE_NAMES)), dtype=np.float32)
    missing = 0

    with Timer("graph"):
        for i, row in df.iterrows():
            path = row.get("cascade_path")
            if not path or not isinstance(path, str) or not os.path.exists(path):
                missing += 1
                continue
            try:
                with open(path) as f:
                    cascade = json.load(f)
                G, root = _build_graph(cascade)
                X[i] = _features_from_graph(G, root)
            except Exception:
                missing += 1
        log(f"  Graph features shape: {X.shape}  (missing cascades: {missing}/{n_rows})")
    return X
