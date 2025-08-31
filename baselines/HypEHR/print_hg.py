import numpy as np

def summarize_hypergraph(edge_index, num_nodes=None, num_hyperedges=None):
    """
    edge_index: np.ndarray shape [2, nnz], 第一行是节点id，第二行是超边id
    num_nodes: 节点总数（可选，不给就自动取 max(node_id)+1）
    num_hyperedges: 超边总数（可选，不给就自动取 max(edge_id)+1）
    """
    node_ids, hedge_ids = edge_index
    if num_nodes is None:
        num_nodes = int(node_ids.max()) + 1
    if num_hyperedges is None:
        num_hyperedges = int(hedge_ids.max()) + 1
    nnz = edge_index.shape[1]

    density = nnz / (num_nodes * num_hyperedges)

    # 节点度数
    node_deg = np.bincount(node_ids, minlength=num_nodes)
    # 超边大小
    hedge_size = np.bincount(hedge_ids, minlength=num_hyperedges)

    def describe(arr):
        return {
            "avg": np.mean(arr),
            "max": np.max(arr),
            "min": np.min(arr),
            "25": np.percentile(arr, 25),
            "50": np.percentile(arr, 50),
            "75": np.percentile(arr, 75)
        }

    node_stat = describe(node_deg)
    hedge_stat = describe(hedge_size)

    # === 打印 ===
    print("=== Hypergraph Summary ===")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of hyperedges: {num_hyperedges}")
    print(f"Number of non-zero entries (incidences): {nnz}")
    print(f"Density (nnz / (N * E)): {density:.6f}\n")

    print("-- Node Degree --")
    print(f"  Avg: {node_stat['avg']:.2f}")
    print(f"  Max: {node_stat['max']}")
    print(f"  Min: {node_stat['min']}")
    print(f"  25th: {node_stat['25']:.2f}")
    print(f"  50th (Median): {node_stat['50']:.2f}")
    print(f"  75th: {node_stat['75']:.2f}\n")

    print("-- Hyperedge Size --")
    print(f"  Avg: {hedge_stat['avg']:.2f}")
    print(f"  Max: {hedge_stat['max']}")
    print(f"  Min: {hedge_stat['min']}")
    print(f"  25th: {hedge_stat['25']:.2f}")
    print(f"  50th (Median): {hedge_stat['50']:.2f}")
    print(f"  75th: {hedge_stat['75']:.2f}")