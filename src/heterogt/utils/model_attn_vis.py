from typing import Dict, Iterable, List, Optional, Tuple, Sequence, Union
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from heterogt.utils.train import PHENO_ORDER
import torch.nn.functional as F

node_color ={
    "[SEQ]": "#FF0000",
    "diag":  "#6AC2EE",
    "med": "#FA822C",
    "lab": "#58F43C",
    "pro": "#F080EE", 
    "group": "#5B03FF",
    "visit": "#0004FF"  
}


class SampleVis():
    def __init__(self, gnn_plot_set, seq_plot_set, onc_plot_set, agg_plot_set, tokenizer, model):
        self.gnn_plot_set = gnn_plot_set
        self.seq_plot_set = seq_plot_set
        self.onc_plot_set = onc_plot_set
        self.agg_plot_set = agg_plot_set
        self.model = model
        self.tokenizer = tokenizer
    
    def visualize(self, input_ids, token_types, adm_index, age_ids, diag_code_group_dicts, labels, sample_idx):
        logits, pipeline_out = self.model(input_ids, token_types, adm_index, age_ids, diag_code_group_dicts)
        graph_batch = pipeline_out["graph_batch"]
        gnn_attn_probs = pipeline_out["gnn_attn"]
        tf_attn_probs = pipeline_out["tf_attn"]
        agg_attn_prob = pipeline_out["agg_attn"]
        agg_attn_prob = SampleVis.adjust_agg_attn_prob(agg_attn_prob)
        token_types_full = pipeline_out["token_types_full"]
        adm_index_full = pipeline_out["adm_index_full"]
        gnn_diag_node_names = SampleVis.get_gnn_diag_node_names(input_ids, token_types, sample_idx, self.tokenizer)
        attn_mtx_node_names = SampleVis.get_tf_mtx_node_names(input_ids, token_types_full, sample_idx, self.tokenizer)
        attn_mtx_node_names[0] = "[SEQ]" # switch to [SEQ]
        
        # plot gnn
        SampleVis.visualize_gnn(graph_batch, gnn_attn_probs, sample_idx, occ_labels=gnn_diag_node_names,
                                visit_spacing_x = self.gnn_plot_set["visit_spacing_x"], 
                                occ_ring_radius = self.gnn_plot_set["occ_ring_radius"], 
                                occ_label_offset = self.gnn_plot_set["occ_label_offset"],
                                figure_size = self.gnn_plot_set["figure_size"], 
                                attn_score_fontsize=self.gnn_plot_set["attn_score_fontsize"], 
                                occ_label_fontsize=self.gnn_plot_set["occ_label_fontsize"],
                                min_edge_width= self.gnn_plot_set["min_edge_width"], 
                                max_edge_width = self.gnn_plot_set["max_edge_width"])
        
        # plot seq
        SampleVis.visualize_tf_attn(tf_attn_probs[sample_idx],
                                    query_token_idx=[0],
                                    topk=self.seq_plot_set["topk"],
                                    figure_size = self.seq_plot_set["figure_size"],
                                    col_gap=self.seq_plot_set["col_gap"],
                                    row_gap=self.seq_plot_set["row_gap"],
                                    query_labels=attn_mtx_node_names,
                                    key_labels=attn_mtx_node_names)
        
        # plot onc
        onc_idx = torch.where(token_types_full[sample_idx] == 6)[0]   # Tensor of indices
        onc_idx_list = onc_idx.tolist()
        if len(onc_idx_list) > 0:
            SampleVis.visualize_tf_attn(tf_attn_probs[sample_idx], 
                                        query_token_idx=onc_idx_list,
                                        topk = self.onc_plot_set["topk"],
                                        figure_size = self.onc_plot_set["figure_size"],
                                        col_gap = self.onc_plot_set["col_gap"],
                                        row_gap = self.onc_plot_set["row_gap"],
                                        query_labels = attn_mtx_node_names,
                                        key_labels = attn_mtx_node_names)
        else:
            print("No group code!")
        
        # plot agg
        SampleVis.visualize_tf_attn(agg_attn_prob[sample_idx],
                                    query_token_idx=[0],
                                    topk=self.agg_plot_set["topk"],
                                    figure_size = self.agg_plot_set["figure_size"],
                                    col_gap=self.agg_plot_set["col_gap"],
                                    row_gap=self.agg_plot_set["row_gap"],
                                    query_labels=['[SEQ]'],
                                    key_labels=attn_mtx_node_names)


        print("Next admission diseases:")
        for i, binary_label in enumerate(labels[sample_idx]):
            if binary_label == 1:
                print(PHENO_ORDER[i])

    @staticmethod
    def get_tf_mtx_node_names(input_ids, token_types_full, sample_idx, tokenizer):
        main_seq = input_ids[sample_idx].tolist()
        visit_seq_temp = token_types_full[sample_idx].tolist()
        visit_seq_start = visit_seq_temp.index(7)
        visit_seq = visit_seq_temp[visit_seq_start:]
        main_seq_names = tokenizer.convert_ids_to_tokens(main_seq)
        visit_seq_names = ["V" + str(i + 1) if visit_seq[i] == 7 else "[PAD]" for i in range(len(visit_seq)) ]
        final_names= main_seq_names + visit_seq_names
        assert len(final_names) == len(visit_seq_temp) 
        return final_names
    
    @staticmethod
    def get_gnn_diag_node_names(input_ids, token_types, sample_idx, tokenizer):
        diag_ids = input_ids[sample_idx][token_types[sample_idx] == 2].tolist()
        diag_node_names = tokenizer.convert_ids_to_tokens(diag_ids)
        return diag_node_names

    @staticmethod
    def visualize_tf_attn(
        A: Union[np.ndarray, torch.Tensor],
        query_token_idx: Union[int, Sequence[int]] = 0,  # 支持 int 或 list[int]
        topk: Optional[int] = 10,
        threshold: Optional[float] = None,
        remove_self_loop: bool = True,
        width_scale: float = 6.0,
        show_edge_label: bool = True,
        node_prefix_q: str = "q",
        node_prefix_k: str = "k",
        query_labels: Optional[List[str]] = None,  # query tokens 的名字列表，长度为 M
        key_labels: Optional[List[str]] = None,    # key tokens 的名字列表，长度为 N
        col_gap: float = 1.0,     # 上(Queries)与下(Keys)两行的纵向间距
        row_gap: float = 1.0,     # 节点横向间距
        figure_size: Tuple[int, int] = (12, 8),
        label_offset: float = 0.35,
        label_fontsize: float = 10.0,
        attn_score_fontsize: float = 8,
        # —— 新增：末尾按标签规则重着色 —— #
    ):
        """
        将 M×N 注意力矩阵画成严格两行：上 Query(s)，下 Key(s)。
        - 在最后一步，不改变布局与其他绘制，按节点“标签名/ID”重着色：
            * 等于 "[SEQ]"            -> node_color["[SEQ]"]
            * 以 "DIAG_" 开头        -> node_color["diag"]
            * 以 "MED_"  开头        -> node_color["med"]
            * 以 "LAB_"  开头        -> node_color["lab"]
            * 以 "PRO_"  开头        -> node_color["pro"]
            * 标签中包含 "-"         -> node_color["group"]
            * 属于 {"V1",...,"V8"}  -> node_color["visit"]
        （上述判断顺序与需求一致；只有匹配时才覆盖原色）
        """
        # --- to numpy ---
        if isinstance(A, torch.Tensor):
            A = A.detach().cpu().float().numpy()
        else:
            A = np.asarray(A, dtype=np.float32)
        if A.ndim != 2:
            raise ValueError(f"A 必须是二维矩阵，当前维度: {A.ndim}")

        M, N = A.shape  # M queries, N keys

        # --- 验证标签长度 ---
        if query_labels is not None and len(query_labels) != M:
            raise ValueError(f"query_labels 长度 {len(query_labels)} 与 query 数量 M={M} 不匹配")
        if key_labels is not None and len(key_labels) != N:
            raise ValueError(f"key_labels 长度 {len(key_labels)} 与 key 数量 N={N} 不匹配")

        # --- 规范 query_token_idx 为列表 ---
        if isinstance(query_token_idx, int):
            query_list = [query_token_idx]
        else:
            query_list = list(query_token_idx)
        if len(query_list) == 0:
            raise ValueError("query_token_idx 为空。请至少指定一个 query 索引。")
        for qi in query_list:
            if not (0 <= qi < M):
                raise ValueError(f"query_token_idx 中存在越界值 {qi}，query 数量 M={M}")

        # --- 为每个 query 行裁剪/筛选 ---
        active_keys = set()
        row_weights = {}  # (q_idx -> dict[key_j -> w])
        for qi in query_list:
            row = A[qi].copy()  # shape [N]

            # 移除自环（仅方阵且索引有效）
            if remove_self_loop and M == N and qi < N:
                row[qi] = 0.0

            # top-k
            if topk is not None and topk > 0:
                k_eff = min(topk, N)
                cols = np.argpartition(-row, kth=k_eff-1)[:k_eff]
                mask = np.zeros(N, dtype=bool)
                mask[cols] = True
                row = np.where(mask, row, 0.0)

            # threshold
            if threshold is not None:
                row = np.where(row > threshold, row, 0.0)

            # 过滤极小值
            eps = 1e-6
            nz = np.nonzero(row > eps)[0].tolist()
            weights = {j: float(row[j]) for j in nz}
            row_weights[qi] = weights
            active_keys.update(weights.keys())

        # 无边时直接返回空图
        if len(active_keys) == 0:
            plt.figure(figsize=(6, 3))
            plt.axis("off")
            plt.show()
            return

        # --- 构图 ---
        G = nx.DiGraph()

        # Queries（上排）- 使用自定义标签或默认前缀
        q_nodes = []
        for qi in query_list:
            node_id = query_labels[qi] if query_labels is not None else f"{node_prefix_q}{qi}"
            q_nodes.append(node_id)
            G.add_node(node_id, role="query")

        # Keys（下排，按原始索引升序）- 使用自定义标签或默认前缀
        active_keys_sorted = sorted(active_keys)
        k_nodes = []
        for j in active_keys_sorted:
            node_id = key_labels[j] if key_labels is not None else f"{node_prefix_k}{j}"
            k_nodes.append(node_id)
            G.add_node(node_id, role="key")

        # 边
        for qi in query_list:
            q_node_id = query_labels[qi] if query_labels is not None else f"{node_prefix_q}{qi}"
            for j, w in row_weights[qi].items():
                k_node_id = key_labels[j] if key_labels is not None else f"{node_prefix_k}{j}"
                G.add_edge(q_node_id, k_node_id, weight=w)

        # --- 固定两行坐标 ---
        pos = {}
        # Queries（上排）
        if len(q_nodes) == 1:
            pos[q_nodes[0]] = (0.0, col_gap)
        else:
            for i, qn in enumerate(q_nodes):
                pos[qn] = (i * row_gap, col_gap)
        # Keys（下排）
        for i, kn in enumerate(k_nodes):
            pos[kn] = (i * row_gap, 0.0)

        # --- 绘制（初始配色；稍后覆盖） ---
        plt.figure(figsize=figure_size)

        # 初始颜色：query 蓝绿区分
        node_colors = ["#90caf9" if G.nodes[n]["role"] == "query" else "#a5d6a7" for n in G.nodes()]
        node_sizes  = [600 for _ in G.nodes()]  # 固定大小（如需参数化可外提）

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=1.0)

        # 标签：Query 上方 / Key 下方
        query_label_pos = {n: (pos[n][0], pos[n][1] + label_offset) for n in q_nodes}
        nx.draw_networkx_labels(G, query_label_pos, labels={n: n for n in q_nodes}, font_size=label_fontsize)

        key_label_pos = {n: (pos[n][0], pos[n][1] - label_offset) for n in k_nodes}
        nx.draw_networkx_labels(G, key_label_pos, labels={n: n for n in k_nodes}, font_size=label_fontsize)

        # 边与权重标签
        widths = [d.get("weight", 0.0) * width_scale for _, _, d in G.edges(data=True)]
        nx.draw_networkx_edges(
            G, pos,
            arrows=True,
            width=widths,
            alpha=0.9,
            edge_color="gray",
            arrowsize=20,
            connectionstyle="arc3,rad=0.0"
        )
        if show_edge_label:
            edge_labels = {(u, v): f"{d.get('weight', 0.0):.3f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=attn_score_fontsize)

        plt.axis("off")
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.tight_layout()

        # ===========================================================
        # 末尾重着色：不改变布局/边/标签，仅覆盖节点颜色
        # ===========================================================

        # 先复制当前颜色，匹配则覆盖
        nodes_list = list(G.nodes())
        recolors = list(node_colors)  # 与 nodes_list 对应
        visit_name_set = {"V1","V2","V3","V4","V5","V6","V7","V8"}

        for idx, n in enumerate(nodes_list):
            label = str(n)

            # 规则按给定顺序依次判断
            if label == "[SEQ]" and "[SEQ]" in node_color:
                recolors[idx] = node_color["[SEQ]"]
                continue

            if label.startswith("DIAG_") and "diag" in node_color:
                recolors[idx] = node_color["diag"]
                continue

            if label.startswith("MED_") and "med" in node_color:
                recolors[idx] = node_color["med"]
                continue

            if label.startswith("LAB_") and "lab" in node_color:
                recolors[idx] = node_color["lab"]
                continue

            if label.startswith("PRO_") and "pro" in node_color:
                recolors[idx] = node_color["pro"]
                continue

            if "-" in label and "group" in node_color:
                recolors[idx] = node_color["group"]
                continue

            if label in visit_name_set and "visit" in node_color:
                recolors[idx] = node_color["visit"]
                continue

        # 用相同坐标/尺寸再绘一遍节点以覆盖原色（不改布局与边/标签）
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=recolors, alpha=1.0)

        # 最终展示
        plt.show()

    @staticmethod
    def visualize_gnn(
        hg_batch,
        attn_dict: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
        patient_idx: int,
        head: Optional[str] = "mean",               # "mean" | "first" | None
        rels_to_draw: Optional[Iterable[Tuple[str, str, str]]] = None,
        layout_seed: int = 42,
        # ---- 外观参数 ----
        visit_node_size: int = 1800,
        occ_node_size: int = 600,
        occ_labels: Optional[List[str]] = None,     # occurrence (基) 节点标签，长度应与本病人 occ 数一致
        # ---- 布局参数（全面优化）----
        visits_per_row: int = 8,                    # 每行最多的 visit 数
        visit_spacing_x: float = 8.0,               # 同一行 visit 的横向间距（增大）
        visit_spacing_y: float = 6.0,               # 不同行之间的纵向间距（增大）
        occ_ring_radius: float = 3.5,               # 固定的环形半径
        occ_label_offset: float = 0.8,              # occurrence节点标签向外偏移距离
        y_stretch: Optional[float] = None,          # 若提供，对整体 y 方向额外拉伸（>1 拉高）
        min_edge_width: float = 0.8,                # 最小边宽
        max_edge_width: float = 1.5,                # 最大边宽
        figure_size: Tuple[int, int] = (12, 8),
        occ_label_fontsize: int = 10,                # occurrence节点标签字体大小
        attn_score_fontsize: int = 10,         # 注意力分数标签字体大小
    ):
        """
        从一个批 (HeteroBatch) 中选择第 patient_idx 个样本并可视化注意力。
        本版本优化了布局和视觉效果：
        1. 固定的圆环半径
        2. 可调节visit节点间距
        3. 基于注意力分数的边宽度
        4. occurrence节点标签显示在节点外面
        
        优化参数说明：
        - occ_ring_radius: 每个visit周围occurrence节点的固定环形半径
        - visit_spacing_x/y: visit节点之间的间距（横向/纵向）
        - occ_label_offset: occurrence节点标签向外偏移的距离
        - min/max_edge_width: 边宽度的范围，由注意力分数映射
        """
        # ---------- 1) 拆 batch -> 单病人 ----------
        hg_list = hg_batch.to_data_list()
        assert 0 <= patient_idx < len(hg_list), f"patient_idx 超出范围 [0, {len(hg_list)-1}]"
        hg_p = hg_list[patient_idx]

        # ---------- 2) 计算各类型节点数与偏移 ----------
        node_types = list(hg_batch.node_types)  # 例如 ["visit", "occ"]
        counts_per_patient = {
            nt: [g[nt].num_nodes if nt in g.node_types else 0 for g in hg_list]
            for nt in node_types
        }
        offsets = {
            nt: torch.tensor(
                [0] + torch.cumsum(torch.tensor(counts_per_patient[nt][:-1]), dim=0).tolist(),
                dtype=torch.long,
                device=hg_batch.device if hasattr(hg_batch, "device") else None,
            )
            for nt in node_types
        }
        offset_p = {nt: offsets[nt][patient_idx].item() for nt in node_types}
        size_p   = {nt: counts_per_patient[nt][patient_idx] for nt in node_types}

        # ---------- 3) 校验 occ_labels ----------
        if occ_labels is not None and "occ" in size_p:
            if len(occ_labels) != size_p["occ"]:
                raise ValueError(
                    f"occ_labels 长度 {len(occ_labels)} 与第 {patient_idx} 位病人的 occ 节点数 {size_p['occ']} 不匹配"
                )

        # ---------- 4) 从 batch 级 attn_dict 中切出当前病人的边与注意力 ----------
        attn_p: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}
        for rel, pair in attn_dict.items():
            if isinstance(pair, dict):
                edge_index = pair["edge_index"]
                alpha      = pair["alpha"]
            else:
                edge_index, alpha = pair

            src_t, _, dst_t = rel
            if src_t not in offset_p or dst_t not in offset_p:
                continue
            src_off, dst_off = offset_p[src_t], offset_p[dst_t]
            src_num, dst_num = size_p[src_t], size_p[dst_t]
            if src_num == 0 or dst_num == 0:
                continue

            src = edge_index[0]
            dst = edge_index[1]
            mask = (src >= src_off) & (src < src_off + src_num) & (dst >= dst_off) & (dst < dst_off + dst_num)
            if not mask.any():
                continue

            ei_local    = torch.stack([src[mask] - src_off, dst[mask] - dst_off], dim=0)  # [2, E_p]
            alpha_local = alpha[mask]  # [E_p, H] 或 [E_p]

            if head == "mean" and alpha_local.dim() == 2:
                alpha_plot = alpha_local.mean(dim=-1)
            elif head == "first" and alpha_local.dim() == 2:
                alpha_plot = alpha_local[:, 0]
            else:
                alpha_plot = alpha_local.mean(dim=-1) if alpha_local.dim() == 2 else alpha_local

            attn_p[rel] = (ei_local, alpha_plot)

        if rels_to_draw is not None:
            allow = set(rels_to_draw)
            attn_p = {rel: val for rel, val in attn_p.items() if rel in allow}

        # ---------- 5) 构图并为 visit 连接的 occ 生成克隆 ----------
        G = nx.MultiDiGraph()
        np.random.seed(layout_seed)

        # 5.1 Visit 节点
        visit_nodes: List[str] = []
        if "visit" in size_p:
            for i in range(size_p["visit"]):
                vname = f"V{i+1}"
                G.add_node(vname, type="visit")
                visit_nodes.append(vname)

        # 5.2 工具：occ 基名
        def _occ_base_name(idx: int) -> str:
            if occ_labels is not None:
                return occ_labels[idx]
            return f"O{idx+1}"

        # 5.3 记录每个 visit 的克隆 occ（用于布局）
        visit_to_occs: Dict[str, List[str]] = {v: [] for v in visit_nodes}

        # 5.4 添加边，生成克隆
        def _add_clone_for_visit(visit_name: str, base_idx: int) -> str:
            base_name = _occ_base_name(base_idx)
            clone = f"{base_name}_for_{visit_name}"
            if not G.has_node(clone):
                G.add_node(clone, type="occ", base=base_name, base_idx=base_idx, visit=visit_name)
            if clone not in visit_to_occs[visit_name]:
                visit_to_occs[visit_name].append(clone)
            return clone

        # 收集所有权重用于归一化
        all_weights = []
        edge_data = []  # 存储边的详细信息

        for rel, (ei_local, alpha_plot) in attn_p.items():
            s_t, r, d_t = rel
            uv_pairs = ei_local.t().tolist()
            ws       = alpha_plot.tolist()

            for (u, v), w in zip(uv_pairs, ws):
                all_weights.append(w)
                edge_data.append((s_t, d_t, r, u, v, w))

        # 计算权重的归一化范围
        if all_weights:
            min_weight = min(all_weights)
            max_weight = max(all_weights)
            weight_range = max_weight - min_weight
            if weight_range == 0:
                weight_range = 1.0  # 避免除零

        # 添加边和节点
        for s_t, d_t, r, u, v, w in edge_data:
            if s_t == "visit" and d_t == "occ":
                # 忽略 visit -> occ 的边（通常都是1.0，没有信息价值）
                vname = f"V{u+1}"
                occ_clone = _add_clone_for_visit(vname, v)
                # 不添加这条边到图中
                continue

            elif s_t == "occ" and d_t == "visit":
                vname = f"V{v+1}"
                occ_clone = _add_clone_for_visit(vname, u)
                # 反转边的方向：从 occ_clone -> vname 改为 vname -> occ_clone
                # 但保持原始的注意力权重
                G.add_edge(vname, occ_clone, rel=r, weight=float(w), normalized_weight=float(w))

            else:
                # 其他关系（若存在）：不克隆，简单命名（首字母 + 编号）
                def _name_for(t: str, idx: int) -> str:
                    if t == "visit":
                        return f"V{idx+1}"
                    elif t == "occ":
                        name = _occ_base_name(idx)
                        if not G.has_node(name):
                            G.add_node(name, type="occ_base", base=name, base_idx=idx)
                        return name
                    else:
                        name = f"{t[:1].upper()}{idx+1}"
                        if not G.has_node(name):
                            G.add_node(name, type=t)
                        return name

                src_name = _name_for(s_t, u)
                dst_name = _name_for(d_t, v)
                G.add_edge(src_name, dst_name, rel=r, weight=float(w), normalized_weight=float(w))

        # ---------- 6) 优化布局：固定半径 + 可调间距 ----------
        pos: Dict[str, Tuple[float, float]] = {}

        # 6.1 多行放置 visit，使用更大间距
        num_visits = len(visit_nodes)
        if num_visits > 0:
            rows = (num_visits + visits_per_row - 1) // visits_per_row
            for idx, v in enumerate(visit_nodes):
                r = idx // visits_per_row
                c = idx % visits_per_row
                cols_in_row = visits_per_row if (r < rows - 1) else (num_visits - r * visits_per_row)
                start_x = -(cols_in_row - 1) * visit_spacing_x / 2.0
                x = start_x + c * visit_spacing_x
                y = -r * visit_spacing_y
                pos[v] = (x, y)

        # 6.2 每个 visit 周围放置其克隆 occ（固定半径）
        for v in visit_nodes:
            cx, cy = pos[v]
            clones = visit_to_occs.get(v, [])
            if not clones:
                continue
            n = len(clones)
            # 使用固定半径
            radius = occ_ring_radius
            for k, occ_clone in enumerate(clones):
                angle = 2 * np.pi * k / n
                pos[occ_clone] = (cx + radius * np.cos(angle), cy + radius * np.sin(angle))

        # 6.3 可选：对 y 方向整体拉伸
        if y_stretch is not None and y_stretch != 1.0:
            for n, (xv, yv) in list(pos.items()):
                pos[n] = (xv, yv * y_stretch)

        # ---------- 7) 优化绘图 ----------
        plt.figure(figsize=figure_size)  # 增大画布以适应更大间距
        
        # 节点颜色和大小
        node_colors, node_sizes = [], []
        for n in G.nodes():
            ntype = G.nodes[n].get("type")
            if ntype == "visit":
                node_colors.append(node_color["visit"])
                node_sizes.append(visit_node_size)
            else:
                node_colors.append(node_color["diag"])
                node_sizes.append(occ_node_size)

        # 根据注意力分数计算边宽
        edge_weights_visual = []
        edge_labels = {}
        
        for u, v, d in G.edges(data=True):
            weight = d.get("weight", 1.0)
            if all_weights and len(all_weights) > 1 and weight_range > 0:
                # 将权重归一化到 [0, 1] 范围
                normalized = (weight - min_weight) / weight_range
                # 使用更温和的缩放，避免所有边都很粗
                visual_width = min_edge_width + normalized * (max_edge_width - min_edge_width)
            else:
                # 如果所有权重相同或只有一条边，使用中等宽度
                visual_width = min_edge_width + 0.5 * (max_edge_width - min_edge_width)
            
            edge_weights_visual.append(visual_width)
            edge_labels[(u, v)] = f"{weight:.3f}"

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=1.0)

        # 7.1 优化标签显示
        # Visit 节点标签：居中显示
        visit_labels = {v: v for v in visit_nodes}
        nx.draw_networkx_labels(G, {v: pos[v] for v in visit_labels}, labels=visit_labels,
                                font_size=12, font_weight="bold", font_color="white")

        # Occurrence 节点标签：显示在节点外面
        occ_label_pos = {}
        occ_display_labels = {}

        for n in G.nodes():
            node_data = G.nodes[n]
            if node_data.get("type") == "visit":
                continue
                
            x, y = pos[n]
            # 计算标签位置（向外偏移）
            if n in visit_nodes:
                continue
            
            # 找到对应的visit节点中心
            visit_name = node_data.get("visit")
            if visit_name and visit_name in pos:
                vx, vy = pos[visit_name]
                # 计算从visit中心到occ节点的方向
                dx, dy = x - vx, y - vy
                length = max(np.hypot(dx, dy), 1e-6)
                # 标签向外偏移
                offset_x = x + occ_label_offset * dx / length
                offset_y = y + occ_label_offset * dy / length
            else:
                # 如果没有关联visit，就向外偏移
                length = max(np.hypot(x, y), 1e-6)
                offset_x = x + occ_label_offset * x / length if length > 0 else x + occ_label_offset
                offset_y = y + occ_label_offset * y / length if length > 0 else y
                
            occ_label_pos[n] = (offset_x, offset_y)
            
            # 显示基础名称（去掉_for_后缀）
            base_name = node_data.get("base")
            if base_name is None:
                base_name = n.split("_for_")[0] if "_for_" in n else n
            occ_display_labels[n] = base_name

        # 绘制occurrence节点标签
        if occ_display_labels:
            nx.draw_networkx_labels(G, occ_label_pos, labels=occ_display_labels,
                                    font_size=occ_label_fontsize, font_weight="bold", font_color="black")

        # 绘制边（使用基于注意力分数的宽度）
        nx.draw_networkx_edges(G, pos, arrows=True, width=edge_weights_visual, alpha=0.9,
                            edge_color="black", arrowsize=15, arrowstyle='->')

        # 绘制边标签（注意力分数）
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=attn_score_fontsize)

        plt.axis("off")
        ax = plt.gca()
        ax.margins(0.1)  # 减少边距以更好利用空间
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout(pad=1.5)
        plt.show()

    @staticmethod
    def adjust_agg_attn_prob(attn_probs: torch.Tensor):
        """
        attn_probs: shape (B, 1, N)
        - 第 0 个位置 (CLS) 强制为 0
        - 原本为 0 的位置保持为 0
        - 其余位置 softmax 归一化
        """
        attn = attn_probs.clone()

        # 构造 mask：非零的位置才参与 softmax
        mask = (attn > 0).float()
        mask[:, 0, 0] = 0.0   # 强制 CLS 为 0

        # 对非零部分取 logit，再 softmax
        # step1: 将不参与的位置置为 -inf
        masked_logits = attn.masked_fill(mask == 0, float('-inf'))

        # step2: softmax 归一化
        new_attn = F.softmax(masked_logits, dim=-1)

        # step3: 确保原本 0 的位置仍为 0
        new_attn = new_attn * mask

        return new_attn