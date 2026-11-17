"""特征合一句法：有限原子域上的接地展开 + 加权 CFG chart。

接口
----
analyze_sentence(grammar, candidates)
    grammar    : start（无变量）、带 id/weight 的规则（rhs 长度 1 或 2，无 epsilon）、
                 atoms（有限非空原子值域）
    candidates : 每个 token 一个有限词形候选列表；候选含位置内唯一 analysis_id、
                 cat、完整原子 features、正 weight；允许空列表（合法的无解析输入）

表示与代价
----------
每条规则至多 3 个变量、atoms 至多 4 个，因此一条规则的变量赋值至多 4^3 = 64 种。
构造语法时把每条规则的全部合法赋值**预接地**为普通加权 CFG 规则：合一在接地阶段
完成，chart 阶段只做类别相同 + 特征子集匹配（子节点多出的特征不冲突、不传播，
父节点特征恰为本次 lhs 的实例化映射）。同一规则的不同变量赋值是不同规则使用，
权重分别计入；不同规则的变量天然隔离（各自接地，不共享名字）。

chart 用标准 inside（分区、真正最大树）与 outside（逐叶后验）：
    I(label, span) = 以该 label 覆盖该 span 的全部完整子推导质量和；
    O(label, span) = span 外的上下文质量和；
    叶候选 c 的后验 = O(label, span) * w_c / Z(root, 全句)。
这与“每位置局部 softmax”不同：后验由整句分区决定。
一元规则的类别依赖图在构造时做环检测，存在类别环直接拒绝（不依赖搜索碰巧终止）。
"""

from __future__ import annotations

import itertools
import math

__all__ = ["analyze_sentence"]

MAX_CATS = 8
MAX_ATTRS = 3
MAX_ATOMS = 4
MAX_RULES = 16
MAX_VARS = 3
MAX_TOKENS = 8
MAX_CANDIDATES = 8
RULE_W_LO, RULE_W_HI = 0.1, 3.0
CAND_W_LO, CAND_W_HI = 1e-5, 243.0
NEG_INF = float("-inf")


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _is_bool(x):
    return isinstance(x, bool)


def _name(value, name):
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{name} 必须是非空字符串，得到 {value!r}")
    if value.startswith("?"):
        raise ValueError(f"{name} 不能以 '?' 开头：{value!r}")
    return value


def _weight(value, name, lo, hi):
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限数值，得到 {value!r}")
    value = float(value)
    if not math.isfinite(value) or not (lo <= value <= hi):
        raise ValueError(f"{name}={value!r} 超出有限区间 [{lo}, {hi}]")
    return value


def _symbol(sym, name, atoms):
    """校验符号，返回 (cat, ((attr, value 或变量), ...))，特征按属性名排序。"""
    if not isinstance(sym, dict):
        raise ValueError(f"{name} 必须是对象/字典")
    cat = _name(sym.get("cat"), f"{name}.cat")
    feats = sym.get("features", {})
    if not isinstance(feats, dict):
        raise ValueError(f"{name}.features 必须是对象/字典")
    items = []
    for key, val in feats.items():
        _name(key, f"{name}.{key} 的属性名")
        if not isinstance(val, str) or val == "":
            raise ValueError(f"{name}.{key} 的值必须是非空字符串")
        if val.startswith("?"):
            if len(val) == 1:
                raise ValueError(f"{name}.{key} 的变量 {val!r} 非法")
        elif val not in atoms:
            raise ValueError(f"{name}.{key} 的原子 {val!r} 不在声明值域 {atoms}")
        items.append((key, val))
    return cat, tuple(sorted(items))


def _ground_symbol(sym, assignment):
    cat, items = sym
    grounded = []
    for key, val in items:
        grounded.append((key, assignment[val] if val.startswith("?") else val))
    return cat, tuple(grounded)


def _label(cat, feats):
    return cat, tuple(sorted(feats.items()))


def _label_matches(symbol, label):
    """子树根/词形候选类别相同，并包含且满足符号的全部特征键值。"""
    if symbol[0] != label[0]:
        return False
    feats = dict(label[1])
    return all(feats.get(k) == v for k, v in symbol[1])


def _tree_key(node):
    """最优并列时的确定性平局键（只看结构身份，不看权重）。"""
    if node["node"] == "leaf":
        return ("leaf", node["analysis_id"])
    return ("rule", node["rule_id"],
            tuple(sorted(node["assignment"].items())),
            tuple(_tree_key(c) for c in node["children"]))


# --------------------------------------------------------------------------- #
# 语法校验与接地
# --------------------------------------------------------------------------- #
class _Grammar:
    def __init__(self, grammar):
        if not isinstance(grammar, dict):
            raise ValueError("grammar 必须是对象/字典")

        atoms = grammar.get("atoms")
        if not isinstance(atoms, (list, tuple)) or not (1 <= len(atoms) <= MAX_ATOMS):
            raise ValueError(f"grammar.atoms 必须含 1..{MAX_ATOMS} 个非空原子")
        self.atoms = []
        seen = set()
        for a in atoms:
            _name(a, "atom")
            if a in seen:
                raise ValueError(f"原子重复：{a!r}")
            seen.add(a)
            self.atoms.append(a)

        self.start = _symbol(grammar.get("start", {}), "start", self.atoms)
        if any(v.startswith("?") for _, v in self.start[1]):
            raise ValueError("start 符号不能含变量")

        rules = grammar.get("rules", [])
        if not isinstance(rules, (list, tuple)) or not (1 <= len(rules) <= MAX_RULES):
            raise ValueError(f"rules 必须含 1..{MAX_RULES} 条规则")
        raw_rules, ids = [], set()
        cats, attrs = {self.start[0]}, set(k for k, _ in self.start[1])
        for raw in rules:
            if not isinstance(raw, dict):
                raise ValueError("规则必须是对象/字典")
            rid = _name(raw.get("id"), "rule.id")
            if rid in ids:
                raise ValueError(f"rule id 重复：{rid!r}")
            ids.add(rid)
            lhs = _symbol(raw.get("lhs", {}), f"rule {rid}.lhs", self.atoms)
            rhs_raw = raw.get("rhs")
            if not isinstance(rhs_raw, (list, tuple)) or not (1 <= len(rhs_raw) <= 2):
                raise ValueError(f"rule {rid}.rhs 长度必须是 1 或 2，无 epsilon")
            rhs = [_symbol(s, f"rule {rid}.rhs[{i}]", self.atoms) for i, s in enumerate(rhs_raw)]
            weight = _weight(raw.get("weight"), f"rule {rid}.weight", RULE_W_LO, RULE_W_HI)

            variables = sorted({
                v for sym in (lhs, *rhs) for _, v in sym[1] if v.startswith("?")
            })
            if len(variables) > MAX_VARS:
                raise ValueError(f"rule {rid} 不同变量至多 {MAX_VARS} 个，得到 {len(variables)}")
            # 每个变量至少在本规则某个符号特征中出现（variables 本身即从符号收集）。
            raw_rules.append((rid, lhs, rhs, weight, variables))
            for sym in (lhs, *rhs):
                cats.add(sym[0])
                attrs.update(k for k, _ in sym[1])

        if len(cats) > MAX_CATS:
            raise ValueError(f"不同 cat 至多 {MAX_CATS} 个，得到 {len(cats)}")
        if len(attrs) > MAX_ATTRS:
            raise ValueError(f"不同属性至多 {MAX_ATTRS} 个，得到 {len(attrs)}")

        # 一元类别依赖图必须无环（声明的环即非法，不依赖数据可达性）。
        unary_edges = {}
        for _, lhs, rhs, _, _ in raw_rules:
            if len(rhs) == 1:
                unary_edges.setdefault(lhs[0], set()).add(rhs[0][0])
        self.unary_topo = self._topo_sort(unary_edges)

        # 预接地：每条规则的每个变量赋值生成一条普通加权 CFG 规则，
        # 并保留 assignment，使树节点可独立重算规则使用身份。
        self.unary = []          # (parent, child, rid, weight, assignment)
        self.binary = []         # (parent, left, right, rid, weight, assignment)
        for rid, lhs, rhs, weight, variables in raw_rules:
            for combo in itertools.product(self.atoms, repeat=len(variables)):
                assignment = dict(zip(variables, combo))
                g_lhs = _ground_symbol(lhs, assignment)
                g_rhs = tuple(_ground_symbol(s, assignment) for s in rhs)
                if len(g_rhs) == 1:
                    self.unary.append((g_lhs, g_rhs[0], rid, weight, dict(assignment)))
                else:
                    self.binary.append(
                        (g_lhs, g_rhs[0], g_rhs[1], rid, weight, dict(assignment))
                    )

        # 索引：按子类别，chart 中只尝试同类接地规则。
        self.unary_by_child = {}
        for parent, child, rid, w, asg in self.unary:
            self.unary_by_child.setdefault(child[0], []).append((parent, child, rid, w, asg))
        self.binary_by_pair = {}
        for parent, left, right, rid, w, asg in self.binary:
            self.binary_by_pair.setdefault((left[0], right[0]), []).append(
                (parent, left, right, rid, w, asg)
            )

    @staticmethod
    def _topo_sort(edges):
        """Kahn 拓扑排序；edges: parent_cat -> child_cat 集合。返回 child 先于 parent 的顺序。"""
        cats = set(edges)
        for kids in edges.values():
            cats.update(kids)
        indeg = {c: 0 for c in cats}
        for parent, kids in edges.items():
            indeg[parent] = indeg.get(parent, 0) + len(kids)
        queue = sorted(c for c in cats if indeg[c] == 0)
        order = []
        while queue:
            cat = queue.pop(0)
            order.append(cat)
            for parent, kids in edges.items():
                if cat in kids:
                    indeg[parent] -= 1
                    if indeg[parent] == 0:
                        queue.append(parent)
        if len(order) != len(cats):
            raise ValueError("一元规则的类别依赖图存在环，非法语法")
        return order


# --------------------------------------------------------------------------- #
# 候选校验
# --------------------------------------------------------------------------- #
def _validate_candidates(candidates, grammar):
    if not isinstance(candidates, (list, tuple)) or not (1 <= len(candidates) <= MAX_TOKENS):
        raise ValueError(f"句子必须含 1..{MAX_TOKENS} 个 token 位置")
    positions, cats, attrs = [], set(), set()
    for i, pos in enumerate(candidates):
        if not isinstance(pos, (list, tuple)) or len(pos) > MAX_CANDIDATES:
            raise ValueError(f"位置 {i} 候选至多 {MAX_CANDIDATES} 个且必须是列表")
        ids = set()
        leaves = []
        for cand in pos:
            if not isinstance(cand, dict):
                raise ValueError(f"位置 {i} 的候选必须是对象/字典")
            aid = _name(cand.get("analysis_id"), f"位置 {i} 候选 analysis_id")
            if aid in ids:
                raise ValueError(f"位置 {i} 的 analysis_id 重复：{aid!r}")
            ids.add(aid)
            cat = _name(cand.get("cat"), f"候选 {aid}.cat")
            feats = cand.get("features", {})
            if not isinstance(feats, dict):
                raise ValueError(f"候选 {aid}.features 必须是对象/字典")
            clean = {}
            for key, val in feats.items():
                _name(key, f"候选 {aid} 属性名")
                if not isinstance(val, str) or val == "" or val.startswith("?"):
                    raise ValueError(f"候选 {aid}.{key} 必须是值域内原子字符串")
                if val not in grammar.atoms:
                    raise ValueError(f"候选 {aid}.{key} 的原子 {val!r} 不在声明值域")
                clean[key] = val
            w = _weight(cand.get("weight"), f"候选 {aid}.weight", CAND_W_LO, CAND_W_HI)
            cats.add(cat)
            attrs.update(clean)
            leaves.append({"analysis_id": aid, "label": _label(cat, clean), "weight": w})
        positions.append(leaves)
    if len(cats) > MAX_CATS or len(attrs) > MAX_ATTRS:
        raise ValueError(f"候选导致 cat 至多 {MAX_CATS}、属性至多 {MAX_ATTRS} 的约束被破坏")
    return positions


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def analyze_sentence(grammar, candidates):
    g = _Grammar(grammar)
    positions = _validate_candidates(candidates, g)
    n = len(positions)

    # 任一位置无候选 => 合法的无解析输入，直接短路。
    if any(len(pos) == 0 for pos in positions):
        return _no_parse()

    # chart[start][end] : label -> {"inside", "best", "best_w"}
    chart = [[{} for _ in range(n + 1)] for _ in range(n + 1)]
    outside = [[{} for _ in range(n + 1)] for _ in range(n + 1)]

    def add_inside(cell, label, mass, node):
        """同一 label 的多个推导：inside 累加全部分区质量；best 只保留最大子树。

        关键：node 的 weight 必须是该次规则使用下真正的子树质量（由子节点的
        最大树组合而来），而 inside 的 mass 是该规则使用的全部子推导质量和，
        两者分别维护，不能用 inside 之和去拼最优树。
        """
        entry = cell.get(label)
        if entry is None:
            cell[label] = {"inside": mass, "best": node, "best_w": node["weight"]}
            return
        entry["inside"] += mass
        if node["weight"] > entry["best_w"] or (
            node["weight"] == entry["best_w"] and _tree_key(node) < _tree_key(entry["best"])
        ):
            entry["best"] = node
            entry["best_w"] = node["weight"]

    # ---------------- inside：叶 ---------------- #
    for i, leaves in enumerate(positions):
        cell = chart[i][i + 1]
        for leaf in leaves:
            add_inside(
                cell, leaf["label"], leaf["weight"],
                {
                    "node": "leaf",
                    "analysis_id": leaf["analysis_id"],
                    "cat": leaf["label"][0],
                    "features": dict(leaf["label"][1]),
                    "weight": leaf["weight"],
                },
            )

    # ---------------- inside：按 span 长度自底向上 ---------------- #
    for span in range(1, n + 1):
        for start in range(0, n - span + 1):
            end = start + span
            cell = chart[start][end]

            # 二元：枚举切分点与接地规则。
            for split in range(start + 1, end):
                left_cell, right_cell = chart[start][split], chart[split][end]
                for (lcat, rcat), rules in g.binary_by_pair.items():
                    left_labels = [lab for lab in left_cell if lab[0] == lcat]
                    right_labels = [lab for lab in right_cell if lab[0] == rcat]
                    if not left_labels or not right_labels:
                        continue
                    for parent, lsym, rsym, rid, rw, asg in rules:
                        for llab in left_labels:
                            if not _label_matches(lsym, llab):
                                continue
                            lnode = left_cell[llab]
                            for rlab in right_labels:
                                if not _label_matches(rsym, rlab):
                                    continue
                                rnode = right_cell[rlab]
                                mass = rw * lnode["inside"] * rnode["inside"]
                                node = {
                                    "node": "rule",
                                    "rule_id": rid,
                                    "assignment": dict(asg),
                                    "cat": parent[0],
                                    "features": dict(parent[1]),
                                    "children": [lnode["best"], rnode["best"]],
                                    "weight": rw * lnode["best_w"] * rnode["best_w"],
                                }
                                add_inside(cell, parent, mass, node)

            # 一元闭合：类别 DAG 拓扑序（child 先于 parent），单趟即可传递。
            for cat in g.unary_topo:
                rules = g.unary_by_child.get(cat)
                if not rules:
                    continue
                for parent, child_sym, rid, rw, asg in rules:
                    for lab in list(cell):
                        if lab[0] != cat or not _label_matches(child_sym, lab):
                            continue
                        child = cell[lab]
                        mass = rw * child["inside"]
                        node = {
                            "node": "rule",
                            "rule_id": rid,
                            "assignment": dict(asg),
                            "cat": parent[0],
                            "features": dict(parent[1]),
                            "children": [child["best"]],
                            "weight": rw * child["best_w"],
                        }
                        add_inside(cell, parent, mass, node)

    root_cell = chart[0][n]
    root_entries = [
        (lab, ent) for lab, ent in root_cell.items() if _label_matches(g.start, lab)
    ]
    if not root_entries:
        return _no_parse()

    z = sum(ent["inside"] for _, ent in root_entries)
    max_best = max(ent["best_w"] for _, ent in root_entries)
    # 真正最大质量；并列时按结构键确定性任选其一。
    best_label, best_ent = min(
        (item for item in root_entries if item[1]["best_w"] == max_best),
        key=lambda item: _tree_key(item[1]["best"]),
    )

    # ---------------- outside ---------------- #
    # 根满足 start 的 label 拥有空上下文（outside=1）；其余根 label 不参与整句。
    out_root = outside[0][n]
    for lab, _ in root_entries:
        out_root[lab] = 1.0

    for span in range(n, 0, -1):
        for start in range(0, n - span + 1):
            end = start + span
            cell, out_cell = chart[start][end], outside[start][end]

            # 一元：parent -> child，反向拓扑序（parent 先于 child）单趟传递。
            for cat in reversed(g.unary_topo):
                rules = g.unary_by_child.get(cat)
                if not rules:
                    continue
                for parent, child_sym, rid, rw, _asg in rules:
                    if parent not in out_cell:
                        continue
                    ctx = rw * out_cell[parent]
                    for lab in cell:
                        if lab[0] == cat and _label_matches(child_sym, lab):
                            out_cell[lab] = out_cell.get(lab, 0.0) + ctx

            # 二元：parent -> left / right 两种上下文。
            if span >= 2:
                for split in range(start + 1, end):
                    left_cell, right_cell = chart[start][split], chart[split][end]
                    out_left, out_right = outside[start][split], outside[split][end]
                    for (_lcat, _rcat), rules in g.binary_by_pair.items():
                        for parent, lsym, rsym, rid, rw, _asg in rules:
                            if parent not in out_cell:
                                continue
                            ctx = rw * out_cell[parent]
                            for llab, lent in left_cell.items():
                                if llab[0] != lsym[0] or not _label_matches(lsym, llab):
                                    continue
                                for rlab, rent in right_cell.items():
                                    if rlab[0] != rsym[0] or not _label_matches(rsym, rlab):
                                        continue
                                    out_left[llab] = out_left.get(llab, 0.0) + ctx * rent["inside"]
                                    out_right[rlab] = out_right.get(rlab, 0.0) + ctx * lent["inside"]

    # ---------------- 逐叶后验（整句质量 / 分区，不做局部 softmax） ---------------- #
    marginals = []
    for i, leaves in enumerate(positions):
        pos_out = {}
        for leaf in leaves:
            aid, label, w = leaf["analysis_id"], leaf["label"], leaf["weight"]
            ctx = outside[i][i + 1].get(label, 0.0)
            pos_out[aid] = ctx * w / z
        marginals.append(pos_out)

    return {
        "status": "ok",
        "log_partition": math.log(z),
        "best_log_weight": math.log(best_ent["best_w"]),
        "best": best_ent["best"],
        "marginals": marginals,
        "partition_weight": z,
    }


def _no_parse():
    return {
        "status": "no_parse",
        "log_partition": NEG_INF,
        "best_log_weight": NEG_INF,
        "best": None,
        "marginals": None,
        "partition_weight": 0.0,
    }
