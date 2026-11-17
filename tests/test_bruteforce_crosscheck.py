"""独立暴力枚举交叉验证（与 chart 实现不共享任何算法代码）。

思路：递归枚举覆盖每个 span 的全部接地推导——
- 长度 1 的 span：每个同类且满足特征子集的词形候选各产生一棵叶树；
- 更长 span：枚举一元/二元规则的全部变量赋值与切分点，递归构树。
由此直接累加分区、最大树与“使用某叶候选的总质量”，与 analyze_sentence 对照。
"""

import itertools
import math
import random
import unittest

from syntax import analyze_sentence

ATOMS = ["a", "b", "c"]
CATS = ["N", "V", "S", "T"]
ATTRS = ["f", "g"]


def _ground(sym, asg):
    return sym[0], tuple(sorted(
        (k, asg[v] if v.startswith("?") else v) for k, v in sym[1]
    ))


def _match(sym, label):
    if sym[0] != label[0]:
        return False
    feats = dict(label[1])
    return all(feats.get(k) == v for k, v in sym[1])


def brute_force(grammar, candidates):
    """返回 (Z, best_w, 每位置 {analysis_id: 含该叶的推导质量和})，无解析返回 None。"""
    rules = grammar["rules"]
    expanded = []
    for r in rules:
        lhs = (r["lhs"]["cat"], tuple(sorted(r["lhs"]["features"].items())))
        rhs = [
            (s["cat"], tuple(sorted(s["features"].items())))
            for s in r["rhs"]
        ]
        variables = sorted({
            v for sym in (lhs, *rhs) for _, v in sym[1] if v.startswith("?")
        })
        for combo in itertools.product(grammar["atoms"], repeat=len(variables)):
            asg = dict(zip(variables, combo))
            expanded.append((r["id"], _ground(lhs, asg),
                             [_ground(s, asg) for s in rhs], r["weight"], dict(asg)))

    # 同一 label 允许有多个不同 identity 的叶候选（同形同特征也分别保留），
    # 因此按 (位置, label) 存列表，不能用字典互相覆盖。
    leaves = {}
    for i, pos in enumerate(candidates):
        for c in pos:
            label = (c["cat"], tuple(sorted(c["features"].items())))
            leaves.setdefault((i, label), []).append((c["analysis_id"], c["weight"]))

    from functools import lru_cache

    # 全部可能出现的 label：叶 label + 接地 lhs，再对一元规则做闭包
    # （父 lhs 可由任意满足 rhs 子集要求的 label 推导而来，链式延伸）。
    universe = {lab for (_, lab) in leaves}
    universe |= {lhs for (_rid, lhs, _rhs, _rw, _asg) in expanded}
    changed = True
    while changed:
        changed = False
        for _rid, lhs, rhs, _rw, _asg in expanded:
            if len(rhs) == 1 and any(_match(rhs[0], lab) for lab in universe):
                if lhs not in universe:
                    universe.add(lhs)
                    changed = True

    @lru_cache(maxsize=None)
    def derivs(label, start, end):
        """根恰为 label 的全部 (weight, tree)；label 必须是 universe 中的接地 label。"""
        out = []
        if end - start == 1:
            for (i, lab), leaf_list in leaves.items():
                if i == start and lab == label:
                    for aid, w in leaf_list:
                        out.append((w, ("leaf", aid)))
        for rid, lhs, rhs, rw, asg in expanded:
            if lhs != label:
                continue
            if len(rhs) == 1:
                # 子树根 label 只需满足 rhs[0] 的特征子集要求。
                for child_label in universe:
                    if not _match(rhs[0], child_label):
                        continue
                    for cw, ct in derivs(child_label, start, end):
                        out.append((
                            rw * cw,
                            ("rule", rid, tuple(sorted(asg.items())), (ct,)),
                        ))
            else:
                for split in range(start + 1, end):
                    for l_label in universe:
                        if not _match(rhs[0], l_label):
                            continue
                        for r_label in universe:
                            if not _match(rhs[1], r_label):
                                continue
                            for lw, lt in derivs(l_label, start, split):
                                for rw2, rt in derivs(r_label, split, end):
                                    out.append((
                                        rw * lw * rw2,
                                        ("rule", rid, tuple(sorted(asg.items())), (lt, rt)),
                                    ))
        return tuple(out)

    start_sym = (grammar["start"]["cat"],
                 tuple(sorted(grammar["start"]["features"].items())))
    n = len(candidates)
    # 根须“满足” start 要求：类别相同且包含 start 全部特征（可多带特征）。
    trees = []
    for lab in universe:
        if _match(start_sym, lab):
            trees.extend(derivs(lab, 0, n))
    if not trees:
        return None

    z = sum(w for w, _ in trees)
    best_w = max(w for w, _ in trees)

    used = [{} for _ in range(n)]

    def walk(tree, mass):
        kind = tree[0]
        if kind == "leaf":
            aid = tree[1]
            for i, pos in enumerate(candidates):
                if any(c["analysis_id"] == aid for c in pos):
                    used[i][aid] = used[i].get(aid, 0.0) + mass
                    return
        for child in tree[3]:
            walk(child, mass)

    for w, t in trees:
        walk(t, w)
    return z, best_w, used


def random_grammar(rng):
    rule_count = rng.randint(1, 7)
    rules = []
    used_unary = set()
    for ridx in range(rule_count):
        rid = f"r{ridx}"
        arity = rng.choice([1, 1, 2])
        lhs_cat = rng.choice(CATS)
        rhs_cats = [rng.choice(CATS) for _ in range(arity)]
        if arity == 1 and (lhs_cat, rhs_cats[0]) in used_unary:
            arity = 2
            rhs_cats = [rng.choice(CATS) for _ in range(2)]
        if arity == 1:
            used_unary.add((lhs_cat, rhs_cats[0]))

        def rand_sym(cat):
            feats = {}
            for attr in rng.sample(ATTRS, rng.randint(0, 2)):
                feats[attr] = rng.choice(["?x", "?y", *ATOMS])
            return {"cat": cat, "features": feats}

        rule = {
            "id": rid, "weight": round(rng.uniform(0.1, 3.0), 3),
            "lhs": rand_sym(lhs_cat),
            "rhs": [rand_sym(c) for c in rhs_cats],
        }
        rules.append(rule)

    grammar = {
        "start": {"cat": rng.choice(CATS), "features": {}},
        "atoms": ATOMS,
        "rules": rules,
    }
    # 一元环在本工具中是非法语法：抽到就跳过该随机场
    try:
        analyze_sentence(grammar, [[
            {"analysis_id": "x", "cat": CATS[0], "features": {}, "weight": 1.0}]])
    except ValueError:
        return None
    return grammar


def random_candidates(rng, n):
    cands = []
    for _ in range(n):
        pos = []
        for k in range(rng.randint(0, 3)):
            feats = {}
            for attr in rng.sample(ATTRS, rng.randint(0, 2)):
                feats[attr] = rng.choice(ATOMS)
            pos.append({
                "analysis_id": f"c{len(cands)}-{k}",
                "cat": rng.choice(CATS),
                "features": feats,
                "weight": round(rng.uniform(1e-3, 5.0), 4),
            })
        # 去重 identity
        seen, clean = set(), []
        for c in pos:
            if c["analysis_id"] not in seen:
                seen.add(c["analysis_id"])
                clean.append(c)
        cands.append(clean)
    return cands


class TestBruteForceCrossCheck(unittest.TestCase):
    def test_random_cross_check(self):
        rng = random.Random(20261117)
        trials, parses = 0, 0
        while parses < 40 and trials < 3000:
            trials += 1
            grammar = random_grammar(rng)
            if grammar is None:
                continue
            n = rng.randint(1, 5)
            cands = random_candidates(rng, n)
            expected = brute_force(grammar, cands)

            got = analyze_sentence(grammar, cands)
            if expected is None:
                self.assertEqual(got["status"], "no_parse")
                continue
            parses += 1
            z, best_w, used = expected
            self.assertEqual(got["status"], "ok")
            self.assertAlmostEqual(got["partition_weight"], z, delta=z * 1e-9 + 1e-12)
            self.assertAlmostEqual(math.exp(got["log_partition"]), z,
                                   delta=z * 1e-9 + 1e-12)
            self.assertAlmostEqual(math.exp(got["best_log_weight"]), best_w,
                                   delta=best_w * 1e-9 + 1e-12)
            self.assertAlmostEqual(got["best"]["weight"], best_w,
                                   delta=best_w * 1e-9 + 1e-12)
            for i, pos in enumerate(cands):
                row_sum = 0.0
                for c in pos:
                    aid = c["analysis_id"]
                    want = used[i].get(aid, 0.0) / z
                    got_p = got["marginals"][i].get(aid, 0.0)
                    self.assertAlmostEqual(got_p, want, delta=1e-9)
                    row_sum += got_p
                if pos:
                    self.assertAlmostEqual(row_sum, 1.0, delta=1e-9)
        self.assertGreaterEqual(parses, 40)


if __name__ == "__main__":
    unittest.main(verbosity=2)
