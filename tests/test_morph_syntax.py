"""手算核对测试（标准库 unittest，运行：python -m unittest -v）。

核对方式刻意避免“analyze 与 generate 互相 round-trip 自证”：
- 词形：在独立小词典上逐条手列接受路径（串/特征/转移序列/权重）再断言；
- 句法：在极小语法上完整展开规则赋值与全部推导，手算分区、后验与最优树。
"""

import math
import unittest

from morph import Morphology
from syntax import analyze_sentence
from pipeline import analyze_tokens


# --------------------------------------------------------------------------- #
# 词形手算小词典
# --------------------------------------------------------------------------- #
ATOMS_M = ["x", "y", "p", "q"]

M_ENTRIES = [
    {"id": "e1", "stem": "ab", "cat": "C1", "features": {"f": "x"}, "weight": 2.0},
    {"id": "e2", "stem": "cd", "cat": "C2", "features": {"f": "y", "g": "p"}, "weight": 1.5},
]

# 状态：0(start, accept)、1(accept)、2(accept)。最长路径 2 条边，无环。
M_AUTOMATON = {"start": 0, "accept": [0, 1, 2], "transitions": [
    # 前缀边，无门控
    {"id": "t-prefix", "from": 0, "to": 1, "requires": {},
     "set_features": {}, "replace_tail": ["", ""], "prefix": "z", "suffix": "",
     "weight": 0.5},
    # 尾替换边：去 b 接 k
    {"id": "t-tail", "from": 0, "to": 1, "requires": {},
     "set_features": {"f": "y"}, "replace_tail": ["b", "k"],
     "prefix": "", "suffix": "", "weight": 2.0},
    # 门控边：仅 f=y 可用（e1 永远走不了，e2 可以）
    {"id": "t-gate", "from": 0, "to": 1, "requires": {"f": "y"},
     "set_features": {}, "replace_tail": ["", ""], "prefix": "g", "suffix": "",
     "weight": 3.0},
    # accept 状态 1 上继续扩展：后缀边
    {"id": "t-suffix", "from": 1, "to": 2, "requires": {},
     "set_features": {"g": "q"}, "replace_tail": ["", ""],
     "prefix": "", "suffix": "h", "weight": 0.25},
    # 不可用 requires：声明值域里没有任何 entry/边会产生 f=q
    {"id": "t-dead", "from": 0, "to": 2, "requires": {"f": "q"},
     "set_features": {}, "replace_tail": ["", ""], "prefix": "D", "suffix": "",
     "weight": 1.0},
]}


def _index(rows):
    return {(r["entry_id"], tuple(r["transitions"])): r for r in rows}


class TestMorphologyHandCalc(unittest.TestCase):
    def setUp(self):
        self.m = Morphology(M_ENTRIES, M_AUTOMATON, ATOMS_M)

    def test_e1_paths_hand_calc(self):
        # e1: stem=ab, f=x, w=2.0；t-gate 因 f!=y 不可走，t-dead 因 f!=q 不可走。
        rows = self.m.generate("e1", {})
        idx = _index(rows)
        # 零次变换
        self.assertEqual(idx[("e1", ())]["surface"], "ab")
        self.assertEqual(idx[("e1", ())]["features"], {"f": "x"})
        self.assertEqual(idx[("e1", ())]["weight"], 2.0)
        # 前缀：z + ab
        r = idx[("e1", ("t-prefix",))]
        self.assertEqual(r["surface"], "zab")
        self.assertEqual(r["features"], {"f": "x"})
        self.assertAlmostEqual(r["weight"], 2.0 * 0.5, places=12)
        # 尾替换：ab 去 b -> a，接 k
        r = idx[("e1", ("t-tail",))]
        self.assertEqual(r["surface"], "ak")
        self.assertEqual(r["features"], {"f": "y"})
        self.assertAlmostEqual(r["weight"], 2.0 * 2.0, places=12)
        # 前缀后再后缀（accept 继续走边）
        r = idx[("e1", ("t-prefix", "t-suffix"))]
        self.assertEqual(r["surface"], "zabh")
        self.assertEqual(r["features"], {"f": "x", "g": "q"})
        self.assertAlmostEqual(r["weight"], 2.0 * 0.5 * 0.25, places=12)
        # 尾替换后再后缀
        r = idx[("e1", ("t-tail", "t-suffix"))]
        self.assertEqual(r["surface"], "akh")
        self.assertEqual(r["features"], {"f": "y", "g": "q"})
        self.assertAlmostEqual(r["weight"], 2.0 * 2.0 * 0.25, places=12)
        self.assertEqual(len(rows), 5)

    def test_e2_gate_available_hand_calc(self):
        # e2: stem=cd, f=y g=p, w=1.5；t-gate 可用，t-dead 仍不可用，
        # t-tail 因 cd 不以 b 结尾同样不可走（尾串门控独立于特征门控）。
        rows = self.m.generate("e2", {})
        idx = _index(rows)
        self.assertEqual(idx[("e2", ())]["surface"], "cd")
        self.assertEqual(idx[("e2", ("t-prefix",))]["surface"], "zcd")
        self.assertNotIn(("e2", ("t-tail",)), idx)
        gated = idx[("e2", ("t-gate",))]
        self.assertEqual(gated["surface"], "gcd")
        self.assertEqual(gated["features"], {"f": "y", "g": "p"})
        self.assertAlmostEqual(gated["weight"], 1.5 * 3.0, places=12)
        self.assertEqual(len(rows), 5)        # 零/prefix/gate 及各自接 t-suffix

    def test_analyze_keeps_homograph_paths_distinct(self):
        # 造一个同形：让 e2 的两条路径落到同一串。
        entries = [{"id": "h", "stem": "ab", "cat": "C1",
                    "features": {"f": "x"}, "weight": 1.0}]
        automaton = {"start": 0, "accept": [0, 1], "transitions": [
            {"id": "p1", "from": 0, "to": 1, "requires": {},
             "set_features": {"g": "p"}, "replace_tail": ["", ""],
             "prefix": "z", "suffix": "", "weight": 0.5},
            {"id": "p2", "from": 0, "to": 1, "requires": {},
             "set_features": {"g": "q"}, "replace_tail": ["", ""],
             "prefix": "z", "suffix": "", "weight": 0.7},
        ]}
        mm = Morphology(entries, automaton, ATOMS_M)
        rows = mm.analyze("zab")
        self.assertEqual(len(rows), 2)                      # 不去重、不丢质量
        self.assertEqual({tuple(r["transitions"]) for r in rows}, {("p1",), ("p2",)})
        self.assertAlmostEqual(sorted(r["weight"] for r in rows), [0.5, 0.7], places=12)
        self.assertNotEqual(rows[0]["analysis_id"], rows[1]["analysis_id"])
        # 两个不同 entry 同形也必须分别保留
        entries2 = entries + [
            {"id": "h2", "stem": "zab", "cat": "C1",
             "features": {"f": "x"}, "weight": 3.0},
        ]
        mm2 = Morphology(entries2, automaton, ATOMS_M)
        rows2 = mm2.analyze("zab")
        self.assertEqual({r["entry_id"] for r in rows2}, {"h", "h2"})
        self.assertEqual(len(rows2), 3)

    def test_analyze_filters_by_exact_surface(self):
        rows = self.m.analyze("ab")
        self.assertEqual([(r["entry_id"], r["transitions"]) for r in rows],
                         [("e1", [])])
        self.assertEqual(self.m.analyze("nope"), [])
        # t-dead 永远不可走：Dab/Dcd 都不该出现
        self.assertEqual(self.m.analyze("Dab"), [])
        self.assertEqual(self.m.analyze("Dcd"), [])

    def test_generate_constraints_are_subset(self):
        rows = self.m.generate("e1", {"f": "y"})
        self.assertEqual({tuple(r["transitions"]) for r in rows},
                         {("t-tail",), ("t-tail", "t-suffix")})
        rows = self.m.generate("e2", {"f": "y", "g": "q"})
        # 只有经过 t-suffix 的路径最终 g=q；e2 上 t-tail 不可走
        self.assertEqual({tuple(r["transitions"]) for r in rows},
                         {("t-prefix", "t-suffix"), ("t-gate", "t-suffix")})
        with self.assertRaises(ValueError):
            self.m.generate("missing", {})

    def test_stable_ids_and_order_independence(self):
        r1 = self.m.analyze("akh")
        r2 = Morphology(M_ENTRIES, M_AUTOMATON, ATOMS_M).analyze("akh")
        self.assertEqual([r["analysis_id"] for r in r1],
                         [r["analysis_id"] for r in r2])
        # entries 输入顺序改变不改变语义与 id
        m_rev = Morphology(list(reversed(M_ENTRIES)), M_AUTOMATON, ATOMS_M)
        a = sorted(x["analysis_id"] for x in self.m.analyze("zab"))
        b = sorted(x["analysis_id"] for x in m_rev.analyze("zab"))
        self.assertEqual(a, b)
        # transitions 输入顺序改变同样不改变语义
        aut_rev = dict(M_AUTOMATON,
                       transitions=list(reversed(M_AUTOMATON["transitions"])))
        m_trev = Morphology(M_ENTRIES, aut_rev, ATOMS_M)
        c = sorted(x["analysis_id"] for x in m_trev.analyze("zab"))
        self.assertEqual(a, c)

    def test_zero_transform_is_accepting(self):
        # start 本身可接受：零条边路径
        rows = self.m.analyze("cd")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transitions"], [])
        self.assertEqual(rows[0]["weight"], 1.5)

    def test_replace_tail_mismatch_blocks_edge(self):
        # old 非空且当前串不以 old 结束 => 边不可走
        entries = [{"id": "e", "stem": "xyz", "cat": "C",
                    "features": {}, "weight": 1.0}]
        automaton = {"start": 0, "accept": [0, 1], "transitions": [
            {"id": "need-ab", "from": 0, "to": 1, "requires": {},
             "set_features": {}, "replace_tail": ["ab", "k"],
             "prefix": "", "suffix": "", "weight": 1.0},
        ]}
        mm = Morphology(entries, automaton, ATOMS_M)
        self.assertEqual([r["transitions"] for r in mm.generate("e", {})], [[]])
        self.assertEqual(mm.analyze("xyk"), [])

    def test_rejects_cycle_and_overlong_path(self):
        # 声明的环即非法，即使该环在某 entry 数据上不可达
        cyclic = {"start": 0, "accept": [0, 1], "transitions": [
            {"id": "a", "from": 0, "to": 1, "requires": {}, "set_features": {},
             "replace_tail": ["", ""], "prefix": "x", "suffix": "", "weight": 1.0},
            {"id": "b", "from": 1, "to": 0, "requires": {}, "set_features": {},
             "replace_tail": ["", ""], "prefix": "y", "suffix": "", "weight": 1.0},
        ]}
        with self.assertRaises(ValueError):
            Morphology(M_ENTRIES, cyclic, ATOMS_M)

    def test_rejects_bad_shapes(self):
        good_entry = M_ENTRIES[0]
        with self.assertRaises(ValueError):                       # 重复 entry id
            Morphology([good_entry, dict(good_entry)], M_AUTOMATON, ATOMS_M)
        with self.assertRaises(ValueError):                       # 原子不在值域
            Morphology([dict(good_entry, features={"f": "zzz"})],
                       M_AUTOMATON, ATOMS_M)
        with self.assertRaises(ValueError):                       # 权重越界
            Morphology([dict(good_entry, weight=9.0)], M_AUTOMATON, ATOMS_M)
        with self.assertRaises(ValueError):                       # 重复 transition id
            bad = dict(M_AUTOMATON, transitions=M_AUTOMATON["transitions"] + [
                dict(M_AUTOMATON["transitions"][0])])
            Morphology(M_ENTRIES, bad, ATOMS_M)
        with self.assertRaises(ValueError):                       # 空 accept
            Morphology(M_ENTRIES, {"start": 0, "accept": [],
                                   "transitions": []}, ATOMS_M)


# --------------------------------------------------------------------------- #
# 句法手算小语法
# --------------------------------------------------------------------------- #
ATOMS_S = ["a", "b"]

# 两条接地前即可见的二元规则；变量 ?n 跨两个子节点共享。
HAND_GRAMMAR = {
    "start": {"cat": "S", "features": {}},
    "atoms": ATOMS_S,
    "rules": [
        {"id": "r1", "weight": 2.0, "lhs": {"cat": "S", "features": {}},
         "rhs": [{"cat": "A", "features": {"f": "?n"}},
                 {"cat": "B", "features": {"f": "?n"}}]},
        {"id": "r2", "weight": 3.0, "lhs": {"cat": "S", "features": {}},
         "rhs": [{"cat": "A", "features": {}},
                 {"cat": "B", "features": {}}]},
    ],
}

# 位置0两个候选（同 cat，f 不同），位置1一个候选。
HAND_CANDS = [
    [
        {"analysis_id": "x1", "cat": "A", "features": {"f": "a"}, "weight": 1.0},
        {"analysis_id": "x2", "cat": "A", "features": {"f": "b"}, "weight": 2.0},
    ],
    [
        {"analysis_id": "y1", "cat": "B", "features": {"f": "a"}, "weight": 3.5},
    ],
]


class TestSyntaxHandCalc(unittest.TestCase):
    def test_partition_marginals_best(self):
        # 手算完整接地与推导（y1 只有 f=a）：
        #   r1 接地 ?n=a: x1*y1*2 = 1*3.5*2 = 7
        #   r1 接地 ?n=b: 需要 B f=b，y1 不满足 => 0
        #   r2 无特征约束: (x1+x2)*y1*3 = 3*3.5*3 = 31.5
        # 分区 Z = 7 + 31.5 = 38.5
        res = analyze_sentence(HAND_GRAMMAR, HAND_CANDS)
        self.assertEqual(res["status"], "ok")
        self.assertAlmostEqual(res["partition_weight"], 38.5, places=9)
        self.assertAlmostEqual(res["log_partition"], math.log(38.5), places=9)
        # 后验：x1 参与全部 7+31.5 中 (7 + 1*3.5*3=10.5)=17.5 质量
        m0 = res["marginals"][0]
        self.assertAlmostEqual(m0["x1"], 17.5 / 38.5, places=9)
        self.assertAlmostEqual(m0["x2"], (2 * 3.5 * 3) / 38.5, places=9)
        self.assertAlmostEqual(sum(m0.values()), 1.0, places=9)
        self.assertAlmostEqual(res["marginals"][1]["y1"], 1.0, places=9)
        # 最优单树：r2 + x2 + y1 = 2*3.5*3 = 21
        self.assertAlmostEqual(res["best_log_weight"], math.log(21.0), places=9)
        best = res["best"]
        self.assertEqual(best["rule_id"], "r2")
        self.assertEqual([c["analysis_id"] for c in best["children"]], ["x2", "y1"])
        self.assertAlmostEqual(best["weight"], 21.0, places=9)
        # 树携带实际规则 id / 变量赋值 / 叶 analysis_id
        self.assertIn("assignment", best)

    def test_agreement_rejects_number_mismatch(self):
        grammar = {
            "start": {"cat": "S", "features": {}}, "atoms": ["sg", "pl"],
            "rules": [
                {"id": "agr", "weight": 1.0, "lhs": {"cat": "S", "features": {}},
                 "rhs": [{"cat": "N", "features": {"number": "?n"}},
                         {"cat": "V", "features": {"number": "?n"}}]},
            ],
        }
        ok = analyze_sentence(grammar, [
            [{"analysis_id": "n", "cat": "N", "features": {"number": "pl"},
              "weight": 1.0}],
            [{"analysis_id": "v", "cat": "V", "features": {"number": "pl"},
              "weight": 1.0}],
        ])
        self.assertEqual(ok["status"], "ok")
        bad = analyze_sentence(grammar, [
            [{"analysis_id": "n", "cat": "N", "features": {"number": "sg"},
              "weight": 1.0}],
            [{"analysis_id": "v", "cat": "V", "features": {"number": "pl"},
              "weight": 1.0}],
        ])
        self.assertEqual(bad["status"], "no_parse")
        self.assertEqual(bad["best"], None)
        self.assertEqual(bad["marginals"], None)
        self.assertEqual(bad["log_partition"], float("-inf"))

    def test_shared_variable_binds_both_children_together(self):
        # 两个候选都同时有 a/b 值；r1 的 ?n 必须取同一个原子 => 只产生匹配的两棵
        grammar = {
            "start": {"cat": "S", "features": {}}, "atoms": ["a", "b"],
            "rules": [
                {"id": "r", "weight": 1.0, "lhs": {"cat": "S", "features": {}},
                 "rhs": [{"cat": "A", "features": {"f": "?n"}},
                         {"cat": "B", "features": {"f": "?n"}}]},
            ],
        }
        cands = [
            [{"analysis_id": "a1", "cat": "A", "features": {"f": "a"}, "weight": 1.0},
             {"analysis_id": "a2", "cat": "A", "features": {"f": "b"}, "weight": 1.0}],
            [{"analysis_id": "b1", "cat": "B", "features": {"f": "a"}, "weight": 1.0},
             {"analysis_id": "b2", "cat": "B", "features": {"f": "b"}, "weight": 1.0}],
        ]
        res = analyze_sentence(grammar, cands)
        # 仅 (a1,b1) 与 (a2,b2) 合法，Z=2
        self.assertAlmostEqual(res["partition_weight"], 2.0, places=9)
        self.assertAlmostEqual(res["marginals"][0]["a1"], 0.5, places=9)
        self.assertAlmostEqual(res["marginals"][0]["a2"], 0.5, places=9)
        self.assertAlmostEqual(res["marginals"][1]["b1"], 0.5, places=9)
        self.assertAlmostEqual(res["marginals"][1]["b2"], 0.5, places=9)
        # 树携带实际变量赋值（接地后仍保留规则使用身份）
        self.assertEqual(res["best"]["assignment"], {"?n": "a"})
        self.assertEqual(res["best"]["rule_id"], "r")

    def test_variable_isolation_between_rule_uses(self):
        # 两条一元规则各自声明同名 ?x；同一颗树两次使用时取值互不共享。
        grammar = {
            "start": {"cat": "S", "features": {}}, "atoms": ["a", "b"],
            "rules": [
                {"id": "u1", "weight": 1.0,
                 "lhs": {"cat": "T", "features": {}},
                 "rhs": [{"cat": "L", "features": {"f": "?x"}}]},
                {"id": "u2", "weight": 1.0,
                 "lhs": {"cat": "S", "features": {}},
                 "rhs": [{"cat": "T", "features": {}}]},
                {"id": "top", "weight": 1.0,
                 "lhs": {"cat": "S", "features": {}},
                 "rhs": [{"cat": "L", "features": {}},
                         {"cat": "L", "features": {}}]},
            ],
        }
        cands = [
            [{"analysis_id": "l1", "cat": "L", "features": {"f": "a"}, "weight": 1.0}],
            [{"analysis_id": "l2", "cat": "L", "features": {"f": "b"}, "weight": 1.0}],
        ]
        res = analyze_sentence(grammar, cands)
        self.assertEqual(res["status"], "ok")
        # top 推导 1 条；u2(u1(l1)) 只能吃位置0 的单叶 span，整句覆盖需要二元 top，
        # 这里主要验证不同规则使用的同名变量可分别取 a 与 b 而不报错、不互相约束。
        self.assertTrue(res["partition_weight"] >= 1.0)

    def test_external_candidates_and_empty_position(self):
        # 句法入口独立接受无法由本地词典生成的外部候选
        res = analyze_sentence(HAND_GRAMMAR, [
            [{"analysis_id": "ext", "cat": "A", "features": {"f": "a"},
              "weight": 243.0}],
            [{"analysis_id": "ext2", "cat": "B", "features": {"f": "a"},
              "weight": 1e-5}],
        ])
        self.assertEqual(res["status"], "ok")
        # 空候选列表是合法的无解析输入
        no = analyze_sentence(HAND_GRAMMAR, [[]])
        self.assertEqual(no["status"], "no_parse")

    def test_duplicate_identity_kept_separate(self):
        # 同位置两个完全同形同特征但身份不同的候选，质量分别计
        cands = [
            [{"analysis_id": "d1", "cat": "A", "features": {"f": "a"}, "weight": 1.0},
             {"analysis_id": "d2", "cat": "A", "features": {"f": "a"}, "weight": 1.0}],
            [{"analysis_id": "b", "cat": "B", "features": {"f": "a"}, "weight": 1.0}],
        ]
        res = analyze_sentence(HAND_GRAMMAR, cands)
        # r1(?n=a): 2 个身份各 1*1*2 => 4；r2: 2*1*3 = 6；Z=10
        self.assertAlmostEqual(res["partition_weight"], 10.0, places=9)
        self.assertAlmostEqual(res["marginals"][0]["d1"], 0.5, places=9)
        self.assertAlmostEqual(res["marginals"][0]["d2"], 0.5, places=9)

    def test_true_tie_between_distinct_trees(self):
        grammar = {
            "start": {"cat": "S", "features": {}}, "atoms": ["a"],
            "rules": [
                {"id": "r1", "weight": 1.0, "lhs": {"cat": "S", "features": {}},
                 "rhs": [{"cat": "A", "features": {}}, {"cat": "B", "features": {}}]},
                {"id": "r2", "weight": 1.0, "lhs": {"cat": "S", "features": {}},
                 "rhs": [{"cat": "A", "features": {}}, {"cat": "B", "features": {}}]},
            ],
        }
        cands = [
            [{"analysis_id": "a", "cat": "A", "features": {}, "weight": 1.0}],
            [{"analysis_id": "b", "cat": "B", "features": {}, "weight": 1.0}],
        ]
        res = analyze_sentence(grammar, cands)
        self.assertAlmostEqual(res["partition_weight"], 2.0, places=9)
        # 两棵树质量都是 1，真正并列；任选其一但结果确定
        self.assertAlmostEqual(res["best"]["weight"], 1.0, places=9)
        self.assertIn(res["best"]["rule_id"], {"r1", "r2"})
        first = res["best"]["rule_id"]
        res2 = analyze_sentence(grammar, cands)
        self.assertEqual(res2["best"]["rule_id"], first)

    def test_missing_feature_is_not_wildcard(self):
        grammar = {
            "start": {"cat": "S", "features": {}}, "atoms": ["a"],
            "rules": [
                {"id": "r", "weight": 1.0, "lhs": {"cat": "S", "features": {}},
                 "rhs": [{"cat": "A", "features": {"f": "a"}},
                         {"cat": "B", "features": {}}]},
            ],
        }
        # A 候选缺 f 属性：规则要求 f 时必须实际提供 => no_parse
        res = analyze_sentence(grammar, [
            [{"analysis_id": "a", "cat": "A", "features": {}, "weight": 1.0}],
            [{"analysis_id": "b", "cat": "B", "features": {}, "weight": 1.0}],
        ])
        self.assertEqual(res["status"], "no_parse")
        # 子节点多出的特征不冲突
        res2 = analyze_sentence(grammar, [
            [{"analysis_id": "a", "cat": "A",
              "features": {"f": "a", "extra": "a"}, "weight": 1.0}],
            [{"analysis_id": "b", "cat": "B",
              "features": {"g": "a"}, "weight": 1.0}],
        ])
        self.assertEqual(res2["status"], "ok")

    def test_rejects_unary_category_cycle(self):
        grammar = {
            "start": {"cat": "S", "features": {}}, "atoms": ["a"],
            "rules": [
                {"id": "u1", "weight": 1.0, "lhs": {"cat": "X", "features": {}},
                 "rhs": [{"cat": "Y", "features": {}}]},
                {"id": "u2", "weight": 1.0, "lhs": {"cat": "Y", "features": {}},
                 "rhs": [{"cat": "X", "features": {}}]},
            ],
        }
        with self.assertRaises(ValueError):
            analyze_sentence(grammar, [[
                {"analysis_id": "x", "cat": "X", "features": {}, "weight": 1.0}]])

    def test_rejects_self_loop_even_when_data_unreachable(self):
        # 自环；即使特征使该一元使用在给定数据上不可达，声明的类别环仍非法
        grammar = {
            "start": {"cat": "S", "features": {}}, "atoms": ["a", "b"],
            "rules": [
                {"id": "self", "weight": 1.0,
                 "lhs": {"cat": "X", "features": {"f": "a"}},
                 "rhs": [{"cat": "X", "features": {"f": "b"}}]},
                {"id": "top", "weight": 1.0,
                 "lhs": {"cat": "S", "features": {}},
                 "rhs": [{"cat": "X", "features": {}}]},
            ],
        }
        with self.assertRaises(ValueError):
            analyze_sentence(grammar, [[
                {"analysis_id": "x", "cat": "X", "features": {"f": "b"},
                 "weight": 1.0}]])

    def test_unary_dag_chain_diamond_and_outside(self):
        atoms = ["x"]
        # 一元链 C -> B -> A -> S：拓扑序单趟闭合，唯一推导 7*2*3*0.5=21
        chain = {
            "start": {"cat": "S", "features": {}}, "atoms": atoms,
            "rules": [
                {"id": "c2b", "weight": 2.0, "lhs": {"cat": "B", "features": {}},
                 "rhs": [{"cat": "C", "features": {}}]},
                {"id": "b2a", "weight": 3.0, "lhs": {"cat": "A", "features": {}},
                 "rhs": [{"cat": "B", "features": {}}]},
                {"id": "a2s", "weight": 0.5, "lhs": {"cat": "S", "features": {}},
                 "rhs": [{"cat": "A", "features": {}}]},
            ],
        }
        leaf = [[{"analysis_id": "leaf", "cat": "C", "features": {},
                  "weight": 7.0}]]
        r = analyze_sentence(chain, leaf)
        self.assertAlmostEqual(r["partition_weight"], 21.0, places=9)
        self.assertAlmostEqual(r["best"]["weight"], 21.0, places=9)
        self.assertAlmostEqual(r["marginals"][0]["leaf"], 1.0, places=9)

        # 菱形：C 经 B1/B2 两条一元路径到 A，inside 累加（42+35=77），最优 42
        diamond = {
            "start": {"cat": "A", "features": {}}, "atoms": atoms,
            "rules": [
                {"id": "c2b1", "weight": 2.0, "lhs": {"cat": "B1", "features": {}},
                 "rhs": [{"cat": "C", "features": {}}]},
                {"id": "c2b2", "weight": 2.5, "lhs": {"cat": "B2", "features": {}},
                 "rhs": [{"cat": "C", "features": {}}]},
                {"id": "b12a", "weight": 3.0, "lhs": {"cat": "A", "features": {}},
                 "rhs": [{"cat": "B1", "features": {}}]},
                {"id": "b22a", "weight": 2.0, "lhs": {"cat": "A", "features": {}},
                 "rhs": [{"cat": "B2", "features": {}}]},
            ],
        }
        r2 = analyze_sentence(diamond, leaf)
        self.assertAlmostEqual(r2["partition_weight"], 77.0, places=9)
        self.assertAlmostEqual(math.exp(r2["best_log_weight"]), 42.0, places=9)

        # 一元 + 二元混合：outside 经一元链回传到叶，后验按整句质量分配
        mixed = {
            "start": {"cat": "S", "features": {}}, "atoms": atoms,
            "rules": [
                {"id": "c2b", "weight": 2.0, "lhs": {"cat": "B", "features": {}},
                 "rhs": [{"cat": "C", "features": {}}]},
                {"id": "join", "weight": 3.0, "lhs": {"cat": "S", "features": {}},
                 "rhs": [{"cat": "B", "features": {}},
                         {"cat": "D", "features": {}}]},
            ],
        }
        cands3 = [
            [{"analysis_id": "c", "cat": "C", "features": {}, "weight": 7.0}],
            [{"analysis_id": "d1", "cat": "D", "features": {}, "weight": 1.0},
             {"analysis_id": "d2", "cat": "D", "features": {}, "weight": 2.0}],
        ]
        r3 = analyze_sentence(mixed, cands3)
        self.assertAlmostEqual(r3["partition_weight"], 126.0, places=9)
        self.assertAlmostEqual(r3["marginals"][1]["d1"], 42 / 126, places=9)
        self.assertAlmostEqual(r3["marginals"][1]["d2"], 84 / 126, places=9)

    def test_unary_closure_with_multiple_ground_labels(self):
        # 同一类别 B 在闭合中产生两个不同接地 label（带/不带 mark），
        # 新 label 必须在同一趟拓扑闭合中继续被上层规则消费。
        grammar = {
            "start": {"cat": "S", "features": {}}, "atoms": ["a"],
            "rules": [
                {"id": "r1", "weight": 2.0, "lhs": {"cat": "B", "features": {}},
                 "rhs": [{"cat": "C", "features": {}}]},
                {"id": "r2", "weight": 2.0, "lhs": {"cat": "A", "features": {}},
                 "rhs": [{"cat": "B", "features": {}}]},
                {"id": "r3", "weight": 2.0, "lhs": {"cat": "S", "features": {}},
                 "rhs": [{"cat": "A", "features": {}}]},
                {"id": "r1b", "weight": 1.5,
                 "lhs": {"cat": "B", "features": {"mark": "a"}},
                 "rhs": [{"cat": "C", "features": {}}]},
                {"id": "r2b", "weight": 1.5, "lhs": {"cat": "A", "features": {}},
                 "rhs": [{"cat": "B", "features": {"mark": "a"}}]},
            ],
        }
        # 手算：B{}=2.0，B{mark}=1.5；
        # A{} = r2(B{}) 4.0 + r2b(B{mark}) 2.25 + r2 也子集接受 B{mark} 3.0 = 9.25；Z=18.5
        r = analyze_sentence(grammar, [[
            {"analysis_id": "c", "cat": "C", "features": {}, "weight": 1.0}]])
        self.assertAlmostEqual(r["partition_weight"], 18.5, places=9)
        self.assertAlmostEqual(r["best"]["weight"], 8.0, places=9)
        self.assertAlmostEqual(r["marginals"][0]["c"], 1.0, places=9)

    def test_rejects_bad_grammar_shapes(self):
        base = HAND_GRAMMAR
        with self.assertRaises(ValueError):                  # start 含变量
            analyze_sentence(dict(base, start={"cat": "S", "features": {"f": "?n"}}),
                             HAND_CANDS)
        with self.assertRaises(ValueError):                  # 规则 id 重复
            bad = dict(base, rules=base["rules"] + [dict(base["rules"][0])])
            analyze_sentence(bad, HAND_CANDS)
        with self.assertRaises(ValueError):                  # 候选原子不在值域
            analyze_sentence(base, [
                [{"analysis_id": "a", "cat": "A", "features": {"f": "zz"},
                  "weight": 1.0}],
                [{"analysis_id": "b", "cat": "B", "features": {}, "weight": 1.0}],
            ])
        with self.assertRaises(ValueError):                  # 位置内 id 重复
            analyze_sentence(base, [
                [{"analysis_id": "a", "cat": "A", "features": {}, "weight": 1.0},
                 {"analysis_id": "a", "cat": "A", "features": {}, "weight": 1.0}],
                [{"analysis_id": "b", "cat": "B", "features": {}, "weight": 1.0}],
            ])
        with self.assertRaises(ValueError):                  # 权重越界
            analyze_sentence(base, [
                [{"analysis_id": "a", "cat": "A", "features": {}, "weight": 0.0}],
                [{"analysis_id": "b", "cat": "B", "features": {}, "weight": 1.0}],
            ])


# --------------------------------------------------------------------------- #
# 组合入口
# --------------------------------------------------------------------------- #
class TestPipeline(unittest.TestCase):
    def _toy(self):
        atoms = ["sg", "pl", "reg"]
        entries = [
            {"id": "cat", "stem": "cat", "cat": "N",
             "features": {"number": "sg", "pattern": "reg"}, "weight": 1.0},
            {"id": "sleep", "stem": "sleep", "cat": "V",
             "features": {"number": "pl"}, "weight": 1.0},
        ]
        automaton = {"start": 0, "accept": [0, 1], "transitions": [
            {"id": "plural-s", "from": 0, "to": 1,
             "requires": {"number": "sg"}, "set_features": {"number": "pl"},
             "replace_tail": ["", ""], "prefix": "", "suffix": "s", "weight": 1.0},
        ]}
        grammar = {
            "start": {"cat": "S", "features": {}}, "atoms": atoms,
            "rules": [
                {"id": "sentence", "weight": 1.0,
                 "lhs": {"cat": "S", "features": {}},
                 "rhs": [{"cat": "N", "features": {"number": "?n"}},
                         {"cat": "V", "features": {"number": "?n"}}]},
            ],
        }
        return Morphology(entries, automaton, atoms), grammar

    def test_composition_uses_all_morph_candidates(self):
        morph, grammar = self._toy()
        # cats 只有复数分析；sleep 是复数动词 => ?n=pl 一致，合法
        res = analyze_tokens(morph, grammar, ["cats", "sleep"])
        self.assertEqual(res["status"], "ok")
        self.assertEqual(len(res["morph_analyses"][0]), 1)
        self.assertEqual(res["marginals"][0][res["candidates"][0][0]["analysis_id"]], 1.0)
        # cat（单数零变换）+ sleep（复数动词）=> 数不一致，no_parse
        res2 = analyze_tokens(morph, grammar, ["cat", "sleep"])
        self.assertEqual(res2["status"], "no_parse")

    def test_adding_homograph_changes_partition(self):
        morph, grammar = self._toy()
        before = analyze_tokens(morph, grammar, ["cats", "sleep"])
        # 合法增加一个同形的复数词典分析（不同 entry 身份、不同权重）
        morph2 = Morphology(
            morph._entries.values() and [
                dict(e) for e in morph._entries.values()
            ] + [{"id": "cat-lex-pl", "stem": "cats", "cat": "N",
                  "features": {"number": "pl", "pattern": "reg"}, "weight": 0.25}],
            {"start": 0, "accept": [0, 1], "transitions": [
                {"id": "plural-s", "from": 0, "to": 1,
                 "requires": {"number": "sg"}, "set_features": {"number": "pl"},
                 "replace_tail": ["", ""], "prefix": "", "suffix": "s",
                 "weight": 1.0}]},
            ["sg", "pl", "reg"],
        )
        after = analyze_tokens(morph2, grammar, ["cats", "sleep"])
        self.assertEqual(after["status"], "ok")
        self.assertEqual(len(after["morph_analyses"][0]), 2)
        self.assertNotAlmostEqual(after["log_partition"], before["log_partition"],
                                  places=9)
        # 新候选的后验由整句分区决定，且两个身份都保留
        ids = {c["analysis_id"] for c in after["candidates"][0]}
        self.assertEqual(len(ids), 2)
        self.assertAlmostEqual(sum(after["marginals"][0].values()), 1.0, places=9)
        # 消费处继续使用旧候选 => 分区仍是旧值（对照给“错”结果）
        stale = analyze_sentence(grammar, before["candidates"])
        self.assertAlmostEqual(stale["log_partition"], before["log_partition"], places=9)
        self.assertNotAlmostEqual(stale["log_partition"], after["log_partition"],
                                  places=9)

    def test_too_many_analyses_rejected_not_truncated(self):
        # 构造 9 个落到同形的 entry
        entries = [
            {"id": f"e{i}", "stem": "w", "cat": "N",
             "features": {"number": "sg"}, "weight": 1.0}
            for i in range(9)
        ]
        automaton = {"start": 0, "accept": [0], "transitions": []}
        morph = Morphology(entries, automaton, ["sg", "pl"])
        grammar = {"start": {"cat": "S", "features": {}}, "atoms": ["sg", "pl"],
                   "rules": [{"id": "r", "weight": 1.0,
                              "lhs": {"cat": "S", "features": {}},
                              "rhs": [{"cat": "N", "features": {}}]}]}
        # 独立词形查询仍返回全部 9 个
        self.assertEqual(len(morph.analyze("w")), 9)
        with self.assertRaises(ValueError):
            analyze_tokens(morph, grammar, ["w"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
