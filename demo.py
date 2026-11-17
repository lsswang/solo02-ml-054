"""可重复运行的端到端演示：词形变换、特征合一句法、组合入口与对照。

运行：python demo.py
产出：fixtures/demo_results.json（结构化结果，含手算核对值），并在终端打印摘要。
本演示是教学规模的合成语法，不声称任何真实语言解析精度。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from morph import Morphology
from syntax import analyze_sentence
from pipeline import analyze_tokens

HERE = Path(__file__).parent
DATA = json.loads((HERE / "fixtures" / "demo_data.json").read_text(encoding="utf-8"))
RESULTS_PATH = HERE / "fixtures" / "demo_results.json"


def build_morph(entries=None, automaton=None, atoms=None):
    return Morphology(
        entries if entries is not None else DATA["entries"],
        automaton if automaton is not None else DATA["automaton"],
        atoms if atoms is not None else DATA["atoms"],
    )


def section_morphology(morph):
    """场景 1-5：零变换、尾替换+前缀、accept 继续扩展、同形多路径、不可用 requires。"""
    out = {}

    # 1) 普通 cats/sleep：零次变换与简单后缀
    out["cats_analyses"] = morph.analyze("cats")
    out["sleep_analyses"] = morph.analyze("sleep")

    # 2) 词干尾替换后再加前缀：party -> parties（y->ies），再 un- 前缀（accept 继续走）
    out["parties_analyses"] = morph.analyze("parties")
    out["unparties_analyses"] = morph.analyze("unparties")

    # 3) generate 展示 happy 的尾替换与 accept 状态 1 上继续接 un- 前缀
    out["happy_generation"] = morph.generate("happy", {})

    # 4) 同形多路径：sheep 两个 entry（sg/pl）零变换得到同一串，身份与权重分别保留
    out["sheep_analyses"] = morph.analyze("sheep")

    # 5) 不可用 requires：dead-gate 要求 extra=y，没有任何 entry/边提供 extra 属性，
    #    门控永远不满足，X 前缀串不会被生成
    out["dead_edge_surfaces_absent"] = {
        "Xsheep": morph.analyze("Xsheep"),
        "Xcat": morph.analyze("Xcat"),
    }
    return out


def section_syntax(morph):
    """场景 6-9：数一致、变量共享/隔离、外部候选、无解析、真正并列树。"""
    out = {}
    grammar = DATA["agreement_grammar"]

    # 6) 词形歧义 + 数一致：sheep 单/复数同形，sleep 为复数动词；
    #    组合入口把两个身份都送入句法，只有复数身份能与动词一致。
    composed = analyze_tokens(morph, grammar, ["sheep", "sleep"])
    out["agreement_composition"] = {
        "status": composed["status"],
        "log_partition": composed["log_partition"],
        "best_log_weight": composed["best_log_weight"],
        "marginals": composed["marginals"],
        "best": composed["best"],
        "candidate_count_per_token": [len(p) for p in composed["morph_analyses"]],
    }

    # 7) 规则变量跨两个子节点共享：cat(sg) + run(sg 动词) 必须同时取 ?n=sg
    ok_sg = analyze_tokens(morph, grammar, ["cat", "run"])
    #    数不一致：cat(sg) + sleep(pl) => no_parse
    mismatch = analyze_tokens(morph, grammar, ["cat", "sleep"])
    out["shared_variable"] = {
        "singular_agreement_status": ok_sg["status"],
        "singular_marginals": ok_sg["marginals"],
        "mismatch_status": mismatch["status"],
        "mismatch_log_partition": mismatch["log_partition"],
    }

    # 8) 外部候选（无法由本地词典生成）句法入口独立接受
    external = analyze_sentence(grammar, [
        [{"analysis_id": "ext-n-pl", "cat": "N",
          "features": {"number": "pl"}, "weight": 0.3},
         {"analysis_id": "ext-n-sg", "cat": "N",
          "features": {"number": "sg"}, "weight": 0.7}],
        [{"analysis_id": "ext-v-pl", "cat": "V",
          "features": {"number": "pl"}, "weight": 1.0}],
    ])
    out["external_candidates"] = {
        "status": external["status"],
        "log_partition": external["log_partition"],
        "marginals": external["marginals"],
        "hand_check": {
            "only_pl_agrees": True,
            "partition_weight": 0.3,
            "marginal_ext_n_pl": 1.0,
            "marginal_ext_n_sg": 0.0,
        },
    }

    # 9) 真正并列树：两条同权规则、同样的叶，两棵完整推导质量相等
    tie_grammar = {
        "start": {"cat": "S", "features": {}}, "atoms": ["a"],
        "rules": [
            {"id": "way-one", "weight": 1.0,
             "lhs": {"cat": "S", "features": {}},
             "rhs": [{"cat": "A", "features": {}}, {"cat": "B", "features": {}}]},
            {"id": "way-two", "weight": 1.0,
             "lhs": {"cat": "S", "features": {}},
             "rhs": [{"cat": "A", "features": {}}, {"cat": "B", "features": {}}]},
        ],
    }
    tie = analyze_sentence(tie_grammar, [
        [{"analysis_id": "leaf-a", "cat": "A", "features": {}, "weight": 1.0}],
        [{"analysis_id": "leaf-b", "cat": "B", "features": {}, "weight": 1.0}],
    ])
    out["true_tie"] = {
        "partition_weight": tie["partition_weight"],
        "best_rule_id": tie["best"]["rule_id"],
        "best_log_weight": tie["best_log_weight"],
        "hand_check": {"two_derivations_each_weight_1": True, "partition": 2.0},
    }

    # 10) 变量隔离：不同规则的同名变量各自接地取值，互不共享
    iso_grammar = {
        "start": {"cat": "S", "features": {}}, "atoms": ["sg", "pl"],
        "rules": [
            {"id": "np", "weight": 1.0,
             "lhs": {"cat": "NP", "features": {"number": "?n"}},
             "rhs": [{"cat": "D", "features": {"number": "?n"}},
                     {"cat": "N", "features": {"number": "?n"}}]},
            {"id": "sentence", "weight": 1.0,
             "lhs": {"cat": "S", "features": {}},
             "rhs": [{"cat": "NP", "features": {"number": "?n"}},
                     {"cat": "V", "features": {"number": "?n"}}]},
        ],
    }
    iso = analyze_sentence(iso_grammar, [
        [{"analysis_id": "d-pl", "cat": "D", "features": {"number": "pl"},
          "weight": 1.0}],
        [{"analysis_id": "n-pl", "cat": "N", "features": {"number": "pl"},
          "weight": 1.0}],
        [{"analysis_id": "v-pl", "cat": "V", "features": {"number": "pl"},
          "weight": 1.0}],
    ])
    out["variable_isolation"] = {
        "status": iso["status"],
        "rule_assignments_in_best": [
            {"rule_id": node["rule_id"], "assignment": node["assignment"]}
            for node in _walk_rules(iso["best"])
        ],
    }
    return out


def _walk_rules(node):
    if node["node"] == "leaf":
        return []
    out = [node]
    for ch in node["children"]:
        out.extend(_walk_rules(ch))
    return out


def section_lexicon_change(morph):
    """场景 11：合法增加同形词典分析 / 改变词形变换，如何改变分区与后验；
    并给出消费处继续使用旧候选的对照（旧候选 -> 旧分区，即“错”结果）。"""
    grammar = DATA["agreement_grammar"]
    before = analyze_tokens(morph, grammar, ["sheep", "sleep"])

    # 11a) 再增加一个同形复数 sheep 词典分析（新 entry 身份，权重 0.5）
    entries_plus = [dict(e) for e in DATA["entries"]] + [
        {"id": "sheep-pl-extra", "stem": "sheep", "cat": "N",
         "features": {"number": "pl"}, "weight": 0.5},
    ]
    morph_plus = build_morph(entries=entries_plus)
    after_entry = analyze_tokens(morph_plus, grammar, ["sheep", "sleep"])

    # 11b) 改变一条词形变换的权重（plural-s: 1.0 -> 0.25），影响 cats 的词形质量
    automaton_light = json.loads(json.dumps(DATA["automaton"]))
    for t in automaton_light["transitions"]:
        if t["id"] == "plural-s":
            t["weight"] = 0.25
    morph_light = build_morph(automaton=automaton_light)
    light = analyze_tokens(morph_light, grammar, ["cats", "sleep"])
    base = analyze_tokens(morph, grammar, ["cats", "sleep"])

    # 11c) 消费处仍使用 before 的旧候选：分区停留在旧值
    stale = analyze_sentence(grammar, before["candidates"])

    return {
        "before": {
            "log_partition": before["log_partition"],
            "sheep_candidates": before["morph_analyses"][0],
        },
        "after_adding_homograph": {
            "log_partition": after_entry["log_partition"],
            "sheep_candidate_count": len(after_entry["morph_analyses"][0]),
            "marginals": after_entry["marginals"],
            "hand_check": {
                "pl_mass_before": 2.0,
                "pl_mass_after": 2.0 + 0.5,
                "partition_tracks_full_mass": True,
            },
        },
        "after_edge_weight_change": {
            "edge": "plural-s 1.0 -> 0.25",
            "log_partition_light": light["log_partition"],
            "log_partition_base": base["log_partition"],
            "cats_analysis_weight_light": morph_light.analyze("cats")[0]["weight"],
            "cats_analysis_weight_base": morph.analyze("cats")[0]["weight"],
        },
        "stale_consumer_using_old_candidates": {
            "stale_log_partition": stale["log_partition"],
            "equals_before_not_after": (
                math.isclose(stale["log_partition"], before["log_partition"], rel_tol=1e-12)
                and not math.isclose(
                    stale["log_partition"], after_entry["log_partition"], rel_tol=1e-12)
            ),
        },
    }


def _strict_jsonable(obj):
    """把 float 的 +/-inf/NaN 转成 None，使落盘文件为严格 JSON（状态字段仍可判别）。"""
    if isinstance(obj, dict):
        return {k: _strict_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strict_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main():
    morph = build_morph()
    results = {
        "description": "教学规模合成词法/句法演示；不代表真实语言解析精度",
        "morphology": section_morphology(morph),
        "syntax": section_syntax(morph),
        "lexicon_change_and_stale_consumer": section_lexicon_change(morph),
    }
    RESULTS_PATH.write_text(
        json.dumps(_strict_jsonable(results), ensure_ascii=False, indent=2,
                   allow_nan=False),
        encoding="utf-8",
    )

    # 终端摘要
    print("== 词形：cats / parties / unparties / sheep ==")
    for word in ("cats", "parties", "unparties", "sheep"):
        rows = morph.analyze(word)
        print(f"  {word:10s} -> {len(rows)} 个分析: "
              + "; ".join(f"{r['entry_id']}{r['transitions']} w={r['weight']}"
                           for r in rows))
    print("\n== 数一致：sheep(sg/pl 同形) + sleep(pl) ==")
    sec = results["syntax"]["agreement_composition"]
    print(f"  status={sec['status']} logZ={sec['log_partition']:.6f}")
    print(f"  marginals={json.dumps(sec['marginals'], ensure_ascii=False)}")
    print("\n== 数不一致：cat(sg) + sleep(pl) ==")
    print(f"  status={results['syntax']['shared_variable']['mismatch_status']}")
    print("\n== 词典变更与陈旧候选对照 ==")
    ch = results["lexicon_change_and_stale_consumer"]
    print(f"  before logZ={ch['before']['log_partition']:.6f}")
    print(f"  after  logZ={ch['after_adding_homograph']['log_partition']:.6f}")
    print(f"  stale  logZ={ch['stale_consumer_using_old_candidates']['stale_log_partition']:.6f}"
          f" (沿用旧候选 => 旧分区)")
    print(f"\n结构化结果已写入 {RESULTS_PATH.relative_to(HERE)}")


if __name__ == "__main__":
    main()
