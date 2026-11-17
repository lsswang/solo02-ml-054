"""有限加权词形变换：无环变换自动机上的接受路径枚举。

接口
----
Morphology(entries, automaton, atoms)
    entries   : 若干词元条目，含唯一 id、stem、cat、features、正 weight
    automaton : 无环变换自动机，含 start、accept、transitions
    atoms     : 本次调用声明的有限原子值域

analyze(word)
    返回所有最终字符串逐字等于 word 的 (entry, 接受路径) 分析。

generate(entry_id, constraints)
    返回指定 entry 的全部接受路径中，最终特征满足 constraints（子集）者。

设计表示
--------
自动机无环且从 start 可达的最长路径 <= 4，接受路径集合有限。构造时即为每个
entry 枚举全部接受路径并缓存；analyze/generate 只做过滤，因此同一条 entry 的
不同路径、不同 entry 的同形结果都会作为独立分析保留，权重分别保留、不归一。
"""

from __future__ import annotations

import hashlib
import json
import math

__all__ = ["Morphology"]

MAX_STRING = 24          # stem / word / surface / 中间字符串的码点长度上界
MAX_PATH_EDGES = 4       # 从 start 可达的最长路径边数
MAX_STATES = 8
MAX_TRANSITIONS = 12
MAX_ENTRIES = 30
MAX_CATS = 8
MAX_ATTRS = 3
MAX_ATOMS = 4
WEIGHT_LO, WEIGHT_HI = 0.1, 3.0


# --------------------------------------------------------------------------- #
# 基础校验
# --------------------------------------------------------------------------- #
def _is_bool(x):
    return isinstance(x, bool)


def _number(value, name, lo, hi):
    """有限标量权重：拒绝 bool，要求落在 [lo, hi]。"""
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是有限数值，得到 {value!r}")
    value = float(value)
    if not math.isfinite(value) or not (lo <= value <= hi):
        raise ValueError(f"{name}={value!r} 超出有限区间 [{lo}, {hi}]")
    return value


def _text(value, name, *, allow_empty=False, max_len=MAX_STRING):
    """非空 Unicode 字符串（除非显式允许空），禁止未配对代理，限长。"""
    if not isinstance(value, str):
        raise ValueError(f"{name} 必须是字符串，得到 {type(value).__name__}")
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        raise ValueError(f"{name} 含未配对 surrogate：{value!r}")
    if not allow_empty and value == "":
        raise ValueError(f"{name} 不能为空字符串")
    if len(value) > max_len:
        raise ValueError(f"{name} 长度 {len(value)} 超过 {max_len} 个码点")
    return value


def _name(value, name):
    """ID / cat / 属性名 / 原子：非空字符串，不允许 '?' 开头。"""
    _text(value, name)
    if value.startswith("?"):
        raise ValueError(f"{name} 不能以 '?' 开头：{value!r}")
    return value


def _feature_map(obj, name, atoms, *, allow_variable=False):
    """校验特征映射，返回拷贝。原子值必须属于 atoms；句法侧允许 '?x' 变量。"""
    if not isinstance(obj, dict):
        raise ValueError(f"{name} 必须是对象/字典，得到 {type(obj).__name__}")
    out = {}
    for key, val in obj.items():
        _name(key, f"{name} 的属性名")
        if not isinstance(val, str) or val == "":
            raise ValueError(f"{name}.{key} 的值必须是非空字符串")
        if val.startswith("?"):
            if not allow_variable or len(val) == 1:
                raise ValueError(f"{name}.{key} 的变量 {val!r} 非法")
        elif val not in atoms:
            raise ValueError(f"{name}.{key} 的原子 {val!r} 不在声明值域 {atoms}")
        out[key] = val
    return out


def _stable_id(payload):
    """由完整语义资料生成稳定 analysis_id（与输入/字典插入顺序无关）。"""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "m-" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 内部记录
# --------------------------------------------------------------------------- #
class _Transition:
    __slots__ = ("tid", "src", "dst", "requires", "set_features",
                 "old", "new", "prefix", "suffix", "weight")

    def __init__(self, raw, atoms):
        tid = _name(raw.get("id"), "transition.id")
        for key in ("from", "to"):
            if _is_bool(raw.get(key)) or not isinstance(raw.get(key), int):
                raise ValueError(f"transition {tid} 的 {key} 必须是整数状态")
        self.tid = tid
        self.src = raw["from"]
        self.dst = raw["to"]

        self.requires = _feature_map(raw.get("requires", {}), f"transition {tid}.requires", atoms)
        self.set_features = _feature_map(
            raw.get("set_features", {}), f"transition {tid}.set_features", atoms
        )

        tail = raw.get("replace_tail")
        if not isinstance(tail, (list, tuple)) or len(tail) != 2:
            raise ValueError(f"transition {tid}.replace_tail 必须是 [old, new] 两个字符串")
        self.old = _text(tail[0], f"transition {tid}.replace_tail[0]", allow_empty=True)
        self.new = _text(tail[1], f"transition {tid}.replace_tail[1]", allow_empty=True)
        self.prefix = _text(raw.get("prefix", ""), f"transition {tid}.prefix", allow_empty=True)
        self.suffix = _text(raw.get("suffix", ""), f"transition {tid}.suffix", allow_empty=True)
        self.weight = _number(raw.get("weight"), f"transition {tid}.weight", WEIGHT_LO, WEIGHT_HI)


class _Path:
    """一条接受路径的完整语义资料。"""
    __slots__ = ("entry_id", "transitions", "surface", "cat", "features",
                 "weight", "analysis_id")

    def __init__(self, entry_id, transitions, surface, cat, features, weight):
        self.entry_id = entry_id
        self.transitions = tuple(transitions)
        self.surface = surface
        self.cat = cat
        self.features = dict(features)
        self.weight = float(weight)
        self.analysis_id = _stable_id({
            "entry_id": entry_id,
            "transitions": list(self.transitions),
            "surface": surface,
            "cat": cat,
            "features": self.features,
            # repr(float) 稳定且跨机器一致，避免 json 默认把 1.0 写成 1.0 的歧义
            "weight": repr(self.weight),
        })

    def to_dict(self):
        return {
            "analysis_id": self.analysis_id,
            "entry_id": self.entry_id,
            "transitions": list(self.transitions),
            "surface": self.surface,
            "cat": self.cat,
            "features": dict(self.features),
            "weight": self.weight,
        }


# --------------------------------------------------------------------------- #
# Morphology
# --------------------------------------------------------------------------- #
class Morphology:
    def __init__(self, entries, automaton, atoms):
        atoms = self._validate_atoms(atoms)
        self._atoms = tuple(atoms)
        self._entries = self._validate_entries(entries, atoms)
        self._start, self._accept, self._transitions, self._out = self._validate_automaton(
            automaton, atoms
        )
        # 每个 entry 的全部接受路径在构造时枚举并缓存。
        self._paths_by_entry = {
            eid: self._enumerate(entry) for eid, entry in self._entries.items()
        }

    # ---------------- 校验 ---------------- #
    @staticmethod
    def _validate_atoms(atoms):
        if not isinstance(atoms, (list, tuple)) or not (1 <= len(atoms) <= MAX_ATOMS):
            raise ValueError(f"atoms 必须含 1..{MAX_ATOMS} 个原子字符串")
        seen = set()
        for a in atoms:
            _name(a, "atom")
            if a in seen:
                raise ValueError(f"原子重复：{a!r}")
            seen.add(a)
        return list(atoms)

    @staticmethod
    def _validate_entries(entries, atoms):
        if not isinstance(entries, (list, tuple)) or not (1 <= len(entries) <= MAX_ENTRIES):
            raise ValueError(f"entries 必须含 1..{MAX_ENTRIES} 个条目")
        out = {}
        cats, attrs = set(), set()
        for raw in entries:
            if not isinstance(raw, dict):
                raise ValueError("entry 必须是对象/字典")
            eid = _name(raw.get("id"), "entry.id")
            if eid in out:
                raise ValueError(f"entry id 重复：{eid!r}")
            stem = _text(raw.get("stem"), f"entry {eid}.stem")
            cat = _name(raw.get("cat"), f"entry {eid}.cat")
            feats = _feature_map(raw.get("features", {}), f"entry {eid}.features", atoms)
            weight = _number(raw.get("weight"), f"entry {eid}.weight", WEIGHT_LO, WEIGHT_HI)
            out[eid] = {
                "id": eid, "stem": stem, "cat": cat,
                "features": feats, "weight": weight,
            }
            cats.add(cat)
            attrs.update(feats)
        if len(cats) > MAX_CATS:
            raise ValueError(f"不同 cat 至多 {MAX_CATS} 个，得到 {len(cats)}")
        if len(attrs) > MAX_ATTRS:
            raise ValueError(f"不同属性至多 {MAX_ATTRS} 个，得到 {len(attrs)}")
        return out

    def _validate_automaton(self, automaton, atoms):
        if not isinstance(automaton, dict):
            raise ValueError("automaton 必须是对象/字典")
        if _is_bool(automaton.get("start")) or not isinstance(automaton.get("start"), int):
            raise ValueError("automaton.start 必须是整数状态")
        start = automaton["start"]

        accept = automaton.get("accept")
        if not isinstance(accept, (list, tuple)) or not accept:
            raise ValueError("automaton.accept 必须是非空状态列表")
        if len(set(accept)) != len(accept):
            raise ValueError("automaton.accept 状态重复")

        raw_trans = automaton.get("transitions", [])
        if not isinstance(raw_trans, (list, tuple)) or not (0 <= len(raw_trans) <= MAX_TRANSITIONS):
            raise ValueError(f"transitions 数量必须在 0..{MAX_TRANSITIONS}")
        transitions = []
        tids = set()
        states = {start}
        for raw in raw_trans:
            t = _Transition(raw, atoms)
            if t.tid in tids:
                raise ValueError(f"transition id 重复：{t.tid!r}")
            tids.add(t.tid)
            transitions.append(t)
            states.update((t.src, t.dst))
        states.update(accept)
        if len(states) > MAX_STATES:
            raise ValueError(f"自动机状态至多 {MAX_STATES} 个，得到 {len(states)}")
        for st in (*accept, start):
            if st not in states:
                raise ValueError(f"状态 {st} 未在 transitions 中声明")
        for t in transitions:
            if t.src not in states or t.dst not in states:
                raise ValueError(f"transition {t.tid} 引用未知状态")

        out = {}
        for t in transitions:
            out.setdefault(t.src, []).append(t)

        # 仅在 start 可达子图上检查无环与最长路径 <= 4。
        reachable = set()
        stack = [start]
        while stack:
            st = stack.pop()
            if st in reachable:
                continue
            reachable.add(st)
            stack.extend(t.dst for t in out.get(st, []))
        indeg = {st: 0 for st in reachable}
        reach_edges = 0
        for t in transitions:
            if t.src in reachable:
                indeg[t.dst] = indeg.get(t.dst, 0) + 1
                reach_edges += 1
        kahn = [st for st in reachable if indeg[st] == 0]
        seen_cnt = 0
        for st in kahn:
            seen_cnt += 1
            for t in out.get(st, []):
                if t.dst in reachable:
                    indeg[t.dst] -= 1
                    if indeg[t.dst] == 0:
                        kahn.append(t.dst)
        if seen_cnt != len(reachable):
            raise ValueError("自动机 start 可达子图存在环，必须无环")

        # 最长路径边数（DAG 上的动态规划）。
        longest = {start: 0}
        for st in kahn:                       # Kahn 输出即拓扑序
            for t in out.get(st, []):
                if t.dst in reachable:
                    longest[t.dst] = max(longest.get(t.dst, 0), longest[st] + 1)
        if longest and max(longest.values()) > MAX_PATH_EDGES:
            raise ValueError(f"start 可达最长路径超过 {MAX_PATH_EDGES} 条边")

        # 属性总量约束（entry + requires + set_features 的并集）。
        attrs = set()
        for e in self._entries.values():
            attrs.update(e["features"])
        for t in transitions:
            attrs.update(t.requires)
            attrs.update(t.set_features)
        if len(attrs) > MAX_ATTRS:
            raise ValueError(f"不同属性至多 {MAX_ATTRS} 个，得到 {len(attrs)}")
        return start, set(accept), transitions, out

    # ---------------- 路径枚举 ---------------- #
    def _apply(self, string, feats, trans):
        """边语义：requires 门控 -> 尾替换 -> 前后缀 -> set_features 覆盖。"""
        for key, val in trans.requires.items():
            if feats.get(key) != val:
                return None
        if trans.old:
            if not string.endswith(trans.old):
                return None
            base = string[: len(string) - len(trans.old)]
        else:
            base = string
        new_string = trans.prefix + base + trans.new + trans.suffix
        if len(new_string) > MAX_STRING:
            raise ValueError(
                f"entry {self._cur_entry!r} 经路径 {trans.tid!r} 后字符串长度超过 {MAX_STRING}"
            )
        new_feats = dict(feats)
        new_feats.update(trans.set_features)
        return new_string, new_feats

    def _enumerate(self, entry):
        self._cur_entry = entry["id"]
        results = []
        # (状态, 转移 id 序列, 当前字符串, 当前特征, 累计权重)
        stack = [(self._start, (), entry["stem"], dict(entry["features"]), entry["weight"])]
        while stack:
            state, path, string, feats, weight = stack.pop()
            if state in self._accept:
                # start 本身可接受；accept 状态也可继续走边产生另一条接受路径。
                results.append(
                    _Path(entry["id"], path, string, entry["cat"], feats, weight)
                )
            if len(path) >= MAX_PATH_EDGES:
                continue
            for trans in self._out.get(state, []):
                applied = self._apply(string, feats, trans)
                if applied is None:
                    continue            # 条件不满足的边不可走
                new_string, new_feats = applied
                stack.append((
                    trans.dst,
                    path + (trans.tid,),
                    new_string,
                    new_feats,
                    weight * trans.weight,
                ))
        self._cur_entry = None
        return results

    # ---------------- 公共接口 ---------------- #
    def analyze(self, word):
        """所有最终字符串逐字等于 word 的分析；不同 entry/路径独立保留。"""
        _text(word, "word")
        out = []
        for eid in self._entries:                       # 字典保序，顺序本身不影响语义
            for path in self._paths_by_entry[eid]:
                if path.surface == word:
                    out.append(path.to_dict())
        return out

    def generate(self, entry_id, constraints):
        """该 entry 全部最终特征满足 constraints（子集）的接受路径。"""
        _name(entry_id, "entry_id")
        if entry_id not in self._entries:
            raise ValueError(f"未知 entry_id：{entry_id!r}")
        constraints = _feature_map(constraints or {}, "constraints", self._atoms)
        out = []
        for path in self._paths_by_entry[entry_id]:
            if all(path.features.get(k) == v for k, v in constraints.items()):
                out.append(path.to_dict())
        return out
