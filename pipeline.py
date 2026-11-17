"""组合入口：同一个句法入口实际使用本次词形分析的全部候选。

analyze_tokens(morphology, grammar, tokens)
    对每个 token 调用本次 Morphology.analyze，保留得到的全部分析（entry 身份、
    路径 transition 序列、完整特征、独立权重），原样传给同一个 analyze_sentence。
    不按单复数词尾重猜特征，不另选一套未交付的词形规则，不截断/去重。
"""

from __future__ import annotations

from syntax import MAX_CANDIDATES, analyze_sentence

__all__ = ["analyze_tokens"]


def _to_candidate(analysis):
    """词形分析记录 -> 句法候选；analysis_id 在位置内唯一由词形模块保证。"""
    return {
        "analysis_id": analysis["analysis_id"],
        "cat": analysis["cat"],
        "features": dict(analysis["features"]),
        "weight": analysis["weight"],
    }


def analyze_tokens(morphology, grammar, tokens):
    if not isinstance(tokens, (list, tuple)) or not tokens:
        raise ValueError("tokens 必须是非空列表")
    candidates, per_token = [], []
    for tok in tokens:
        if not isinstance(tok, str) or not tok:
            raise ValueError(f"token 必须是非空字符串，得到 {tok!r}")
        analyses = morphology.analyze(tok)
        # 组合入口的合法输入限定为每个 token 至多 8 个分析；超出即非法，
        # 绝不通过截断或去重“修好”，也不改变独立词形查询可返回更多分析的事实。
        if len(analyses) > MAX_CANDIDATES:
            raise ValueError(
                f"token {tok!r} 有 {len(analyses)} 个词形分析，"
                f"组合入口上限为 {MAX_CANDIDATES}；独立 analyze 调用结果不受此限"
            )
        per_token.append(analyses)
        candidates.append([_to_candidate(a) for a in analyses])
    result = analyze_sentence(grammar, candidates)
    # 回传实际使用的候选，便于审计“同一句法入口看到了哪些词形分析”。
    result["candidates"] = candidates
    result["morph_analyses"] = per_token
    return result
