"""
PriorRG annotation preprocessing (expanded keywords + none label)
==================================================================

Output fields (added to each record):

    change_label : int
        0 = no_prior  -- 无既往影像（不参与 L_tc 监督）
        1 = new       -- 新出现的病变
        2 = worsened  -- 较前加重
        3 = improved  -- 较前好转
        4 = stable    -- 有显式 "stable / unchanged / similar" 等关键词
        5 = none      -- 有既往影像但报告里无任何匹配的变化关键词
                          → ★ 不参与 L_tc 监督（与 no_prior 一致），
                            避免给 CRM 注入伪 stable 信号

    has_prior : 0/1
        是否存在既往影像（dataloader 用作 prior_mask）

Training-side semantics
-----------------------
loss 端只对 label ∈ {1,2,3,4} 做监督；label ∈ {0, 5} 用 ignore_index=-100
屏蔽。这一与 no_prior 的语义切割，是这套预处理的核心动机。

CLI
---
    python section_split.py \
        --input  data-demo/priorrg_mimic_cxr_annotation.json \
        --output data-demo/priorrg_mimic_cxr_annotation_labeled.json
"""

import argparse
import json
import re
from collections import Counter
from typing import Dict, List


# ---------------------------------------------------------------------------
# Change-label keyword vocabulary
# ---------------------------------------------------------------------------
# Priority: new (1) > worsened (2) > improved (3) > stable (4)
# 即一个样本如果同时含 "new" 和 "stable"，将被打 label=1 (new)。
# 没有任何下列关键词命中的 "has_prior=True" 样本 → label=5 (none)。
CHANGE_KEYWORDS: Dict[int, List[str]] = {
    # ----- 1 : new -----
    1: [
        "new", "newly",
        "newly developed", "newly appeared", "newly noted",
        "newly identified", "newly visualized", "newly seen",
        "new onset", "new finding", "new appearance",
        "not previously seen", "not previously present",
        "not previously identified", "not previously visualized",
        "not present previously", "not seen previously",
        "not seen on prior", "not seen on the prior",
        "not present on prior", "not present on the prior",
        "not visualized previously", "not visualized on prior",
        "first time", "first noted",
        "interval development", "interval appearance",
        "appearance of", "emergence of",
        "emerged", "emerging",
        "recurrent", "recurrence",
    ],

    # ----- 2 : worsened -----
    2: [
        "worse", "worsened", "worsening",
        "increased", "increase", "increasing",
        "progression", "progressed", "progressing", "progressive",
        "enlarged", "enlarging", "enlargement", "larger", "expanded", "expansion",
        "interval increase", "interval enlargement",
        "interval worsening", "interval progression", "interval growth",
        "interval expansion", "interval development of",
        "exacerbation", "exacerbated",
        "more pronounced", "more prominent", "more extensive",
        "more conspicuous", "more apparent", "more visible", "more severe",
        "greater extent", "greater opacity", "greater consolidation",
        "now more", "now greater", "now larger",
        "accumulation", "accumulating", "accumulated",
        "advancing", "evolving",
    ],

    # ----- 3 : improved -----
    3: [
        "improved", "improvement", "improving",
        "decreased", "decrease", "decreasing",
        "regression", "regressed", "regressing", "regressive",
        "resolved", "resolving", "resolution",
        "near resolution", "near-complete resolution", "nearly resolved",
        "partial resolution", "partially resolved",
        "smaller", "reduced", "reducing", "reduction",
        "diminished", "diminishing",
        "interval decrease", "interval improvement",
        "interval resolution", "interval regression",
        "cleared", "clearing", "clearance",
        "less pronounced", "less prominent", "less extensive",
        "less conspicuous", "less apparent", "less visible", "less severe",
        "now less", "now smaller",
        "improved aeration",
    ],

    # ----- 4 : stable -----
    4: [
        "unchanged", "stable", "similar",
        "no change", "no significant change",
        "no interval change", "no significant interval change",
        "without significant change", "without interval change",
        "without significant interval change",
        "persistent", "persists", "persisting", "persistence",
        "remain", "remains", "remaining",
        "re-demonstrated", "redemonstrated", "redemonstrate",
        "reexpanded", "re-expanded",
        "again seen", "again demonstrated", "again noted",
        "again identified", "again visualized", "again present",
        "again",
        "still", "still seen", "still present",
        "still demonstrated", "still noted", "still identified",
        "continued", "continues", "continuing",
        "unchanged in appearance", "unchanged compared",
        "grossly stable", "grossly unchanged",
        "relatively stable", "relatively unchanged",
    ],
}

# 用于辅助判断 "is this report describing a comparison at all"
# （仅作诊断字段，不改变 has_prior 的真实值）
TEMPORAL_TRIGGER_WORDS = [
    "compared to", "compared with", "in comparison", "comparison with",
    "interval", "since", "since the prior", "since the previous",
    "prior", "previous", "previously",
    "unchanged", "stable", "similar",
    "new", "newly", "improved", "improvement",
    "worsened", "worsening", "resolved", "resolving",
    "persistent", "persists", "remains", "remaining",
    "increased", "decreased",
    "progression", "progressed", "regression",
    "re-demonstrated", "redemonstrated", "re-expanded", "reexpanded",
    "recurrent", "recurrence",
    "enlarged", "enlargement", "smaller", "larger",
    "again", "still", "continues",
]


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
def _make_pattern(kw: str) -> str:
    """
    Multi-word phrases → raw escape (whitespace match strict).
    Single-word tokens → \b...\b word-boundary match.
    Hyphenated tokens contain no space, so they take the \b...\b branch;
    the internal hyphen is escaped and matched literally.
    """
    if " " in kw:
        return re.escape(kw)
    return rf"\b{re.escape(kw)}\b"


def _has_temporal_signal(text: str) -> bool:
    low = text.lower()
    for kw in TEMPORAL_TRIGGER_WORDS:
        if re.search(_make_pattern(kw), low):
            return True
    return False


def extract_change_label(text: str, has_prior: bool) -> int:
    """
    Returns one of:
        0  = no_prior  (no previous study at all)
        1  = new
        2  = worsened
        3  = improved
        4  = stable    (explicit stable / unchanged / similar / ... 命中)
        5  = none      (has_prior=True 但无任何匹配关键词)

    Both label 0 and label 5 should be ignored by L_tc on the training side.
    """
    if not has_prior:
        return 0

    low = (text or "").lower()
    for cls_id in (1, 2, 3, 4):
        for kw in CHANGE_KEYWORDS[cls_id]:
            if re.search(_make_pattern(kw), low):
                return cls_id

    # has_prior=True 但找不到任何关键词 → none，不参与监督
    return 5


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------
def process_sample(rec: Dict) -> Dict:
    findings   = rec.get("findings", "")   or ""
    impression = rec.get("impression", "") or ""
    prior_study = rec.get("prior_study", None)

    has_prior = (prior_study is not None and bool(prior_study))

    text_for_scan = f"{findings} {impression}"
    text_has_temporal = _has_temporal_signal(text_for_scan)

    change_label = extract_change_label(text_for_scan, has_prior)

    return {
        "change_label": change_label,
        "has_prior":    int(bool(has_prior)),
        # Diagnostic field, not used during training.
        "_text_has_temporal_signal": int(text_has_temporal),
    }


def update_record(rec: Dict) -> Dict:
    rec.update(process_sample(rec))
    return rec


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
LABEL_NAME = {
    0: "no_prior",
    1: "new",
    2: "worsened",
    3: "improved",
    4: "stable",
    5: "none",
}


def collect_stats(records: List[Dict]) -> Dict:
    total = len(records)
    has_prior_cnt = sum(1 for r in records if r.get("has_prior"))
    change_hist = Counter(r["change_label"] for r in records)

    # Sanity checks (理论上应当为空):
    # 1) has_prior=False 但 change_label != 0
    weird_no_prior = [
        r.get("id", "?") for r in records
        if (not r.get("has_prior")) and r.get("change_label", -1) != 0
    ][:5]
    # 2) has_prior=True 但 change_label == 0
    weird_has_prior_zero = [
        r.get("id", "?") for r in records
        if r.get("has_prior") and r.get("change_label", -1) == 0
    ][:5]

    # 比例：multi-class label 中 "none" 占 has_prior 的多少？
    none_among_prior = change_hist.get(5, 0)
    valid_change = sum(change_hist.get(k, 0) for k in (1, 2, 3, 4))

    return {
        "total":                total,
        "has_prior":            has_prior_cnt,
        "change_hist":          dict(change_hist),
        "weird_no_prior":       weird_no_prior,
        "weird_has_prior_zero": weird_has_prior_zero,
        "valid_change":         valid_change,
        "none_among_prior":     none_among_prior,
    }


def print_stats(split_name: str, stats: Dict) -> None:
    t = max(stats["total"], 1)
    print(f"\n── [{split_name}] 总样本 {stats['total']} ──")
    print(f"  has_prior               : {stats['has_prior']:6d}  "
          f"({100 * stats['has_prior'] / t:5.1f}%)")
    print(f"  change_label 分布:")
    for k in sorted(stats["change_hist"].keys()):
        cnt = stats["change_hist"][k]
        note = ""
        if k == 0:
            note = "   ← no_prior, 训练时 ignore"
        elif k == 5:
            note = "   ← none, 训练时 ignore"
        print(f"      {k}={LABEL_NAME.get(k, '?'):10s}: {cnt:6d}  "
              f"({100 * cnt / t:5.1f}%){note}")

    print(f"  受监督样本（label ∈ 1..4）: {stats['valid_change']:6d}  "
          f"({100 * stats['valid_change'] / t:5.1f}%)")
    print(f"  has_prior 内 none 占比   : "
          f"{stats['none_among_prior']:6d}  "
          f"({100 * stats['none_among_prior'] / max(stats['has_prior'], 1):5.1f}%)")

    if stats["weird_no_prior"]:
        print(f"  ⚠ has_prior=False 但 label!=0 的样本（不应存在）: "
              f"{stats['weird_no_prior']}")
    elif stats["weird_has_prior_zero"]:
        print(f"  ⚠ has_prior=True 但 label==0 的样本（不应存在）: "
              f"{stats['weird_has_prior_zero']}")
    else:
        print(f"  ✓ 标签语义一致（no_prior↔label 0；has_prior↔label 1..5）")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Add change_label / has_prior to PriorRG annotation JSON")
    parser.add_argument("--input",  required=True,
                        help="原始 PriorRG 标注 JSON（含 findings / impression / prior_study）")
    parser.add_argument("--output", required=True,
                        help="输出标注 JSON 路径（写入 change_label / has_prior 字段）")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for split_name, records in data.items():
            if not isinstance(records, list):
                continue
            data[split_name] = [update_record(r) for r in records]
            print_stats(split_name, collect_stats(data[split_name]))
    elif isinstance(data, list):
        data = [update_record(r) for r in data]
        print_stats("all", collect_stats(data))
    else:
        raise ValueError(f"Unsupported annotation structure: {type(data)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Wrote {args.output}")


if __name__ == "__main__":
    main()

