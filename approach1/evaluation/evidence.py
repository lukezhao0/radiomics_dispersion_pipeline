"""Evidence quote attribution and lexical feature analysis."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..text_utils import tokenize_quote


def extract_ngram_features(quotes: List[str], max_n: int = 2) -> List[str]:
    feats = set()
    for q in quotes:
        toks = tokenize_quote(q)
        for t in toks:
            feats.add(t)
        if max_n >= 2:
            for i in range(len(toks) - 1):
                feats.add(f"{toks[i]} {toks[i + 1]}")
    return sorted(feats)


def build_evidence_feature_table(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        return pd.DataFrame()
    work = df[df[label_col].isin([0, 1])].copy()
    if len(work) == 0:
        return pd.DataFrame()

    case_features: List[Tuple[int, set[str]]] = []
    for _, row in work.iterrows():
        feats = set(extract_ngram_features(row.get("key_evidence_list", [])))
        label = int(row[label_col])
        case_features.append((label, feats))

    pos_cases = [fs for y, fs in case_features if y == 1]
    neg_cases = [fs for y, fs in case_features if y == 0]
    n_pos = len(pos_cases)
    n_neg = len(neg_cases)

    pos_counter: Counter[str] = Counter()
    neg_counter: Counter[str] = Counter()
    all_feats: set[str] = set()

    for fs in pos_cases:
        for f in fs:
            pos_counter[f] += 1
            all_feats.add(f)
    for fs in neg_cases:
        for f in fs:
            neg_counter[f] += 1
            all_feats.add(f)

    alpha = 0.5
    rows = []
    for feat in all_feats:
        a = pos_counter[feat]
        b = neg_counter[feat]
        support = a + b
        odds_ratio = ((a + alpha) / (n_pos - a + alpha)) / ((b + alpha) / (n_neg - b + alpha)) if (n_pos > 0 and n_neg > 0) else np.nan
        rows.append({
            "feature": feat,
            "pos_count": a,
            "neg_count": b,
            "support": support,
            "odds_ratio_pos_vs_neg": odds_ratio,
        })
    tbl = pd.DataFrame(rows)
    if len(tbl):
        tbl = tbl.sort_values(["support", "odds_ratio_pos_vs_neg"], ascending=[False, False]).reset_index(drop=True)
    return tbl


def evidence_attribution_report(df: pd.DataFrame) -> Tuple[str, Dict[str, pd.DataFrame]]:
    outputs: Dict[str, pd.DataFrame] = {}
    lines: List[str] = []
    for label_col, title in [
        ("dispersion_high_low_pred", "Predicted dispersion high (1) vs low (0)"),
        ("relapse_pred", "Predicted relapse (1) vs non-relapse (0)"),
    ]:
        tbl = build_evidence_feature_table(df, label_col)
        outputs[label_col] = tbl
        lines.append(f"Evidence attribution analysis: {title}")
        if len(tbl) == 0:
            lines.append("  No valid rows / features.")
            lines.append("")
            continue
        tbl2 = tbl[tbl["support"] >= 3].copy()
        if len(tbl2) == 0:
            lines.append("  No features with support >= 3 cases.")
            lines.append("")
            continue
        top_pos = tbl2.sort_values(["odds_ratio_pos_vs_neg", "support"], ascending=[False, False]).head(10)
        top_neg = tbl2.sort_values(["odds_ratio_pos_vs_neg", "support"], ascending=[True, False]).head(10)
        lines.append("  Top features associated with class=1:")
        for _, r in top_pos.iterrows():
            lines.append(f"    {r['feature']}: pos_count={int(r['pos_count'])}, neg_count={int(r['neg_count'])}, support={int(r['support'])}, OR={r['odds_ratio_pos_vs_neg']:.3f}")
        lines.append("  Top features associated with class=0:")
        for _, r in top_neg.iterrows():
            lines.append(f"    {r['feature']}: pos_count={int(r['pos_count'])}, neg_count={int(r['neg_count'])}, support={int(r['support'])}, OR={r['odds_ratio_pos_vs_neg']:.3f}")
        lines.append("")
    return "\n".join(lines), outputs
