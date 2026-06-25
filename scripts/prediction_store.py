#!/usr/bin/env python3
"""
排列3 (Pick3) 预测结果存储模块
自动保存预测结果至知识库，保留最近十期。
"""

MAX_PERIODS = 10

import json
import os
from typing import Optional, Dict, Any, List

STORE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..',
    'memory', 'pick3_predictions.json'
)


def _ensure_store():
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    if not os.path.exists(STORE_PATH):
        with open(STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump({"predictions": []}, f, ensure_ascii=False, indent=2)


def load_all() -> List[Dict[str, Any]]:
    _ensure_store()
    try:
        with open(STORE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get("predictions", [])
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_all(predictions: List[Dict[str, Any]]):
    _ensure_store()
    with open(STORE_PATH, 'w', encoding='utf-8') as f:
        json.dump({"predictions": predictions}, f, ensure_ascii=False, indent=2)


def store_prediction(
    period: str,
    zx_bets: List[Dict[str, Any]],
    z3_bets: Optional[List[Dict[str, Any]]] = None,
    z6_bets: Optional[List[Dict[str, Any]]] = None,
    compound_bets: Optional[Dict[str, Any]] = None,
):
    """
    存储一期预测结果。

    Args:
        period: 期号
        zx_bets: 直选方案 [{'digits': [百,十,个], 'final_score': ..., 'hit_probability': ...}, ...]
        z3_bets: 组三方案（2个相同数字）
        z6_bets: 组六方案（3个不同数字）
        compound_bets: 复式方案
    """
    predictions = load_all()
    predictions = [p for p in predictions if p.get("period") != period]

    entry = {
        "period": period,
        "zx_bets": [
            {
                "digits": bet.get("digits", []),
                "final_score": round(bet.get("final_score", 0), 4),
                "hit_probability": round(bet.get("hit_probability", 0), 2),
            }
            for bet in zx_bets
        ],
    }
    if z3_bets:
        entry["z3_bets"] = [
            {
                "digits": bet.get("digits", []),
                "final_score": round(bet.get("final_score", 0), 4),
            }
            for bet in z3_bets
        ]
    if z6_bets:
        entry["z6_bets"] = [
            {
                "digits": sorted(bet.get("digits", [])),
                "final_score": round(bet.get("final_score", 0), 4),
            }
            for bet in z6_bets
        ]
    if compound_bets:
        entry["compound_bets"] = compound_bets

    predictions.append(entry)
    predictions.sort(key=lambda x: x.get("period", "0"))
    predictions = predictions[-MAX_PERIODS:]
    save_all(predictions)

    kept = [p["period"] for p in predictions]
    print(f"[P3-Store] ✅ 已保存 {period} 期预测 | "
          f"共 {len(predictions)} 条记录 (留存: {kept})")


def load_prediction(period: str) -> Optional[Dict[str, Any]]:
    predictions = load_all()
    for p in predictions:
        if p.get("period") == period:
            return p
    return None


def list_saved_periods() -> List[str]:
    predictions = load_all()
    periods = [p.get("period", "?") for p in predictions]
    periods.sort(reverse=True)
    return periods


def compare_with_actual(
    period: str,
    actual_digits: List[int],
) -> Optional[Dict[str, Any]]:
    """
    对比预测与实际开奖。
    actual_digits: [百, 十, 个]
    """
    pred = load_prediction(period)
    if not pred:
        return None

    actual = sorted(actual_digits)
    actual_tuple = tuple(actual_digits)

    # 直选命中（顺序+数字全匹配）
    zx_hits = []
    best_zx = None
    for bet in pred.get('zx_bets', []):
        digits = bet.get('digits', [])
        exact = (digits == actual_digits)
        z3_match = len(set(digits)) == 2 and sorted(digits) == actual
        z6_match = len(set(digits)) == 3 and sorted(digits) == actual
        hits = []
        if exact:
            hits.append('直选')
        if z3_match:
            hits.append('组三')
        if z6_match:
            hits.append('组六')
        zx_hits.append({
            'digits': digits,
            'match': hits,
            'score': bet.get('final_score', 0),
        })
        if hits:
            best_zx = {'digits': digits, 'match': hits}

    # 组选命中
    z3_hits = []
    for bet in pred.get('z3_bets', []):
        d = sorted(bet.get('digits', []))
        if d == actual and len(set(d)) == 2:
            z3_hits.append({'digits': bet['digits'], 'score': bet.get('final_score', 0)})
    z6_hits = []
    for bet in pred.get('z6_bets', []):
        d = sorted(bet.get('digits', []))
        if d == actual and len(set(d)) == 3:
            z6_hits.append({'digits': bet['digits'], 'score': bet.get('final_score', 0)})

    return {
        'period': period,
        'actual': {'digits': actual_digits, '组选': sorted(actual_digits)},
        'zx_hits': zx_hits,
        'best_zx': best_zx,
        'z3_hits': z3_hits,
        'z6_hits': z6_hits,
        'has_zx_match': any(h['match'] for h in zx_hits),
        'has_z3_match': len(z3_hits) > 0,
        'has_z6_match': len(z6_hits) > 0,
    }
