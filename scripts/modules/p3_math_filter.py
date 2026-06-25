#!/usr/bin/env python3
"""
排列3 数学过滤器 (P3MathFilter)
"""

from typing import List, Dict, Tuple, Optional, Callable
from collections import Counter
import math


class P3MathFilter:
    """排列3数学过滤器 — 对候选号码应用数学约束过滤"""

    def __init__(self):
        pass

    @staticmethod
    def filter_sum_range(candidates: List[List[int]], min_s: int = 1, max_s: int = 26) -> List[List[int]]:
        """过滤和值范围"""
        return [c for c in candidates if min_s <= sum(c) <= max_s]

    @staticmethod
    def filter_span(candidates: List[List[int]], min_sp: int = 0, max_sp: int = 9) -> List[List[int]]:
        """过滤跨度范围"""
        return [c for c in candidates if min_sp <= max(c) - min(c) <= max_sp]

    @staticmethod
    def filter_parity(candidates: List[List[int]], odd_count: int) -> List[List[int]]:
        """过滤奇偶 (odd_count 为奇数的个数)"""
        return [c for c in candidates if sum(1 for x in c if x % 2 == 1) == odd_count]

    @staticmethod
    def filter_group_type(candidates: List[List[int]], gtype: str) -> List[List[int]]:
        """过滤组类型: 组三/组六/豹子"""
        if gtype == '组三':
            return [c for c in candidates if len(set(c)) == 2]
        elif gtype == '组六':
            return [c for c in candidates if len(set(c)) == 3]
        elif gtype == '豹子':
            return [c for c in candidates if len(set(c)) == 1]
        return candidates

    @staticmethod
    def filter_consecutive(candidates: List[List[int]], allow_cons: bool = True) -> List[List[int]]:
        """过滤连号"""
        def has_consecutive(d):
            s = sorted(d)
            for i in range(len(s) - 1):
                if s[i + 1] - s[i] == 1:
                    return True
            return False
        if allow_cons:
            return candidates
        return [c for c in candidates if not has_consecutive(c)]

    @staticmethod
    def filter_route_012(candidates: List[List[int]], route: Tuple[int, int, int]) -> List[List[int]]:
        """过滤012路形态"""
        r0, r1, r2 = route
        return [c for c in candidates if
                sum(1 for x in c if x % 3 == 0) == r0 and
                sum(1 for x in c if x % 3 == 1) == r1 and
                sum(1 for x in c if x % 3 == 2) == r2]

    @staticmethod
    def filter_big_small(candidates: List[List[int]], big_count: int) -> List[List[int]]:
        """过滤大小 (≥5为大)"""
        return [c for c in candidates if sum(1 for x in c if x >= 5) == big_count]

    @staticmethod
    def smart_score(candidate: List[int]) -> float:
        """
        对单注号码计算"合理性"评分 (0~1)
        基于数学特征
        """
        digits = candidate
        scores = []

        # 和值过极端扣分 (0,27 极不常见)
        s = sum(digits)
        if s <= 1 or s >= 26:
            scores.append(0.1)
        elif s <= 3 or s >= 24:
            scores.append(0.4)
        elif 8 <= s <= 19:
            scores.append(1.0)
        else:
            scores.append(0.7)

        # 三同(豹子)扣分
        if len(set(digits)) == 1:
            scores.append(0.1)
        # 组三扣分 (相对于组六)
        elif len(set(digits)) == 2:
            scores.append(0.7)
        else:
            scores.append(0.9)

        # 跨度极端扣分
        sp = max(digits) - min(digits)
        if sp <= 1 or sp >= 9:
            scores.append(0.3)
        elif sp <= 2:
            scores.append(0.6)
        else:
            scores.append(0.9)

        return sum(scores) / len(scores)
