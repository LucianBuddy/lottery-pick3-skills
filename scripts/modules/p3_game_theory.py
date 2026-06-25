#!/usr/bin/env python3
"""
排列3 博弈论分析器 (P3GameTheoryAnalyzer)
基于博弈论期望值排序的选号分析
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
import math


class P3GameTheoryAnalyzer:
    """排列3博弈论分析器"""

    def __init__(self, draws: List[Tuple[int, int, int]]):
        self.draws = draws
        self.n_periods = len(draws)
        # 每位数字出现频率
        self._pos_freq = {
            'bai': Counter(d[0] for d in draws),
            'shi': Counter(d[1] for d in draws),
            'ge': Counter(d[2] for d in draws),
        }
        # 组选组合频率
        self._group_freq = Counter(tuple(sorted(d)) for d in draws)

    def expected_value(self, digits: List[int]) -> float:
        """
        计算一注号码的博弈论期望值
        基于历史频率估算回报概率
        """
        # 直选概率 ≈ (百位频率) × (十位频率) × (个位频率)
        bai_prob = self._pos_freq['bai'].get(digits[0], 0) / max(self.n_periods, 1)
        shi_prob = self._pos_freq['shi'].get(digits[1], 0) / max(self.n_periods, 1)
        ge_prob = self._pos_freq['ge'].get(digits[2], 0) / max(self.n_periods, 1)

        # 组选概率
        group_key = tuple(sorted(digits))
        group_prob = self._group_freq.get(group_key, 0) / max(self.n_periods, 1)

        # 综合期望 = 直选概率 + 组选概率（组三可额外覆盖3注直选，组六6注）
        s = len(set(digits))
        if s == 1:  # 豹子
            return bai_prob * shi_prob * ge_prob * 1000
        elif s == 2:  # 组三，对应3注直选
            return (bai_prob * shi_prob * ge_prob * 1000 + group_prob * 3)
        else:  # 组六，对应6注直选
            return (bai_prob * shi_prob * ge_prob * 1000 + group_prob * 6)

    def top_direct(self, n: int = 10) -> List[Tuple[List[int], float]]:
        """基于期望值排序，获取Top N直选号码"""
        candidates = []
        for bai in range(10):
            for shi in range(10):
                for ge in range(10):
                    ev = self.expected_value([bai, shi, ge])
                    candidates.append(([bai, shi, ge], ev))
        candidates.sort(key=lambda x: -x[1])
        return candidates[:n]

    def top_group(self, n: int = 10) -> List[Tuple[List[int], float]]:
        """基于期望值排序，获取Top N组选号码"""
        seen = set()
        candidates = []
        for bai in range(10):
            for shi in range(10):
                for ge in range(10):
                    key = tuple(sorted([bai, shi, ge]))
                    if key in seen:
                        continue
                    seen.add(key)
                    ev = self.expected_value([bai, shi, ge])
                    candidates.append((list(key), ev))
        candidates.sort(key=lambda x: -x[1])
        return candidates[:n]

    def nash_equilibrium_prob(self, digits: List[int]) -> float:
        """
        纳什均衡概率计算
        在博弈论框架下，考虑其他人的选号策略，选取"被忽视"的号码组合
        """
        # 热号区：竞猜者更倾向于热号 -> 冷号有隐含价值
        avg_freq = sum(
            self._pos_freq['bai'].get(digits[0], 0) / max(self.n_periods, 1),
            self._pos_freq['shi'].get(digits[1], 0) / max(self.n_periods, 1),
            self._pos_freq['ge'].get(digits[2], 0) / max(self.n_periods, 1),
        ) / 3

        # 均衡条件下，中等频率的号码被低估
        # 极热号竞争大 -> 期望收益低；极冷号风险大
        # 均衡点在 0.08~0.12 频率区间的号码
        if 0.08 <= avg_freq <= 0.12:
            return avg_freq * 1.5  # 价值被低估
        elif avg_freq < 0.05:
            return avg_freq * 0.8  # 太冷，风险大
        elif avg_freq > 0.15:
            return avg_freq * 0.7  # 太热，竞争大
        return avg_freq
