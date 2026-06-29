#!/usr/bin/env python3
"""
排列3 跨期模式识别器 (P3PatternRecognizer)

从历史数据提取模式特征分布，对候选号码进行模式匹配度评分。

模式特征:
1. 和值 (Sum) — 百+十+个 (0-27)
2. 跨度 (Span) — 最大位-最小位 (0-9) 
3. 奇偶比 (Parity) — 奇偶分布 (3:0/2:1/1:2/0:3)
4. 大小比 (Size) — ≥5为大 (3:0/2:1/1:2/0:3)
5. 重复模式 (Repeat) — 与上期对应位置相同的位数 (0-3)
6. 012路形态 — 各数字 mod 3 的分布
7. 组三/组六形态 — 重复数字个数
8. 位置冷热 — 每位数字遗漏值
9. 质合比 — 质数(2,3,5,7)个数
"""

from typing import List, Dict, Tuple, Any, Optional, Set
from collections import Counter, defaultdict
import numpy as np
import math
import random
import copy


class P3PatternRecognizer:
    """
    排列3模式识别器
    从历史开奖数据中提取模式分布，对候选号码进行模式匹配度评分。
    """

    PRIME_DIGITS = {2, 3, 5, 7}

    def __init__(self, draws: List[Tuple[int, int, int]]):
        """
        Args:
            draws: 历史数据 [(百,十,个), ...] 按时序排列（最新在最后）
        """
        self.draws = draws
        self.n_periods = len(draws)
        self._pattern_distributions: Dict[str, Dict] = {}
        self._is_built = False

    # ================================================================
    # 模式特征提取
    # ================================================================

    def extract_sum(self, digits: List[int]) -> int:
        return sum(digits)

    def extract_span(self, digits: List[int]) -> int:
        return max(digits) - min(digits)

    def extract_parity_count(self, digits: List[int]) -> Tuple[int, int]:
        """(odd_count, even_count)"""
        odd = sum(1 for d in digits if d % 2 == 1)
        return odd, 3 - odd

    def extract_big_small(self, digits: List[int]) -> Tuple[int, int]:
        """(big_count, small_count), 大≥5"""
        big = sum(1 for d in digits if d >= 5)
        return big, 3 - big

    def extract_sum_tail(self, digits):
        """和值尾数（0-9）"""
        return sum(digits) % 10

    def extract_span_parity(self, digits):
        """跨度奇偶组合: 奇数跨度或偶数跨度"""
        span = max(digits) - min(digits)
        return 'odd' if span % 2 == 1 else 'even'

    def extract_repeat_count(self, current: List[int], prev: List[int]) -> int:
        """与上期对应位置相同的位数"""
        return sum(1 for i in range(3) if current[i] == prev[i])

    def extract_012_route(self, digits: List[int]) -> Tuple[int, int, int]:
        """各数字的 012 路统计"""
        r0 = sum(1 for d in digits if d % 3 == 0)
        r1 = sum(1 for d in digits if d % 3 == 1)
        r2 = sum(1 for d in digits if d % 3 == 2)
        return r0, r1, r2

    def extract_group_type(self, digits: List[int]) -> str:
        """组三(2个相同) / 组六(3个不同) / 豹子(3个相同)"""
        s = len(set(digits))
        if s == 1:
            return '豹子'
        elif s == 2:
            return '组三'
        return '组六'

    def extract_prime_count(self, digits: List[int]) -> int:
        return sum(1 for d in digits if d in {2, 3, 5, 7})

    def extract_all_features(self, digits: List[int], prev: Optional[List[int]] = None) -> Dict[str, Any]:
        features = {
            'sum': self.extract_sum(digits),
            'span': self.extract_span(digits),
            'parity': self.extract_parity_count(digits),
            'big_small': self.extract_big_small(digits),
            'group_type': self.extract_group_type(digits),
            'prime_count': self.extract_prime_count(digits),
            'sum_tail': self.extract_sum_tail(digits),
            'span_parity': self.extract_span_parity(digits),
            'route_012': self.extract_012_route(digits),
        }
        if prev is not None:
            features['repeat'] = self.extract_repeat_count(digits, prev)
        return features

    # ================================================================
    # 模式分布构建
    # ================================================================

    def _build_distributions(self):
        if self._is_built:
            return
        if self.n_periods < 20:
            print(f"[P3-Pattern] 警告: 历史数据不足 ({self.n_periods}期)，模式分布可能不准确")
            self._is_built = True
            return

        # 存原始计数器（用于增量更新）
        self._raw_counts = {}
        self._raw_total = 0

        for i in range(1, self.n_periods):
            prev = self.draws[i - 1]
            curr = self.draws[i]
            self._add_feature_to_raw(
                self.extract_all_features(list(curr), list(prev)))

        if self._raw_total == 0:
            self._is_built = True
            return

        self._finalize_distributions()
        self._is_built = True

    def _add_feature_to_raw(self, features: Dict):
        """向原始计数器添加一个特征"""
        if not hasattr(self, '_raw_counts'):
            self._raw_counts = {}
            self._raw_total = 0

        for key in ['sum', 'span', 'repeat']:
            if key in features:
                self._raw_counts.setdefault(key, Counter())[features[key]] += 1

        for key, val in [('parity', f"{features['parity'][0]}:{features['parity'][1]}"),
                         ('big_small', f"{features['big_small'][0]}:{features['big_small'][1]}"),
                         ('group_type', features['group_type']),
                         ('prime_count', features['prime_count']),
                         ('sum_tail', features['sum_tail']),
                         ('span_parity', features['span_parity']),
                         ('route_012', f"{features['route_012'][0]}:{features['route_012'][1]}:{features['route_012'][2]}")]:
            if key in features or key in ('parity', 'big_small', 'group_type', 'prime_count', 'route_012'):
                self._raw_counts.setdefault(key, Counter())[val] += 1

        self._raw_total += 1

    def _finalize_distributions(self):
        """从原始计数器生成概率分布"""
        if self._raw_total == 0:
            return
        total = self._raw_total
        for key, counter in self._raw_counts.items():
            self._pattern_distributions[key] = {
                k: v / total for k, v in counter.items()
            }

    def add_period(self, period_draw: List[int], prev_draw: List[int]):
        """
        【增量更新】添加一个新期到模式分布
        用于回测场景复用已构建的分布，避免全量重建
        """
        features = self.extract_all_features(period_draw, prev_draw)
        self._add_feature_to_raw(features)
        # 重新归一化
        self._finalize_distributions()

    def get_distribution(self, feature_name: str) -> Dict:
        self._build_distributions()
        return self._pattern_distributions.get(feature_name, {})

    def score_pattern(self, digits: List[int], prev: Optional[List[int]] = None) -> float:
        """
        计算候选号码的模式匹配度 (0.0~1.0)
        """
        self._build_distributions()
        if not self._pattern_distributions:
            return 0.5

        features = self.extract_all_features(digits, prev)

        def get_prob(dist: Dict, key) -> float:
            return dist.get(key, 0.0)

        scores = []

        # 和值匹配度
        sum_dist = self._pattern_distributions.get('sum', {})
        if sum_dist:
            # 取和值附近 ±2 的累计概率
            s = features['sum']
            near_sum = sum(v for k, v in sum_dist.items() if abs(k - s) <= 2)
            scores.append(min(near_sum * 3, 1.0))

        # 跨度匹配度
        span_dist = self._pattern_distributions.get('span', {})
        if span_dist:
            scores.append(get_prob(span_dist, features['span']))

        # 奇偶比匹配度
        parity_dist = self._pattern_distributions.get('parity', {})
        if parity_dist:
            o, e = features['parity']
            scores.append(get_prob(parity_dist, f"{o}:{e}"))

        # 大小比匹配度
        bs_dist = self._pattern_distributions.get('big_small', {})
        if bs_dist:
            b, s = features['big_small']
            scores.append(get_prob(bs_dist, f"{b}:{s}"))

        # 组类型匹配度
        group_dist = self._pattern_distributions.get('group_type', {})
        if group_dist:
            scores.append(get_prob(group_dist, features['group_type']))

        # 质数匹配度
        prime_dist = self._pattern_distributions.get('prime_count', {})
        if prime_dist:
            scores.append(get_prob(prime_dist, features['prime_count']))

        # 重复位数匹配度
        if prev is not None and 'repeat' in features:
            repeat_dist = self._pattern_distributions.get('repeat', {})
            if repeat_dist:
                scores.append(get_prob(repeat_dist, features['repeat']))

        if not scores:
            return 0.5

        avg_score = np.mean(scores)
        # V1.4.0: 降低放大系数避免饱和, *3 保留区分度
        return min(avg_score * 3, 1.0)

    def score_pattern_boost(self, digits: List[int], prev: Optional[List[int]] = None,
                            base_score: float = 0.52) -> float:
        """
        模式评分增强
        FIX: 放大系数从0.20改为0.35，依赖caller传base
        """
        pattern_score = self.score_pattern(digits, prev)
        return base_score + pattern_score * 0.35

    def get_pattern_diversity_pool(self, n: int = 50, prev: Optional[List[int]] = None) -> List[List[int]]:
        """
        从高频模式反推号码集合，生成多样性候选池
        """
        self._build_distributions()

        pool = []

        # 从高频和值区间采样
        sum_dist = self._pattern_distributions.get('sum', {})
        if sum_dist:
            top_sums = sorted(sum_dist.items(), key=lambda x: -x[1])[:5]
            for s_val, _ in top_sums:
                for _ in range(5):
                    bai = random.randint(0, 9)
                    shi = random.randint(0, 9)
                    ge = s_val - bai - shi
                    if 0 <= ge <= 9:
                        pool.append([bai, shi, ge])

        # 从高频跨度区间采样
        span_dist = self._pattern_distributions.get('span', {})
        if span_dist:
            top_spans = sorted(span_dist.items(), key=lambda x: -x[1])[:3]
            for sp_val, _ in top_spans:
                for _ in range(4):
                    d = sorted([random.randint(0, 9 - sp_val),
                                random.randint(0, 9 - sp_val)])
                    if d[1] - d[0] == sp_val:
                        pool.append([d[0], d[1], random.randint(0, 9)])

        # 从高频组类型采样
        group_dist = self._pattern_distributions.get('group_type', {})
        if group_dist:
            top_group = max(group_dist.items(), key=lambda x: x[1])[0]
            for _ in range(6):
                if top_group == '组三':
                    d = random.randint(0, 9)
                    diff = random.randint(0, 9)
                    while diff == d:
                        diff = random.randint(0, 9)
                    pool.append(random.sample([d, d, diff], 3))
                elif top_group == '豹子':
                    d = random.randint(0, 9)
                    pool.append([d, d, d])
                else:
                    pool.append(random.sample(range(10), 3))

        random.shuffle(pool)
        # 去重
        seen = set()
        unique = []
        for d in pool:
            k = tuple(d)
            if k not in seen:
                seen.add(k)
                unique.append(d)

        return unique[:n]
