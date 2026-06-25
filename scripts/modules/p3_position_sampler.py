#!/usr/bin/env python3
"""
排列3 位置采样器 (P3PositionSampler) — V1.0.1
FIX: _build_frequencies 移除布尔锁，改为 window 追踪
      group_z3/z6 采样增强：基于遗漏值+频率混合排序
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import numpy as np
import random


class P3PositionSampler:
    """排列3位置采样器"""

    def __init__(self, draws: List[Tuple[int, int, int]]):
        self.draws = draws
        self.n_periods = len(draws)
        self._freq_cache = {}
        self._cached_window = 0  # 追踪上次构建的 window

        # 每位数字的历史序列
        self.bai_hist = [d[0] for d in draws]
        self.shi_hist = [d[1] for d in draws]
        self.ge_hist = [d[2] for d in draws]

    def _build_frequencies(self, window: int = 30):
        # 【FIX P0】window 变化时重建缓存
        if self._cached_window == window and self._freq_cache:
            return
        self._cached_window = window

        if self.n_periods < window:
            window = max(self.n_periods, 5)

        self._freq_cache = {}

        # 每位近window期频率 + 全历史遗漏
        for pos_name, pos_hist in [('bai', self.bai_hist), ('shi', self.shi_hist),
                                   ('ge', self.ge_hist)]:
            recent = pos_hist[-window:] if len(pos_hist) >= window else pos_hist
            freq = Counter(recent)
            total = len(recent)
            self._freq_cache[f'{pos_name}_freq'] = {k: v / total for k, v in freq.items()}

            # 遗漏值（多少期未出现）
            miss = {}
            for d in range(10):
                for i in range(len(pos_hist) - 1, -1, -1):
                    if pos_hist[i] == d:
                        miss[d] = len(pos_hist) - 1 - i
                        break
                else:
                    miss[d] = len(pos_hist)
            self._freq_cache[f'{pos_name}_miss'] = miss

            # 全历史频率
            full_freq = Counter(pos_hist)
            full_total = len(pos_hist)
            self._freq_cache[f'{pos_name}_full_freq'] = {k: v / full_total for k, v in full_freq.items()}

        # 组选频率（忽略顺序）
        group_draws = [tuple(sorted(d)) for d in self.draws]
        recent_group = group_draws[-window:] if len(group_draws) >= window else group_draws
        group_freq = Counter(recent_group)
        total = len(recent_group)
        self._freq_cache['group_freq'] = {k: v / total for k, v in group_freq.items()}

        # 组选遗漏值（组合级别）
        group_miss = {}
        all_combos = set()
        for d in range(10):
            for e in range(d, 10):
                for f in range(e, 10):
                    all_combos.add((d, e, f))
        for combo in all_combos:
            for i in range(len(group_draws) - 1, -1, -1):
                if group_draws[i] == combo:
                    group_miss[combo] = len(group_draws) - 1 - i
                    break
            else:
                group_miss[combo] = len(group_draws)
        self._freq_cache['group_miss'] = group_miss

    def get_hot_digits(self, pos: str, n: int = 4, window: int = 30) -> List[int]:
        """获取某位热号（频率最高）"""
        self._build_frequencies(window)
        freq = self._freq_cache.get(f'{pos}_freq', {})
        sorted_d = sorted(freq.items(), key=lambda x: -x[1])
        return [d for d, _ in sorted_d[:n]]

    def get_cold_digits(self, pos: str, n: int = 4) -> List[int]:
        """获取某位冷号（遗漏最大）"""
        self._build_frequencies()
        miss = self._freq_cache.get(f'{pos}_miss', {})
        sorted_d = sorted(miss.items(), key=lambda x: -x[1])
        return [d for d, _ in sorted_d[:n]]

    def get_trend_digits(self, pos: str, n: int = 4, window: int = 10) -> List[int]:
        """获取趋势数字（近期上升趋势）"""
        self._build_frequencies()
        pos_hist = {'bai': self.bai_hist, 'shi': self.shi_hist, 'ge': self.ge_hist}[pos]
        recent = pos_hist[-window:] if len(pos_hist) >= window else pos_hist
        freq_recent = Counter(recent)
        total_recent = len(recent)
        miss = self._freq_cache.get(f'{pos}_miss', {})
        full_freq = self._freq_cache.get(f'{pos}_full_freq', {})

        scores = {}
        for d in range(10):
            rf = freq_recent.get(d, 0) / max(total_recent, 1)
            # 趋势 = 近期频率上升幅度 + 冷号反弹
            base_rate = full_freq.get(d, 0.1)
            rise = max(0, rf - base_rate) * 2  # 上升幅度放大
            cold_bounce = min(miss.get(d, 0) / 30, 1) * 0.3
            scores[d] = rf * 0.5 + rise * 0.3 + cold_bounce * 0.2

        sorted_d = sorted(scores.items(), key=lambda x: -x[1])
        return [d for d, _ in sorted_d[:n]]

    def get_balance_digits(self, pos: str, n: int = 4) -> List[int]:
        """均衡采样：热冷混合"""
        self._build_frequencies()
        freq = self._freq_cache.get(f'{pos}_freq', {})
        miss = self._freq_cache.get(f'{pos}_miss', {})
        all_scores = {}
        for d in range(10):
            f = freq.get(d, 0)
            m = miss.get(d, 0)
            all_scores[d] = f * 0.5 + min(m / 20, 1) * 0.5
        sorted_d = sorted(all_scores.items(), key=lambda x: -x[1])
        return [d for d, _ in sorted_d[:n]]

    def sample_position(self, pos: str, strategy: str = 'hot', n: int = 4) -> List[int]:
        """按策略采样某位的数字"""
        if strategy == 'hot':
            return self.get_hot_digits(pos, n)
        elif strategy == 'cold':
            return self.get_cold_digits(pos, n)
        elif strategy == 'trend':
            return self.get_trend_digits(pos, n)
        elif strategy == 'balance':
            return self.get_balance_digits(pos, n)
        else:
            return list(range(10))

    def sample_direct(self, n: int = 5, strategy: str = 'hot',
                      prev: Optional[List[int]] = None) -> List[List[int]]:
        """采样直选号码"""
        bai_pool = self.sample_position('bai', strategy, 6)  # 池扩大到6个
        shi_pool = self.sample_position('shi', strategy, 6)
        ge_pool = self.sample_position('ge', strategy, 6)

        results = []
        for _ in range(n * 4):
            bai = random.choice(bai_pool)
            shi = random.choice(shi_pool)
            ge = random.choice(ge_pool)
            # 除非策略是冷/均衡，组三/豹子比例降低
            if strategy not in ('cold', 'balance'):
                if len({bai, shi, ge}) < 2 and random.random() < 0.6:
                    continue
            results.append([bai, shi, ge])

        random.shuffle(results)
        seen = set()
        unique = []
        for d in results:
            k = tuple(d)
            if k not in seen:
                seen.add(k)
                unique.append(d)
                if len(unique) >= n:
                    break
        return unique[:n]

    def sample_group_z3(self, n: int = 5) -> List[List[int]]:
        """
        采样组三号码 — 【FIX P1】基于频率+遗漏混合排序，取消随机补数
        """
        self._build_frequencies()
        group_freq = self._freq_cache.get('group_freq', {})
        group_miss = self._freq_cache.get('group_miss', {})

        # 混合评分：频率×0.6 + 遗漏归一化×0.4
        z3_scores = {}
        for combo, prob in group_freq.items():
            if len(set(combo)) == 2:
                miss = group_miss.get(combo, 0)
                miss_score = min(miss / 30, 1)
                z3_scores[combo] = prob * 0.6 + miss_score * 0.4

        if not z3_scores:
            # 无历史数据时的fallback
            for _ in range(n):
                d = random.randint(0, 9)
                diff = random.randint(0, 9)
                while diff == d:
                    diff = random.randint(0, 9)
                lst = [d, d, diff]
                random.shuffle(lst)
                z3_scores[tuple(sorted(lst))] = 0.5

        sorted_z3 = sorted(z3_scores.items(), key=lambda x: -x[1])
        results = [list(combo) for combo, _ in sorted_z3[:n]]
        # 随机打乱位置（组三在直选中位置可变）
        for lst in results:
            random.shuffle(lst)
        return results

    def sample_group_z6(self, n: int = 5) -> List[List[int]]:
        """
        采样组六号码 — 【FIX P1】基于频率+遗漏混合排序
        """
        self._build_frequencies()
        group_freq = self._freq_cache.get('group_freq', {})
        group_miss = self._freq_cache.get('group_miss', {})

        z6_scores = {}
        for combo, prob in group_freq.items():
            if len(set(combo)) == 3:
                miss = group_miss.get(combo, 0)
                miss_score = min(miss / 30, 1)
                z6_scores[combo] = prob * 0.6 + miss_score * 0.4

        if not z6_scores:
            for _ in range(n):
                z6_scores[tuple(sorted(random.sample(range(10), 3)))] = 0.5

        sorted_z6 = sorted(z6_scores.items(), key=lambda x: -x[1])
        return [list(combo) for combo, _ in sorted_z6[:n]]

    def get_compound_pools(self, top_n: int = 8) -> Dict[str, List[int]]:
        """
        生成复式用位置数字池 — 【FIX P2】基于热号+冷号混合
        """
        self._build_frequencies()
        pools = {}
        for pos in ['bai', 'shi', 'ge']:
            freq = self._freq_cache.get(f'{pos}_freq', {})
            miss = self._freq_cache.get(f'{pos}_miss', {})
            # 混合评分：频率×0.6 + 遗漏归一化×0.3 + 基础分0.1
            scores = {}
            for d in range(10):
                f = freq.get(d, 0)
                m = min(miss.get(d, 0) / 20, 1)
                scores[d] = f * 0.6 + m * 0.3 + 0.1
            sorted_d = sorted(scores.items(), key=lambda x: -x[1])
            pools[pos] = [d for d, _ in sorted_d[:top_n]]
        return pools

    def get_position_stats(self) -> Dict[str, Dict[int, Dict]]:
        """获取每位统计信息"""
        self._build_frequencies()
        stats = {}
        for pos in ['bai', 'shi', 'ge']:
            freq = self._freq_cache.get(f'{pos}_freq', {})
            miss = self._freq_cache.get(f'{pos}_miss', {})
            stats[pos] = {}
            for d in range(10):
                stats[pos][d] = {
                    'freq': round(freq.get(d, 0) * 100, 1),
                    'miss': miss.get(d, 0),
                    'status': '热' if freq.get(d, 0) > 0.15 else ('冷' if miss.get(d, 0) > 20 else '温'),
                }
        return stats
