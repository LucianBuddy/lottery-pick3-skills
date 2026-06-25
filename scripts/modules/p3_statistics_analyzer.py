#!/usr/bin/env python3
"""
排列3 统计分析器 (P3StatisticsAnalyzer)
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np


class P3StatisticsAnalyzer:
    """排列3历史数据分析器"""

    def __init__(self, draws: List[Tuple[int, int, int]]):
        self.draws = draws
        self.bai_hist = [d[0] for d in draws]
        self.shi_hist = [d[1] for d in draws]
        self.ge_hist = [d[2] for d in draws]

    def sum_stats(self) -> Dict:
        """和值统计"""
        sums = [sum(d) for d in self.draws]
        cnt = Counter(sums)
        return {
            'mean': round(np.mean(sums), 1),
            'std': round(np.std(sums), 1),
            'min': min(sums),
            'max': max(sums),
            'recent_20': [sum(d) for d in self.draws[-20:]],
            'top_5': [s for s, _ in cnt.most_common(5)],
        }

    def span_stats(self) -> Dict:
        """跨度统计"""
        spans = [max(d) - min(d) for d in self.draws]
        cnt = Counter(spans)
        return {
            'mean': round(np.mean(spans), 1),
            'min': min(spans),
            'max': max(spans),
            'recent_20': [max(d) - min(d) for d in self.draws[-20:]],
            'top_5': [s for s, _ in cnt.most_common(5)],
        }

    def parity_stats(self) -> Dict:
        """奇偶分布统计"""
        parity_pairs = []
        for d in self.draws:
            odd = sum(1 for x in d if x % 2 == 1)
            parity_pairs.append(f"{odd}:{3-odd}")
        cnt = Counter(parity_pairs)
        total = len(parity_pairs)
        return {
            'distribution': {k: round(v / total * 100, 1) for k, v in cnt.items()},
            'recent_20': parity_pairs[-20:],
        }

    def big_small_stats(self) -> Dict:
        """大小分布统计 (≥5为大)"""
        bs_pairs = []
        for d in self.draws:
            big = sum(1 for x in d if x >= 5)
            bs_pairs.append(f"{big}:{3-big}")
        cnt = Counter(bs_pairs)
        total = len(bs_pairs)
        return {
            'distribution': {k: round(v / total * 100, 1) for k, v in cnt.items()},
            'recent_20': bs_pairs[-20:],
        }

    def group_type_stats(self) -> Dict:
        """组三/组六/豹子分布"""
        types = []
        for d in self.draws:
            s = len(set(d))
            if s == 1:
                types.append('豹子')
            elif s == 2:
                types.append('组三')
            else:
                types.append('组六')
        cnt = Counter(types)
        total = len(types)
        return {
            'distribution': {k: round(v / total * 100, 1) for k, v in cnt.items()},
            'recent_20': types[-20:],
        }

    def position_frequency(self, pos: str = 'bai', window: int = 30) -> Dict:
        """某位数字频率统计"""
        hist = {'bai': self.bai_hist, 'shi': self.shi_hist, 'ge': self.ge_hist}[pos]
        recent = hist[-window:] if len(hist) >= window else hist
        cnt = Counter(recent)
        total = len(recent)
        return {str(d): round(cnt.get(d, 0) / total * 100, 1) for d in range(10)}

    def miss_stats(self) -> Dict[str, Dict[int, int]]:
        """每位数字遗漏值"""
        result = {}
        for pos_name, hist in [('bai', self.bai_hist), ('shi', self.shi_hist),
                                ('ge', self.ge_hist)]:
            miss = {}
            for d in range(10):
                for i in range(len(hist) - 1, -1, -1):
                    if hist[i] == d:
                        miss[d] = len(hist) - 1 - i
                        break
                else:
                    miss[d] = len(hist)
            result[pos_name] = miss
        return result

    def consecutive_pattern(self) -> Dict:
        """连号模式分析"""
        patterns = []
        for d in self.draws:
            s = sorted(d)
            cons = 0
            for i in range(len(s) - 1):
                if s[i + 1] - s[i] == 1:
                    cons += 1
            patterns.append(cons)
        cnt = Counter(patterns)
        total = len(patterns)
        return {
            'distribution': {k: round(v / total * 100, 1) for k, v in cnt.items()},
            'recent_20': patterns[-20:],
        }

    def route_012_analysis(self) -> Dict:
        """012路形态统计分析"""
        routes = []
        for d in self.draws:
            r = tuple(sum(1 for x in d if x % 3 == i) for i in range(3))
            routes.append(f"{r[0]}:{r[1]}:{r[2]}")
        cnt = Counter(routes)
        total = len(routes)
        return {
            'distribution': {k: round(v / total * 100, 1) for k, v in cnt.items()},
            'recent_20': routes[-20:],
        }
