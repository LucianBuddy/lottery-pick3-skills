"""
排列3 泊松过程建模模块
对每位数字的"出现间隔"建模拟合，输出反常度分数

核心信号:
  数字 d 已连续 miss 期未出现 → 泊松反常度分数
  分数 = 1 - (1-p)^(miss-1)  归一化到 0~1

二阶马尔可夫链:
  P(d_t | d_{t-1}, d_{t-2})  拉普拉斯平滑
"""

import math
from collections import Counter
from typing import Dict, List, Tuple, Optional


class P3PoissonModel:
    """泊松过程建模器"""

    def __init__(self, draws: List[Tuple[int, int, int]]):
        """
        Args:
            draws: [(百,十,个), ...] 按时序排列（最新在最后）
        """
        self._bai_seq = [d[0] for d in draws]
        self._shi_seq = [d[1] for d in draws]
        self._ge_seq = [d[2] for d in draws]

        # 每位数字的全历史概率
        self._p_bai = self._estimate_p(self._bai_seq)
        self._p_shi = self._estimate_p(self._shi_seq)
        self._p_ge = self._estimate_p(self._ge_seq)

        # 每位数字的遗漏值
        self._miss_bai = self._compute_miss(self._bai_seq)
        self._miss_shi = self._compute_miss(self._shi_seq)
        self._miss_ge = self._compute_miss(self._ge_seq)

        # 一阶转移矩阵 10×10
        self._trans1_bai = self._build_trans_1st(self._bai_seq)
        self._trans1_shi = self._build_trans_1st(self._shi_seq)
        self._trans1_ge = self._build_trans_1st(self._ge_seq)

        # 二阶转移矩阵 100×10 (拉普拉斯平滑)
        self._trans2_bai = self._build_trans_2nd(self._bai_seq)
        self._trans2_shi = self._build_trans_2nd(self._shi_seq)
        self._trans2_ge = self._build_trans_2nd(self._ge_seq)

        # 三阶转移矩阵 1000×10 (拉普拉斯平滑)
        self._trans3_bai = self._build_trans_3rd(self._bai_seq)
        self._trans3_shi = self._build_trans_3rd(self._shi_seq)
        self._trans3_ge = self._build_trans_3rd(self._ge_seq)

    # ================================================================
    # 概率估计
    # ================================================================

    @staticmethod
    def _estimate_p(seq: List[int]) -> Dict[int, float]:
        total = max(len(seq), 1)
        cnt = Counter(seq)
        return {d: cnt.get(d, 0) / total for d in range(10)}

    @staticmethod
    def _compute_miss(seq: List[int]) -> Dict[int, int]:
        miss = {}
        for i in range(len(seq) - 1, -1, -1):
            d = seq[i]
            if d not in miss:
                miss[d] = len(seq) - 1 - i
        for d in range(10):
            if d not in miss:
                miss[d] = len(seq)
        return miss

    # ================================================================
    # 转移矩阵
    # ================================================================

    @staticmethod
    def _build_trans_1st(seq: List[int]) -> List[List[float]]:
        """一阶: 10×10, P(next | prev)"""
        cnt = [[0] * 10 for _ in range(10)]
        for i in range(len(seq) - 1):
            cnt[seq[i]][seq[i + 1]] += 1
        trans = [[0.0] * 10 for _ in range(10)]
        for prev in range(10):
            total = sum(cnt[prev])
            if total > 0:
                for curr in range(10):
                    trans[prev][curr] = cnt[prev][curr] / total
        return trans

    @staticmethod
    def _bayesian_alpha(total: int) -> float:
        """贝叶斯收缩: 样本少→强先验, 样本多→弱先验"""
        return 10.0 / (1.0 + total / 20.0)

    @staticmethod
    def _build_trans_2nd(seq: List[int]) -> List[List[float]]:
        """
        二阶: 100×10, P(next | prev1, prev2)
        贝叶斯收缩: (cnt + α) / (total + α*10)
        """
        cnt = [[0] * 10 for _ in range(100)]
        for i in range(len(seq) - 2):
            p1, p2, curr = seq[i], seq[i + 1], seq[i + 2]
            cnt[p1 * 10 + p2][curr] += 1
        trans = [[0.0] * 10 for _ in range(100)]
        for idx in range(100):
            total = sum(cnt[idx])
            alpha = P3PoissonModel._bayesian_alpha(total)
            for curr in range(10):
                trans[idx][curr] = (cnt[idx][curr] + alpha) / (total + alpha * 10)
        return trans

    @staticmethod
    def _build_trans_3rd(seq: List[int]) -> List[List[float]]:
        """
        三阶: 1000×10, P(next | prev1, prev2, prev3)
        贝叶斯收缩: (cnt + α) / (total + α*10)
        """
        cnt = [[0] * 10 for _ in range(1000)]
        for i in range(len(seq) - 3):
            p1, p2, p3, curr = seq[i], seq[i + 1], seq[i + 2], seq[i + 3]
            cnt[p1 * 100 + p2 * 10 + p3][curr] += 1
        trans = [[0.0] * 10 for _ in range(1000)]
        for idx in range(1000):
            total = sum(cnt[idx])
            alpha = P3PoissonModel._bayesian_alpha(total)
            for curr in range(10):
                trans[idx][curr] = (cnt[idx][curr] + alpha) / (total + alpha * 10)
        return trans

    # ================================================================
    # 泊松反常度
    # ================================================================

    @staticmethod
    def poisson_anomaly(miss: int, p: float) -> float:
        """
        泊松反常度分数 0~1
        = 1 - P(间隔 >= miss) = 1 - (1-p)^(miss-1)
        取 -log 后归一化到 0~1

        含义: 数字 d 已经 miss 期没出现，这有多反常？
        分数越高 → 越反常 → 越"该"出现了
        """
        if miss <= 0 or p <= 0:
            return 0.0
        # P(间隔 >= miss) = (1-p)^(miss-1)
        survival = (1.0 - p) ** (miss - 1)
        # -log(0.5) ≈ 0.69, -log(0.05) ≈ 3.0, -log(0.0067)=5.0
        raw = -math.log(max(survival, 1e-15))
        return min(raw / 6.0, 1.0)

    def get_poisson_score(self, pos: str, d: int) -> float:
        """获取数字 d 在位置 pos 的泊松反常度"""
        miss = {'bai': self._miss_bai, 'shi': self._miss_shi,
                'ge': self._miss_ge}[pos].get(d, 100)
        p = {'bai': self._p_bai, 'shi': self._p_shi,
             'ge': self._p_ge}[pos].get(d, 0.1)
        return self.poisson_anomaly(miss, p)

    def get_trans_score(self, pos: str, d: int, prev_draw: List[int],
                        alpha: float = 0.35) -> float:
        """
        混合一阶+二阶+三阶转移概率
        alpha: 二阶+三阶权重（各占一半）
        一阶权重 = 1 - alpha, 二阶 = alpha/2, 三阶 = alpha/2
        pos='bai'时只有一阶可用
        """
        pos_idx = {'bai': 0, 'shi': 1, 'ge': 2}[pos]
        prev_val = prev_draw[pos_idx]

        trans1 = {'bai': self._trans1_bai, 'shi': self._trans1_shi,
                  'ge': self._trans1_ge}[pos]
        p1 = trans1[prev_val][d]

        if pos_idx == 0:
            return p1  # bai 只有一阶

        trans2 = {'bai': self._trans2_bai, 'shi': self._trans2_shi,
                  'ge': self._trans2_ge}[pos]
        prev2_val = prev_draw[pos_idx - 1]
        idx2 = prev2_val * 10 + prev_val
        p2 = trans2[idx2][d]

        # 三阶转移
        if pos_idx >= 2:  # ge 有三阶: prev3=prev_draw[0], prev2=prev_draw[1], prev1=prev_draw[2]
            trans3 = {'bai': self._trans3_bai, 'shi': self._trans3_shi,
                      'ge': self._trans3_ge}[pos]
            prev3_val = prev_draw[pos_idx - 2]
            idx3 = prev3_val * 100 + prev2_val * 10 + prev_val
            p3 = trans3[idx3][d]
            return (1 - alpha) * p1 + (alpha * 0.5) * p2 + (alpha * 0.5) * p3
        else:  # shi 只有一阶+二阶
            return (1 - alpha) * p1 + alpha * p2
