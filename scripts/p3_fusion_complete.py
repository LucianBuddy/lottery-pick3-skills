#!/usr/bin/env python3
"""
排列3 (Pick3) 多策略融合预测系统 V2.9.1

全量枚举(1000/90/120) + 置换检验MI权重校准 + 等权重7层
+ 位置级转移(替代1000×1000) + Z3/Z6裁剪 + IRL在线学习

V2.9.1 优化:
  [1] 冷号衰减 — 长遗漏(>50期)数字L1分数封顶
  [2] 位置多样性约束 — 同一位置同一数字最多2次
  [4] 转移概率Borda权重提升(0.2→0.35)
  [5] 复式每位候选数≥3(自动展宽)

评分体系:
  Layer1: 位置三阶频率乘积 (热/冷/趋势融合)
  Layer2: 模式匹配 (和值/跨度/隔期重号/跨期模式)
  Layer3: 多样性补偿 (奇偶/012路/数学合理性)
  Layer4: 博弈论期望/纳什均衡
"""

import sys
import os
import os.path as _path
import json
import random
import warnings
import time as _time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict

from p3_data_updater import check_and_update
from version import VERSION, RELEASE_DATE


def data_dir() -> str:
    return _path.join(_path.dirname(_path.abspath(__file__)), '..',
                      'assets', 'data', '排列3历史数据.xlsx')


def load_data(data_path: str) -> List[Tuple[int, int, int]]:
    if not os.path.exists(data_path):
        print(f"[P3-Fusion] 数据文件不存在: {data_path}")
        from p3_data_updater import check_and_update
        result = check_and_update()
        if not result.get('updated', False):
            raise FileNotFoundError(f"无法获取数据: {data_path}")

    df = pd.read_excel(data_path, engine='openpyxl')
    df = df.sort_values('期号')
    draws = []
    for _, row in df.iterrows():
        draws.append((int(row['百位']), int(row['十位']), int(row['个位'])))
    print(f"[P3-Fusion] 历史数据加载完成 | 共{len(draws)}期 "
          f"({df['期号'].iloc[0]}~{df['期号'].iloc[-1]})")
    return draws


from modules.p3_pattern_recognizer import P3PatternRecognizer
from modules.p3_statistics_analyzer import P3StatisticsAnalyzer
from modules.p3_poisson_model import P3PoissonModel
from modules.p3_math_filter import P3MathFilter
from modules.p3_game_theory import P3GameTheoryAnalyzer
from p3_constraint_engine import P3ConstraintEngine


class Pick3FusionComplete:
    """排列3多策略融合完全体 V2.9.1 — 全量枚举+3层评分
    + 冷号衰减 + 位置多样性 + 转移权重提升 + 复式保底3候选"""

    def __init__(self, data_path: Optional[str] = None, auto_update: bool = True):
        if data_path is None:
            data_path = data_dir()
        self.data_path = data_path

        if auto_update:
            try:
                check_and_update()
            except Exception as e:
                print(f"[P3-Fusion] 自动更新失败: {e}")

        self.draws = load_data(data_path)
        if not self.draws:
            raise ValueError(f"数据加载失败: {data_path}")

        self._init_modules(self.draws)
        self._df_periods = None  # 回测期号缓存
        self.last_period = self._get_last_period()
        self.prev_draw = list(self.draws[-1]) if self.draws else [0, 0, 0]

        # 预计算位置统计 + 泊松模型 + 二阶马尔可夫 + 博弈论
        self._build_position_stats(self.draws)
        self._poisson = P3PoissonModel(self.draws)
        self._game_theory = P3GameTheoryAnalyzer(self.draws)
        # 预计算数字频率和位置对（用于后续所有评分）
        from collections import Counter as _Cnt
        _all_digits = []
        for d in self.draws:
            _all_digits.extend(d)
        _cnt = _Cnt(_all_digits)
        _total = sum(_cnt.values())
        self._digit_overall_freq = {d: _cnt.get(d, 0) / max(_total, 1) for d in range(10)}
        _total_draws = len(self.draws)
        self._pair_freq_bs = _Cnt((d[0], d[1]) for d in self.draws)
        self._pair_freq_bg = _Cnt((d[0], d[2]) for d in self.draws)
        self._pair_freq_sg = _Cnt((d[1], d[2]) for d in self.draws)
        self._pair_prob_bs = {k: v / _total_draws for k, v in self._pair_freq_bs.items()}
        self._pair_prob_bg = {k: v / _total_draws for k, v in self._pair_freq_bg.items()}
        self._pair_prob_sg = {k: v / _total_draws for k, v in self._pair_freq_sg.items()}
        # 方案3: 跨位置序列转移 — 已移除1000×1000(0.76%稀疏,噪声), 保留位置级转移矩阵
        self._cross_seq_trans = None
        # 方向A: 权重退火优化（延迟到predict时首次调用）
        self._rl_weights = None
        self._rl_history = []
        # Z3组级遗漏：每个(双码,独码)组合上次出现至今的期数
        self._z3_group_miss = {}
        for i in range(len(self.draws) - 1, -1, -1):
            d = self.draws[i]
            if len(set(d)) == 2:
                sd = sorted(d)
                if d[0] == d[1]:
                    pair_d, single_d = d[0], d[2]
                elif d[0] == d[2]:
                    pair_d, single_d = d[0], d[1]
                else:
                    pair_d, single_d = d[1], d[0]
                key = (pair_d, single_d)
                if key not in self._z3_group_miss:
                    self._z3_group_miss[key] = len(self.draws) - 1 - i
        # 从未出现的Z3组合（遗漏=全部期数）
        for pair_d in range(10):
            for single_d in range(10):
                if pair_d == single_d:
                    continue
                key = (pair_d, single_d)
                if key not in self._z3_group_miss:
                    self._z3_group_miss[key] = len(self.draws)

        print(f"[P3-Fusion] 最新期号: {self.last_period} | "
              f"最新开奖: {self.prev_draw[0]} {self.prev_draw[1]} {self.prev_draw[2]}")

        # 延迟加载标记（mc缓存 + 惰性训练）
        self._mc_cache = None
        self._initialized = False

    def _init_modules(self, draws: List[Tuple[int, int, int]]):
        self.draws = draws
        self.recognizer = P3PatternRecognizer(draws)
        self.stats = P3StatisticsAnalyzer(draws)
        self.constraint = P3ConstraintEngine()

    def _build_position_stats(self, draws: List[Tuple[int, int, int]]):
        """预计算每位数字的频率、转移矩阵、遗漏值"""
        bai = [d[0] for d in draws]
        shi = [d[1] for d in draws]
        ge = [d[2] for d in draws]
        total = len(draws)

        def get_miss_fast(seq):
            miss = {}
            for i in range(len(seq) - 1, -1, -1):
                d = seq[i]
                if d not in miss:
                    miss[d] = len(seq) - 1 - i
            for d in range(10):
                if d not in miss:
                    miss[d] = len(seq)
            return miss

        self._bai_miss = get_miss_fast(bai)
        self._shi_miss = get_miss_fast(shi)
        self._ge_miss = get_miss_fast(ge)

        # 指数衰减频率（半衰期=20期，替代固定多窗口）
        def exp_decay_freq(seq, total_len):
            import math as _m
            half_life = 20.0
            lam = _m.log(2) / half_life
            result = {d: 0.01 for d in range(10)}
            total_w = 0.0
            for i in range(max(0, len(seq) - 200), len(seq)):
                age = len(seq) - 1 - i
                w = _m.exp(-lam * age)
                result[seq[i]] += w
                total_w += w
            if total_w > 0:
                for d in range(10):
                    result[d] /= total_w
            return result

        self._bai_recent = exp_decay_freq(bai, total)
        self._shi_recent = exp_decay_freq(shi, total)
        self._ge_recent = exp_decay_freq(ge, total)

        # 位置转移矩阵 10×10: P(next | prev)
        def build_transition(seq):
            tm = [[0] * 10 for _ in range(10)]
            cnt = [[0] * 10 for _ in range(10)]
            for i in range(len(seq) - 1):
                prev = seq[i]
                curr = seq[i + 1]
                cnt[prev][curr] += 1
            for prev in range(10):
                total_cnt = sum(cnt[prev])
                for curr in range(10):
                    tm[prev][curr] = cnt[prev][curr] / max(total_cnt, 1)
            return tm

        self._bai_trans = build_transition(bai)
        self._shi_trans = build_transition(shi)
        self._ge_trans = build_transition(ge)

    def _get_last_period(self) -> str:
        """从数据文件获取最新期号（复用 load_data 结果）"""
        try:
            if self._df_periods is None:
                df = pd.read_excel(self.data_path, engine='openpyxl')
                self._df_periods = df['期号'].tolist()
            return str(int(self._df_periods[-1]) + 1)
        except Exception:
            return "?"

    # ================================================================
    # 三层评分系统
    # ================================================================

    def _build_score_tables(self, prev_draw: Optional[List[int]] = None) -> Dict:
        """
        预计算每位数字的L1原始分
        返回: {'bai': {0: score, ...}, 'shi': {...}, 'ge': {...}}
        score = freq_fusion(d) * 0.5 + trans_score(d) * 0.3 + poisson_score(d) * 0.2
        
        V2.8.1: 冷号衰减 — 长遗漏数字(>50期)分数封顶,防止冷号回补偏差吞噬合理路径
        """
        if prev_draw is None:
            prev_draw = [5, 5, 5]

        l1 = {}
        positions = [
            ('bai', self._bai_recent, self._bai_miss, self._bai_trans, prev_draw[0]),
            ('shi', self._shi_recent, self._shi_miss, self._shi_trans, prev_draw[1]),
            ('ge', self._ge_recent, self._ge_miss, self._ge_trans, prev_draw[2]),
        ]

        for pos_name, recent, miss, trans, prev_val in positions:
            scores = {}
            for d in range(10):
                freq_s = recent.get(d, 0.01)
                trans_s = self._poisson.get_trans_score(pos_name, d, prev_draw, alpha=0.4)
                poisson_s = self._poisson.get_poisson_score(pos_name, d) * 0.2
                scores[d] = freq_s * 0.5 + trans_s * 0.3 + poisson_s
                # 【P3-冷号衰减】长遗漏(>40期)冷号: 遗漏越深, 衰减越大
                # V2.9.1: 增大衰减力度(启动阈值降至40期, 衰减斜率提升至0.015)
                m = miss.get(d, 0)
                if m > 40:
                    cold_damp = max(0.10, 1.0 - (m - 40) * 0.015)
                    scores[d] *= cold_damp
            l1[pos_name] = scores
        return l1

    # ================================================================
    # 评分引擎（归一化）
    # ================================================================

    @staticmethod
    def _normalize(arr: List[float]) -> List[float]:
        mn, mx = min(arr), max(arr)
        if mx == mn:
            return [0.5] * len(arr)
        return [(v - mn) / (mx - mn) for v in arr]

    def _cross_feature_score(self, bai, shi, ge, pair_prob_bs=None, pair_prob_bg=None, pair_prob_sg=None):
        """计算交叉位置特征评分（百十/百个/十个组合的频率匹配度）"""
        bs = pair_prob_bs.get((bai, shi), 0.005) if pair_prob_bs else self._pair_prob_bs.get((bai, shi), 0.005)
        bg = pair_prob_bg.get((bai, ge), 0.005) if pair_prob_bg else self._pair_prob_bg.get((bai, ge), 0.005)
        sg = pair_prob_sg.get((shi, ge), 0.005) if pair_prob_sg else self._pair_prob_sg.get((shi, ge), 0.005)
        # 平均位置对概率 * 缩放因子(使均值接近0.5)
        avg_pair = (bs + bg + sg) / 3.0
        return min(avg_pair * 30, 1.0)

    def _compute_layers(self, bai, shi, ge, prev_draw, l1_tables,
                        pattern_score=None, smart_score=None,
                        is_z3=False, z3_freqs=None, z3_miss_score=None,
                        pair_bs=None, pair_bg=None, pair_sg=None):
        """
        计算三层的原始值
        L1: sb * ss * sg * 400 (ZX/Z6) 或数字对频率 (Z3)
        L2: 和值/跨度/重号 + pattern_score + Z3组级遗漏 + 交叉位置特征
        L3: 奇偶 + smart_score
        """
        # L1
        if is_z3:
            if bai == shi:
                pair_d, single_d = bai, ge
            elif bai == ge:
                pair_d, single_d = bai, shi
            else:
                pair_d, single_d = shi, bai
            freq_table = z3_freqs if z3_freqs is not None else self._digit_overall_freq
            freq_pair = (2 * freq_table.get(pair_d, 0.1) +
                         freq_table.get(single_d, 0.1)) / 3.0
            l1 = min(freq_pair * 3.5, 1.0)
        else:
            sb = l1_tables['bai'].get(bai, 0.02)
            ss = l1_tables['shi'].get(shi, 0.02)
            sg = l1_tables['ge'].get(ge, 0.02)
            l1 = min(sb * ss * sg * 180, 1.0)  # 【P3-1】调整系数400->180(基于max_freq≈0.18)

        # L2: 和值/跨度/重号 + 模式识别 + Z3组级遗漏
        s = bai + shi + ge
        sp = max(bai, shi, ge) - min(bai, shi, ge)
        rpt = sum(1 for i, x in enumerate([bai, shi, ge]) if x == prev_draw[i])
        sum_dev = 1.0 - min(abs(s - 13.6) / 13.5, 1.0) * 0.5
        if is_z3:
            span_ok = 1.0 if 2 <= sp <= 7 else 0.5
        else:
            span_ok = 1.0 if 3 <= sp <= 6 else 0.6
        # 交叉位置特征
        cross_score = self._cross_feature_score(bai, shi, ge, pair_bs, pair_bg, pair_sg)
        l2_classic = (sum_dev * 0.4 + span_ok * 0.3 + 0.15) * (1.0 + rpt * 0.15)
        if is_z3 and z3_miss_score is not None:
            l2 = l2_classic * 0.55 + (pattern_score or 0.5) * 0.15 + z3_miss_score * 0.15 + cross_score * 0.15
        elif pattern_score is not None:
            l2 = l2_classic * 0.60 + pattern_score * 0.25 + cross_score * 0.15
        else:
            l2 = l2_classic * 0.85 + cross_score * 0.15

        # L3: 奇偶 + 数学合理性
        odd = (bai & 1) + (shi & 1) + (ge & 1)
        l3_parity = (1.0 if odd in (1, 2) else 0.3) * 0.6 + 0.4
        if smart_score is not None:
            l3 = l3_parity * 0.5 + smart_score * 0.5
        else:
            l3 = l3_parity

        return l1, l2, l3

    # ================================================================
    # 枚举评分（predict 路径）
    # ================================================================

    _GROUP_WEIGHTS = {
        'zx': (0.45, 0.25, 0.20, 0.10),
        'z3': (0.35, 0.35, 0.30, 0.00),  # Z3去掉GT(冗余), 加权到L2+L3
        'z6': (0.40, 0.25, 0.25, 0.10),
    }

    def _get_group_weights(self, group_type: str) -> Tuple:
        """获取组权重: MI调整版(如已计算)或默认版"""
        if hasattr(self, '_mi_weights') and self._mi_weights is not None:
            return self._mi_weights.get(group_type, self._GROUP_WEIGHTS[group_type])
        return self._GROUP_WEIGHTS[group_type]

    def _borda_rank(self, values: List[float]) -> List[float]:
        """
        【C】分数归一化替代Borda排名 — 保留子评分器的分数间距信息。

        原Borda排名: 最高得N分, 最低得1分 (丢失间距信息)
        新方法: Min-Max归一化到[0.1, 1.0] (保留相对间距)
        """
        n = len(values)
        if n < 2:
            return [0.5] * n
        mn, mx = min(values), max(values)
        if mx - mn < 1e-8:
            return [0.5] * n
        return [0.1 + 0.9 * (v - mn) / (mx - mn) for v in values]

    # ── 置换检验MI: 校准各层权重 (移植自P5 V1.12经验) ──

    def _compute_mi_adjusted_weights(self) -> Dict:
        """
        置换检验MI: 对4层评分(L1/L2/L3/GT)计算净MI, 调整组权重
        返回: {'zx': (w1,w2,w3,w4), 'z3': (w1,w2,w3,w4), 'z6': (w1,w2,w3,w4)}
        """
        n_perm = 30
        test_periods = min(100, len(self.draws) - 20)

        import random as rnd
        rnd.seed(42)
        np.random.seed(42)

        # 收集评分 + 命中数据
        l1_all, l2_all, l3_all, gt_all, hits = [], [], [], [], []
        for idx in range(len(self.draws) - test_periods - 1, len(self.draws) - 1):
            train = self.draws[:idx + 1]
            actual = list(self.draws[idx + 1])
            prev_draw = list(train[-1])

            bt_recognizer = P3PatternRecognizer(train)
            l1_t = self._backtest_build_score_tables([d[0] for d in train], prev_draw[0])[0]

            for _ in range(30):  # 每期30个随机候选
                digits = [rnd.randint(0, 9) for _ in range(3)]
                l1_tables = {'bai': l1_t, 'shi': l1_t, 'ge': l1_t}
                ps = bt_recognizer.score_pattern(digits, prev_draw)
                ss = P3MathFilter.smart_score(digits)
                gt = P3GameTheoryAnalyzer(train).expected_value(digits)
                l1, l2, l3 = self._compute_layers(
                    digits[0], digits[1], digits[2], prev_draw,
                    l1_tables, ps, ss)
                l1_all.append(l1)
                l2_all.append(l2)
                l3_all.append(l3)
                gt_all.append(gt)
                hits.append(1 if digits == actual else 0)

        n = len(hits)
        if n < 100:
            return self._get_default_group_weights()

        def _calc_mi(scores, labels):
            bins = np.linspace(min(scores), max(scores), 11)
            dig = np.digitize(scores, bins) - 1
            joint = np.zeros((10, 2))
            for i in range(n):
                b = dig[i]
                h = labels[i]
                if 0 <= b < 10:
                    joint[b, h] += 1
            joint /= max(n, 1)
            pb = joint.sum(axis=1)
            ph = joint.sum(axis=0)
            mi = 0.0
            for b in range(10):
                for h in range(2):
                    if joint[b, h] > 0:
                        denom = max(pb[b] * ph[h], 1e-10)
                        mi += joint[b, h] * np.log(joint[b, h] / denom)
            return mi

        scores = [l1_all, l2_all, l3_all, gt_all]
        labels = np.array(hits, dtype=np.int32)

        # 实际MI
        actual_mi = [_calc_mi(s, labels) for s in scores]

        # 置换检验
        perm_mi = np.zeros((n_perm, 4))
        for p in range(n_perm):
            np.random.shuffle(labels)
            for li in range(4):
                perm_mi[p, li] = _calc_mi(scores[li], labels)

        noise_floor = perm_mi.mean(axis=0)
        net_mi = [max(0, actual_mi[i] - noise_floor[i]) for i in range(4)]

        layer_names = ['L1频率','L2模式','L3数学','L4博弈']
        for i, name in enumerate(layer_names):
            noise_pct = (1 - net_mi[i]/max(actual_mi[i], 1e-10))*100 if actual_mi[i] > 0 else 100
            print(f"  [P3-MI] {name}: raw={actual_mi[i]:.4f} noise={noise_pct:.0f}% net={net_mi[i]:.4f}")

        # 生成调整后权重: net权重/总net, L4(L4=索引3)上限0.15
        total_net = sum(net_mi) or 1e-10
        adj_w = [max(0.05, n / total_net) for n in net_mi]  # 最小0.05
        adj_w[3] = min(adj_w[3], 0.3)  # L4上限0.30
        total = sum(adj_w)
        adj_w = [w / total for w in adj_w]

        # 每组使用不同默认权重, 但净MI校正统一应用
        result = {}
        for gtype, default in [('zx', (0.45, 0.25, 0.20, 0.10)),
                                ('z3', (0.35, 0.35, 0.30, 0.00)),
                                ('z6', (0.40, 0.25, 0.25, 0.10))]:
            # 融合默认权重和MI校正
            blended = [default[i] * 0.6 + adj_w[i] * 0.4 for i in range(4)]
            total_b = sum(blended)
            result[gtype] = tuple(round(w / total_b, 3) for w in blended)

        print(f"  [P3-MI] \u2795 调整后权重: ZX={result['zx']} Z3={result['z3']} Z6={result['z6']}")
        return result

    def _get_default_group_weights(self) -> Dict:
        return {'zx': (0.45, 0.25, 0.20, 0.10),
                'z3': (0.35, 0.35, 0.30, 0.00),
                'z6': (0.40, 0.25, 0.25, 0.10)}

    def _anneal_weights(self) -> List[float]:
        """方向A: 子模型互信息 + 权重搜索（基于50期验证数据）"""
        n_sub = 8
        periods = 50
        # 收集验证数据: 每期每个候选的子模型排名
        all_ranks = []  # list of (actual_idx, [8个sub_model_ranks])
        all_actuals = []

        from p3_fusion_complete import P3MathFilter as _PMF
        for idx in range(len(self.draws) - periods - 1, len(self.draws) - 1):
            train = self.draws[:idx + 1]
            actual = list(self.draws[idx + 1])
            prev_draw = list(train[-1])
            bai_s = [d[0] for d in train]
            shi_s = [d[1] for d in train]
            ge_s = [d[2] for d in train]
            l1_bai, t1_bai, t2_bai, t3_bai, _ = self._backtest_build_score_tables(bai_s, prev_draw[0])
            l1_shi, t1_shi, t2_shi, t3_shi, _ = self._backtest_build_score_tables(shi_s, prev_draw[1])
            l1_ge, t1_ge, t2_ge, t3_ge, _ = self._backtest_build_score_tables(ge_s, prev_draw[2])
            _a = 0.35
            l1_tables = {}
            sb = {d: l1_bai.get(d,0.01) + t1_bai[prev_draw[0]][d]*0.3 for d in range(10)}
            ss = {d: l1_shi.get(d,0.01) + (t1_shi[prev_draw[1]][d]*0.6 + t2_shi[prev_draw[0]*10+prev_draw[1]][d]*0.4)*0.3 for d in range(10)}
            sg = {d: l1_ge.get(d,0.01) + ((1-_a)*t1_ge[prev_draw[2]][d] + (_a*0.5)*t2_ge[prev_draw[1]*10+prev_draw[2]][d] + (_a*0.5)*t3_ge[prev_draw[0]*100+prev_draw[1]*10+prev_draw[2]][d])*0.3 for d in range(10)}
            l1_tables = {'bai': sb, 'shi': ss, 'ge': sg}

            # 子模型计算仅在debug时启用(默认跳过以节省CPU)
            ranks = None
            if getattr(self, '_debug_submodels', False):
                sub_scores = [[] for _ in range(n_sub)]
                for bai in range(10):
                    for shi in range(10):
                        for ge in range(10):
                            dig = [bai, shi, ge]
                            gt = self._game_theory.expected_value(dig)
                            l1, l2, l3 = self._compute_layers(bai, shi, ge, prev_draw, l1_tables, 0.5, 0.5)
                            w1z, w2z, w3z, w4z = self._GROUP_WEIGHTS['zx']
                            sub_scores[0].append(l1*w1z + l2*w2z + l3*w3z + gt*w4z)
                            sub_scores[1].append(0.5+0.5)
                            sub_scores[2].append(self._zx_trans_prob(bai, shi, ge, prev_draw))
                            sub_scores[3].append(1.0)
                            sub_scores[4].append(0.5)
                            sub_scores[5].append(0.5)
                            sub_scores[6].append(self._cross_seq_score(dig, prev_draw))
                            sub_scores[7].append(0.5)

                ranks = []
                for s in sub_scores:
                    pr = [(v, i) for i, v in enumerate(s)]
                    pr.sort(key=lambda x: -x[0])
                    r = [0]*1000
                    for pos, (_, i) in enumerate(pr):
                        r[i] = pos + 1
                    ranks.append(r)

            actual_idx = actual[0]*100 + actual[1]*10 + actual[2]
            all_ranks.append(ranks)
            all_actuals.append(actual_idx)

        # 尝试权重组合
        configs = [[1.0]*8, [3,1,1,1,1,1,1,1], [2,2,1,1,1,1,1,1],
                   [2,1,2,1,1,1,1,1], [2,1,1,2,1,1,1,1],
                   [2,1,1,1,2,1,1,1], [2,1,1,1,1,1,2,1],
                   [1,1,1,1,1,1,1,3]]
        best_w = [1.0]*8
        best_hits = -1
        for w in configs:
            total = sum(w)
            hits = 0
            for t in range(len(all_ranks)):
                final = [sum(w[s]*all_ranks[t][s][i] for s in range(n_sub)) / total for i in range(1000)]
                top10 = sorted(range(1000), key=lambda i: -final[i])[:10]
                if all_actuals[t] in top10:
                    hits += 1
            if hits > best_hits:
                best_hits = hits
                best_w = [w_i / total * n_sub for w_i in w]

        return best_w

    def _cross_seq_score(self, digits: List[int], prev_draw: List[int],
                          trans_matrix=None) -> float:
        """
        跨位置序列匹配 — 改用位置级转移矩阵乘积 (原1000×1000已移除)
        """
        # 使用3个位置级转移矩阵乘积: P(百)·P(十)·P(个)
        bai_p = self._bai_trans[prev_draw[0]][digits[0]]
        shi_p = self._shi_trans[prev_draw[1]][digits[1]]
        ge_p = self._ge_trans[prev_draw[2]][digits[2]]
        return bai_p * shi_p * ge_p * 1000  # 缩放回[0,1]量纲

    def _group_consensus_score(self, digits: List[int], combined_vals: List[float]) -> float:
        """
        组共识评分：按和值段×跨度段×奇偶分组，组内综合分的均值作为组评分
        每个候选号码取其所在组的组评分
        """
        # 将1000个候选聚为约50个组
        group_scores = {}
        group_counts = {}
        idx = 0
        for bai in range(10):
            for shi in range(10):
                for ge in range(10):
                    s = bai + shi + ge
                    sp = max(bai, shi, ge) - min(bai, shi, ge)
                    odd = (bai & 1) + (shi & 1) + (ge & 1)
                    gkey = (s // 3, sp // 2, 0 if odd == 0 else 1 if odd == 3 else 2)
                    group_scores[gkey] = group_scores.get(gkey, 0) + combined_vals[idx]
                    group_counts[gkey] = group_counts.get(gkey, 0) + 1
                    idx += 1
        for k in group_scores:
            group_scores[k] /= group_counts[k]
        # 每个候选取其组评分
        idx = 0
        result = [0.0] * len(combined_vals)
        for bai in range(10):
            for shi in range(10):
                for ge in range(10):
                    s = bai + shi + ge
                    sp = max(bai, shi, ge) - min(bai, shi, ge)
                    odd = (bai & 1) + (shi & 1) + (ge & 1)
                    gkey = (s // 3, sp // 2, 0 if odd == 0 else 1 if odd == 3 else 2)
                    result[idx] = group_scores.get(gkey, 0.5)
                    idx += 1
        return result

    def _multi_period_mc(self, prev_draw: List[int], n_sim: int = 3000) -> Dict[tuple, int]:
        """
        多期蒙特卡洛共识：模拟 t+1 和 t+2，取两期都出现的组合
        首次计算后缓存结果，避免重复计算
        """
        if self._mc_cache is not None:
            return self._mc_cache
        import random
        pm = self._poisson
        t1_counts = {}
        t2_counts = {}
        for _ in range(n_sim):
            # t+1
            b1 = random.choices(range(10), weights=pm._trans1_bai[prev_draw[0]], k=1)[0]
            s1 = random.choices(range(10), weights=pm._trans2_shi[prev_draw[0]*10+prev_draw[1]], k=1)[0]
            g1 = random.choices(range(10), weights=pm._trans3_ge[prev_draw[0]*100+prev_draw[1]*10+prev_draw[2]], k=1)[0]
            key1 = (b1, s1, g1)
            t1_counts[key1] = t1_counts.get(key1, 0) + 1
            # t+2: 用 t+1 作为输入
            b2 = random.choices(range(10), weights=pm._trans1_bai[b1], k=1)[0]
            s2 = random.choices(range(10), weights=pm._trans2_shi[b1*10+s1], k=1)[0]
            g2 = random.choices(range(10), weights=pm._trans3_ge[b1*100+s1*10+g1], k=1)[0]
            key2 = (b2, s2, g2)
            t2_counts[key2] = t2_counts.get(key2, 0) + 1
        # 取两期都出现的组合（交集），未出现的给0分
        result = {}
        for k in set(list(t1_counts.keys()) + list(t2_counts.keys())):
            result[k] = t1_counts.get(k, 0) + t2_counts.get(k, 0)
        self._mc_cache = result
        return result

    def _fluctuation_score(self, digits: List[int], window: int = 30) -> float:
        """
        涨落检测：用二项检验检测每位数字近期频率是否偏离期望(1/10)
        显著偏离 → 均值回归 → 加分(冷号)或减分(热号)
        """
        from scipy import stats as _st
        bai_s = [d[0] for d in self.draws[-window:]] if len(self.draws) >= window else [d[0] for d in self.draws]
        shi_s = [d[1] for d in self.draws[-window:]] if len(self.draws) >= window else [d[1] for d in self.draws]
        ge_s = [d[2] for d in self.draws[-window:]] if len(self.draws) >= window else [d[2] for d in self.draws]
        score = 0.0
        for seq, pos_d in [(bai_s, digits[0]), (shi_s, digits[1]), (ge_s, digits[2])]:
            n = len(seq)
            k = sum(1 for d in seq if d == pos_d)
            expected = n / 10.0
            if k > expected:
                # 过热 → 减分（均值回归预期）
                try:
                    from scipy.stats import binomtest as _binomtest
                    p_val = _binomtest(k, n, 0.1, alternative='greater').pvalue
                except ImportError:
                    from scipy.stats import binom_test as _binomtest
                    p_val = _binomtest(k, n, 0.1, alternative='greater')
                score -= min(p_val * 5, 0.3)
            elif k < expected:
                # 过冷 → 加分（均值回归预期）
                try:
                    from scipy.stats import binomtest as _binomtest
                    p_val = _binomtest(k, n, 0.1, alternative='less').pvalue
                except ImportError:
                    from scipy.stats import binom_test as _binomtest
                    p_val = _binomtest(k, n, 0.1, alternative='less')
                score += min(p_val * 5, 0.3)
        return max(0.0, 0.5 + score)  # centered at 0.5

    def _monte_carlo_consensus(self, prev_draw: List[int], n_sim: int = 5000) -> Dict[tuple, int]:
        """蒙特卡洛模拟：用1/2/3阶转移矩阵生成n_sim个下一期序列，返回每个组合的出现次数"""
        import random
        pm = self._poisson
        counts = {}
        for _ in range(n_sim):
            # 百位：一阶转移
            probs_bai = pm._trans1_bai[prev_draw[0]]
            bai = random.choices(range(10), weights=probs_bai, k=1)[0]
            # 十位：二阶转移
            probs_shi = pm._trans2_shi[prev_draw[0]*10 + prev_draw[1]]
            shi = random.choices(range(10), weights=probs_shi, k=1)[0]
            # 个位：三阶转移
            probs_ge = pm._trans3_ge[prev_draw[0]*100 + prev_draw[1]*10 + prev_draw[2]]
            ge = random.choices(range(10), weights=probs_ge, k=1)[0]
            key = (bai, shi, ge)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _chi_square_deviation(self, window: int = 20) -> float:
        """
        方案B: 物理偏态检测 — 对近window期每位数字做卡方检验
        返回: p值 (越小=越偏离均匀→信号越强)
        """
        from scipy import stats as _st
        p_vals = []
        for pos_hist in [[d[0] for d in self.draws[-window:]],
                         [d[1] for d in self.draws[-window:]],
                         [d[2] for d in self.draws[-window:]]]:
            if len(pos_hist) < 10:
                continue
            obs = [sum(1 for x in pos_hist if x == d) for d in range(10)]
            _, p = _st.chisquare(obs)
            p_vals.append(p)
        return min(p_vals) if p_vals else 1.0

    def _kmedoids_cover(self, candidates: List[Dict], n: int) -> List[Dict]:
        """
        方案D: 组合覆盖优化 — 贪心最远优先选择
        在Top 30中选n个, 使特征空间覆盖最大化
        """
        def _feat_vec(digits):
            s = sum(digits)
            sp = max(digits) - min(digits)
            odd = sum(1 for d in digits if d & 1)
            big = sum(1 for d in digits if d >= 5)
            prime = sum(1 for d in digits if d in (2,3,5,7))
            route = tuple(d % 3 for d in digits)
            return (s/27.0, sp/9.0, odd/3.0, big/3.0, prime/3.0, route[0]/3.0, route[1]/3.0, route[2]/3.0)
        pool = candidates[:min(30, len(candidates))]
        if len(pool) <= n:
            return pool[:n]
        selected = [pool[0]]
        remaining = pool[1:]
        while len(selected) < n and remaining:
            best_idx, best_dist = 0, -1
            for i, cand in enumerate(remaining):
                v = _feat_vec(cand['digits'])
                min_dist = min(sum((a-b)**2 for a,b in zip(v, _feat_vec(s['digits'])))**0.5 for s in selected)
                if min_dist > best_dist:
                    best_dist = min_dist
                    best_idx = i
            selected.append(remaining.pop(best_idx))
        selected.sort(key=lambda x: -x['final_score'])
        return selected

    def _enforce_position_diversity(self, selected: List[Dict], all_candidates: List[Dict], n: int) -> List[Dict]:
        """
        V2.9.1: 位置多样性强制约束 — 贪心重建法
        从最高分候选开始逐一选取, 确保同一位置同一数字最多不超过ceil(N/3)次
        如无法满足N注, 逐步放宽约束至全部放开
        """
        if not selected:
            return selected

        import math

        # 尝试3种严格度, 从最严格到最宽松
        for strictness in [2, 3, 4, 999]:
            result = []
            used_tuples = set()
            pos_cnts = [Counter(), Counter(), Counter()]

            for cand in all_candidates:
                d = cand['digits']
                t = tuple(d)
                if t in used_tuples:
                    continue
                # 检查是否违反位置约束
                ok = True
                for pos in range(3):
                    if pos_cnts[pos][d[pos]] >= strictness:
                        ok = False
                        break
                if not ok:
                    continue
                result.append(cand)
                used_tuples.add(t)
                for pos in range(3):
                    pos_cnts[pos][d[pos]] += 1
                if len(result) >= n:
                    break

            if len(result) >= n or strictness == 999:
                break

        return result[:n]

    def _kelly_bet(self, p: float, odds: float = 520.0) -> float:
        """
        方向C: 凯利公式 f* = (bp - q) / b
        p = 命中概率, q = 1-p, b = 赔率
        排列三直选: 2元中1040元 → odds = (1040-2)/2 = 519, 用520近似
        返回: 建议投注比例(占本金的百分比)
        """
        if p <= 0 or p >= 1:
            return 0.0
        q = 1.0 - p
        b = odds
        f = (b * p - q) / b
        return max(0.0, min(f, 0.25))  # 上限25%防过度

    def recommend_bet(self, n_zx: int = None, n_z3: int = None, n_z6: int = None,
                      bankroll: float = 100.0) -> Dict[str, Any]:
        """方向C: 推荐投注(含凯利注额)"""
        result = self.predict(n_zx=n_zx, n_z3=n_z3, n_z6=n_z6)
        # 估算每注的命中概率
        for bet in result.get('zx_bets', []):
            p = bet['hit_probability'] / 100.0 if bet['hit_probability'] > 0 else 1.0/1000
            bet['kelly_pct'] = round(self._kelly_bet(p) * 100, 2)
            bet['bet_amount'] = round(bankroll * self._kelly_bet(p), 2)
        result['bankroll'] = bankroll
        result['total_bet'] = sum(b.get('bet_amount', 0) for b in result.get('zx_bets', []))
        return result

    def _zx_trans_prob(self, bai, shi, ge, prev_draw):
        """纯转移概率(1阶bai+2阶shi+3阶ge)，无频率成分"""
        pm = self._poisson
        p_bai = pm._trans1_bai[prev_draw[0]][bai]
        if len(prev_draw) >= 2:
            p_shi = pm._trans2_shi[prev_draw[0]*10+prev_draw[1]][shi]
        else:
            p_shi = pm._trans1_shi[prev_draw[1]][shi] if len(prev_draw) > 1 else 0.1
        if len(prev_draw) >= 3:
            p_ge = pm._trans3_ge[prev_draw[0]*100+prev_draw[1]*10+prev_draw[2]][ge]
        else:
            p_ge = pm._trans2_ge[prev_draw[1]*10+prev_draw[2]][ge] if len(prev_draw) > 2 else pm._trans1_ge[prev_draw[2]][ge] if len(prev_draw) > 2 else 0.1
        return p_bai * p_shi * p_ge

    def _enumerate_direct(self, prev_draw: List[int], mc_counts: Optional[Dict[tuple, int]] = None) -> List[Dict[str, Any]]:
        """枚举全部1000个直选组合，Borda排名融合(L1已被实证为噪声，用转移秩替代)"""
        l1_tables = self._build_score_tables(prev_draw)
        w1, w2, w3, w4 = self._get_group_weights('zx')
        n = 1000
        candidates = []
        combined_vals = []
        ps_ss_vals = []
        trans_vals = []
        mc_multi_vals = []
        fluct_vals = []
        cross_seq_vals = []
        ml_vals = []
        if mc_counts is not None:
            mc_multi_counts = mc_counts
        else:
            mc_multi_counts = self._multi_period_mc(prev_draw, 3000)

        for bai in range(10):
            for shi in range(10):
                for ge in range(10):
                    digits = [bai, shi, ge]
                    ps = self.recognizer.score_pattern(digits, prev_draw)
                    ss = P3MathFilter.smart_score(digits)
                    gt = self._game_theory.expected_value(digits)
                    l1, l2, l3 = self._compute_layers(bai, shi, ge, prev_draw, l1_tables, ps, ss)
                    combined = l1*w1 + l2*w2 + l3*w3 + gt*w4
                    tp = self._zx_trans_prob(bai, shi, ge, prev_draw)
                    mc = mc_multi_counts.get((bai, shi, ge), 0)
                    fs = self._fluctuation_score(digits)
                    cs = self._cross_seq_score(digits, prev_draw)
                    candidates.append(digits)
                    combined_vals.append(combined)
                    ps_ss_vals.append((ps or 0.5) + (ss or 0.5))
                    trans_vals.append(tp)
                    mc_multi_vals.append(mc)
                    fluct_vals.append(fs)
                    cross_seq_vals.append(cs)
                    ml_vals.append(0.5)  # NN已移除, 恒为中性

        # 组共识
        group_vals = self._group_consensus_score(None, combined_vals)

        # P2: 8维Borda冗余相关性分析
        _vecs = [combined_vals, ps_ss_vals, trans_vals, mc_multi_vals, group_vals, fluct_vals, cross_seq_vals, ml_vals]
        _corr = np.corrcoef(_vecs)
        _high_pairs = []
        for _i in range(8):
            for _j in range(_i + 1, 8):
                if abs(_corr[_i][_j]) > 0.8:
                    _high_pairs.append((_i, _j, round(_corr[_i][_j], 3)))
        if _high_pairs:
            print(f"[P3-Borda] ⚠️ 高相关子模型: {_high_pairs}")

        # Borda: RL动态权重(默认值8维, 退火仅在IRL路径中生效)
        # V2.8.1: trans_vals权重提升至0.35, combined/ps_ss降权, 增强转移概率对冲冷号偏差
        w = self._rl_weights if self._rl_weights is not None else [0.05, 0.20, 0.35, 0.05, 0.15, 0.05, 0.05, 0.10]
        b1 = self._borda_rank(combined_vals)
        b2 = self._borda_rank(ps_ss_vals)
        b3 = self._borda_rank(trans_vals)
        b4 = self._borda_rank(mc_multi_vals)
        b5 = self._borda_rank(group_vals)
        b6 = self._borda_rank(fluct_vals)
        b7 = self._borda_rank(cross_seq_vals)
        b8 = self._borda_rank(ml_vals)
        total_w = sum(w)
        scored = [{'digits': candidates[i],
            'final_score': round(combined_vals[i] * 100, 4),
            'hit_probability': round(min(combined_vals[i] * 100, 99.9), 1),
        } for i in range(n)]
        scored.sort(key=lambda x: -x['final_score'])
        return scored

    def _enumerate_z3(self, prev_draw: List[int], mc_counts: Optional[Dict[tuple, int]] = None) -> List[Dict[str, Any]]:
        """枚举全部90个组三组合，取3排列最高分，Borda排名融合"""
        # mc_counts参数保留用于接口一致性（当前Z3不使用MC）
        l1_tables = self._build_score_tables(prev_draw)
        from modules.p3_poisson_model import P3PoissonModel as _PPM
        w1, w2, w3, w4 = self._get_group_weights('z3')
        n = 90

        best = {}
        for d in range(10):
            for diff in range(10):
                if diff == d:
                    continue
                for perm in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
                    b = d if perm[0] == 0 else diff
                    s = d if perm[1] == 0 else diff
                    g = d if perm[2] == 0 else diff
                    digits = [b, s, g]
                    ps = self.recognizer.score_pattern(digits, prev_draw)
                    ss = P3MathFilter.smart_score(digits)
                    gt = self._game_theory.expected_value(digits)
                    _miss = self._z3_group_miss.get((d, diff), len(self.draws))
                    _z3_prob = (2 * self._digit_overall_freq.get(d, 0.1) +
                                self._digit_overall_freq.get(diff, 0.1)) / 3.0
                    _miss_score = _PPM.poisson_anomaly(_miss, _z3_prob)
                    cs = self._cross_feature_score(b, s, g)
                    l1, l2, l3 = self._compute_layers(b, s, g, prev_draw, l1_tables, ps, ss,
                                                       is_z3=True, z3_miss_score=_miss_score)
                    fs = l1*w1 + l2*w2 + l3*w3 + gt*w4
                    key = tuple(sorted([b, s, g]))
                    if key not in best or fs > best[key][0]:
                        best[key] = (l1, l2, l3, gt, ps, ss, cs, fs, _miss_score)

        keys = list(best.keys())
        l1s = [best[k][0] for k in keys]
        gts = [best[k][3] for k in keys]
        ps_ss = [(best[k][4] or 0.5) + (best[k][5] or 0.5) for k in keys]
        css = [best[k][6] for k in keys]
        fss = [best[k][7] for k in keys]
        miss_s = [best[k][8] for k in keys]

        # Z3 Borda: 【1】裁剪ZX专用评分器, 使用L1+L2+L3+cross_feature
        b1 = self._borda_rank(l1s)      # L1位置频率
        l2s_raw = []
        for k in keys:
            _, l2, _ = best[k][:3]  # l2 from _compute_layers
            l2s_raw.append(l2)
        b2 = self._borda_rank(l2s_raw)   # L2模式匹配
        l3s_raw = []
        for k in keys:
            _, _, l3 = best[k][:3]
            l3s_raw.append(l3)
        b3 = self._borda_rank(l3s_raw)   # L3数学合理性
        b4 = self._borda_rank(css)       # cross_feature
        b5 = self._borda_rank(miss_s)    # 组级遗漏
        max_b = n * 5

        scored = [{
            'digits': list(keys[i]),
            'final_score': round(fss[i], 4),
            'hit_probability': 0,
        } for i in range(n)]
        scored.sort(key=lambda x: -x['final_score'])
        return scored

    def _enumerate_z6(self, prev_draw: List[int], mc_counts: Optional[Dict[tuple, int]] = None) -> List[Dict[str, Any]]:
        """枚举全部120个组六组合，取6排列最高分，Borda排名融合(去L1+转移秩)"""
        # mc_counts参数保留用于接口一致性（当前Z6不使用MC）
        l1_tables = self._build_score_tables(prev_draw)
        w1, w2, w3, w4 = self._get_group_weights('z6')
        n = 120
        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

        best = {}
        for d1 in range(10):
            for d2 in range(d1 + 1, 10):
                for d3 in range(d2 + 1, 10):
                    best_tp = -1
                    best_fs = -1
                    best_ps_ss = 0
                    for p in perms:
                        b = d1 if p[0] == 0 else d2 if p[0] == 1 else d3
                        s = d1 if p[1] == 0 else d2 if p[1] == 1 else d3
                        g = d1 if p[2] == 0 else d2 if p[2] == 1 else d3
                        digits = [b, s, g]
                        ps = self.recognizer.score_pattern(digits, prev_draw)
                        ss = P3MathFilter.smart_score(digits)
                        gt = self._game_theory.expected_value(digits)
                        l1, l2, l3 = self._compute_layers(b, s, g, prev_draw, l1_tables, ps, ss)
                        fs = l1*w1 + l2*w2 + l3*w3 + gt*w4
                        tp = self._zx_trans_prob(b, s, g, prev_draw)
                        if fs > best_fs:
                            best_fs = fs
                            best_ps_ss = (ps or 0.5) + (ss or 0.5)
                            best_tp = tp
                    key = (d1, d2, d3)
                    if key not in best or best_fs > best[key][0]:
                        best[key] = (best_fs, best_ps_ss, best_tp)

        keys = list(best.keys())
        fss = [best[k][0] for k in keys]
        ps_ss = [best[k][1] for k in keys]
        tps = [best[k][2] for k in keys]

        # Z6 Borda: 【1】去掉_zx_trans_prob, 使用L1+L2+L3+cross_feature+gt
        # 从best中提取各维度
        l1s_z6, l2s_z6, l3s_z6, gts_z6 = [], [], [], []
        css_z6 = self._cross_feature_score(
            list(keys[0])[0] if keys else 0,  # fallback
            list(keys[0])[1] if keys else 0,
            list(keys[0])[2] if keys else 0,
        )
        # 从best字典中逐个提取
        z6_l1, z6_l2, z6_l3, z6_gt = {}, {}, {}, {}
        z6_ps_ss = {}
        k_list = list(keys)
        # 【3】计算各维度评分(交叉特征移出排列循环)
        z6_l1, z6_l2, z6_l3, z6_gt, z6_cs = {}, {}, {}, {}, {}
        for k in k_list:
            d1, d2, d3 = k
            best_combo = None
            best_score = -1
            for p in perms:
                b = d1 if p[0] == 0 else d2 if p[0] == 1 else d3
                s = d1 if p[1] == 0 else d2 if p[1] == 1 else d3
                g = d1 if p[2] == 0 else d2 if p[2] == 1 else d3
                ps = self.recognizer.score_pattern([b,s,g], prev_draw)
                ss = P3MathFilter.smart_score([b,s,g])
                gt = self._game_theory.expected_value([b,s,g])
                l1, l2, l3 = self._compute_layers(b, s, g, prev_draw, l1_tables, ps, ss)
                fs = l1*w1 + l2*w2 + l3*w3 + gt*w4
                if fs > best_score:
                    best_score = fs
                    best_combo = (b, s, g, l1, l2, l3, gt)
            # 交叉特征只需计算一次(与排列顺序无关)
            z6_cs[k] = self._cross_feature_score(best_combo[0], best_combo[1], best_combo[2])
            z6_l1[k], z6_l2[k], z6_l3[k], z6_gt[k] = best_combo[3], best_combo[4], best_combo[5], best_combo[6]

        l1_v = [z6_l1[k] for k in k_list]
        l2_v = [z6_l2[k] for k in k_list]
        l3_v = [z6_l3[k] for k in k_list]
        gt_v = [z6_gt[k] for k in k_list]
        cs_v = [z6_cs[k] for k in k_list]

        b1 = self._borda_rank(l1_v)
        b2 = self._borda_rank(l2_v)
        b3 = self._borda_rank(l3_v)
        b4 = self._borda_rank(gt_v)
        b5 = self._borda_rank(cs_v)
        max_b = n * 5

        scored = [{
            'digits': list(k_list[i]),
            'final_score': round(l1_v[i]*w1 + l2_v[i]*w2 + l3_v[i]*w3 + gt_v[i]*w4, 4),
            'hit_probability': 0,
        } for i in range(n)]
        scored.sort(key=lambda x: -x['final_score'])
        return scored

        max_b = n * 5

        scored = [{
            'digits': list(k_list[i]),
            'final_score': round(l1_v[i]*w1 + l2_v[i]*w2 + l3_v[i]*w3 + gt_v[i]*w4, 4),
            'hit_probability': 0,
        } for i in range(n)]
        scored.sort(key=lambda x: -x['final_score'])
        return scored

    # ================================================================
    # 隔期重号注入
    # ================================================================

    def _inject_repeat(self, scored, prev_draw, n_repeat=2):
        """【2】条件注入 — 重号已自然进入Top10时不强制注入"""
        already_has_repeat = any(
            sum(1 for i in range(3) if c['digits'][i] == prev_draw[i]) >= 1
            for c in scored[:10]
        )
        if already_has_repeat:
            return scored

        l1_tables = self._build_score_tables(prev_draw)
        top_set = set(tuple(c['digits']) for c in scored[:10])
        repeats = []
        for pos in range(3):
            base = prev_draw[:]
            for alt in range(10):
                digits = base[:]
                digits[pos] = alt
                if tuple(digits) not in top_set:
                    ps = self.recognizer.score_pattern(digits, prev_draw)
                    ss = P3MathFilter.smart_score(digits)
                    gt = self._game_theory.expected_value(digits)
                    l1, l2, l3 = self._compute_layers(*digits, prev_draw, l1_tables, ps, ss)
                    w1, w2, w3, w4 = self._get_group_weights('zx')
                    score = l1 * w1 + l2 * w2 + l3 * w3 + gt * w4
                    repeats.append({'digits': digits, 'final_score': score,
                                    'hit_probability': round(min(score * 15, 10.0), 1)})

        repeats.sort(key=lambda x: -x['final_score'])
        scored_ext = scored[:]
        seen = set(tuple(c['digits']) for c in scored_ext)
        for r in repeats:
            if len(scored_ext) >= len(scored) + n_repeat:
                break
            if tuple(r['digits']) not in seen:
                scored_ext.append(r)
                seen.add(tuple(r['digits']))
        return scored_ext

    def _recommend_alloc(self, n_total: int = 10) -> tuple:
        """基于近期组类型分布动态分配推荐注数"""
        recent = self.draws[-50:] if len(self.draws) >= 50 else self.draws
        types = Counter()
        for d in recent:
            s = len(set(d))
            if s == 1: types['豹子'] += 1
            elif s == 2: types['组三'] += 1
            else: types['组六'] += 1
        total = sum(types.values())
        if total == 0:
            return n_total, max(3, int(n_total * 0.3)), max(3, int(n_total * 0.7))
        p_z3 = types['组三'] / total
        # 组三最少3注，最多n_total-3注，其余给组六
        n3 = max(3, min(int(n_total * p_z3), n_total - 3))
        n6 = max(3, n_total - n3)
        # 【4】根据命中历史动态调整：如果Z3近期命中率高，增加Z3配额
        if hasattr(self, '_hit_history') and len(self._hit_history) >= 3:
            avg_hits = sum(self._hit_history) / len(self._hit_history)
            if avg_hits >= 1.5:
                pass  # 命中正常，不做调整
            elif avg_hits < 1.0:
                # 命中偏低，增加ZX配额（直选更稳定）
                n_total = min(n_total + 2, 15)
        return n_total, n3, n6
    def predict(self, n_zx: int = 10, n_z3: int = None, n_z6: int = None,
                n_total: int = 10,
                include_compound: bool = True, _silent: bool = False,
                actual_draw: List[int] = None) -> Dict[str, Any]:
        """
        全量枚举 → 三层评分 → 多样性选择
        """
        # P1: 延迟初始化（避免OOM）
        if not self._initialized:
            if self._rl_weights is None:
                # V2.8.1: trans_vals权重提升至0.35, 增强转移概率对冲冷号偏差
                self._rl_weights = [0.05, 0.20, 0.35, 0.05, 0.15, 0.05, 0.05, 0.10]
            self._ensure_mi_weights()
            self._initialized = True

        if not _silent:
            print(f"\n{'='*50}")
            print(f"  排列3 第{self.last_period}期 全量枚举+三层评分")
            print(f"{'='*50}")

        # 方案D: K-Medoids覆盖优化（替代原简单Top-N选择）
        if n_z3 is None or n_z6 is None:
            _zx_cnt, n_z3, n_z6 = self._recommend_alloc(n_total)

        prev_draw = self.prev_draw

        # P4: 预计算MC（三个枚举方法共享，避免重复计算3次）
        mc_multi_counts = self._multi_period_mc(prev_draw, 3000)

        zx_all = self._enumerate_direct(prev_draw, mc_counts=mc_multi_counts)
        z3_all = self._enumerate_z3(prev_draw, mc_counts=mc_multi_counts)
        z6_all = self._enumerate_z6(prev_draw, mc_counts=mc_multi_counts)

        zx_with_repeat = self._inject_repeat(zx_all, prev_draw, n_repeat=2)

        zx_final = self._kmedoids_cover(zx_with_repeat, n_zx)
        # V2.8.1: 位置多样性强制约束 — 同一位置同一数字最多出现2次
        zx_final = self._enforce_position_diversity(zx_final, zx_with_repeat, n_zx)
        z3_final = self._diverse_selection(z3_all, n_z3)
        z6_final = self._diverse_selection(z6_all, n_z6)

        if not _silent:
            print(f"[P3-Fusion] K-Medoids覆盖: ZX={n_zx}注 Z3={n_z3}注 Z6={n_z6}注")

        result = {
            'period': self.last_period,
        'confidence': round(self._estimate_confidence(zx_all), 4),
            'zx_bets': zx_final,
            'z3_bets': z3_final,
            'z6_bets': z6_final,
        }

        if include_compound:
            result['compound_bets'] = self._generate_compound(zx_all[:30])

        # RL更新：若提供了实际开奖，更新子模型权重
        if actual_draw is not None:
            self._meta_learn_update(actual_draw, zx_all)

        # 自动存储预测结果（保留最近2期，用于复盘对比）
        try:
            from prediction_store import store_prediction
            store_prediction(
                period=self.last_period,
                zx_bets=zx_final,
                z3_bets=z3_final if z3_final else None,
                z6_bets=z6_final if z6_final else None,
                compound_bets=result.get('compound_bets') if include_compound else None,
            )
        except Exception as e:
            if not _silent:
                print(f"[P3-Fusion] ⚠️ 存储失败: {e}")

        return result

    def _diverse_selection(self, candidates: List[Dict], n: int) -> List[Dict]:
        """多样性选择（V1.3.1: 和值+跨度双维度分桶）"""
        if not candidates:
            return []

        # 双维度分桶：和值段 × 跨度段
        buckets = defaultdict(list)
        for c in candidates:
            digits = c['digits']
            s = sum(digits)
            sp = max(digits) - min(digits)
            sum_bucket = s // 3  # 0~9
            span_bucket = sp // 2  # 0~4
            buckets[(sum_bucket, span_bucket)].append(c)

        selected = []
        selected_tuples = set()

        # 轮换扫描所有桶
        bucket_keys = sorted(buckets.keys())
        while len(selected) < n and bucket_keys:
            # 每轮从每个桶取一个最好的
            surviving = []
            for key in bucket_keys:
                group = buckets[key]
                if not group:
                    continue
                # 取桶内最好且未选的
                group.sort(key=lambda x: -x['final_score'])
                for c in group:
                    t = tuple(c['digits'])
                    if t not in selected_tuples:
                        selected.append(c)
                        selected_tuples.add(t)
                        group.remove(c)
                        break
                if group:
                    surviving.append(key)
                if len(selected) >= n:
                    break
            bucket_keys = surviving
            if len(selected) >= n:
                break

        # 名额没满时从剩余中补
        if len(selected) < n:
            for c in candidates:
                if len(selected) >= n:
                    break
                k = tuple(c['digits'])
                if k not in selected_tuples:
                    selected.append(c)
                    selected_tuples.add(k)

        return selected[:n]

    # ================================================================
    # 复式方案
    # ================================================================

    def _generate_compound(self, top_scored: List[Dict]) -> Dict[str, Any]:
        """基于Top N的三层评分结果生成立体复式
        V2.8.1: 每个位置候选数≥3(不足时展宽到Top 50+), 防止十位单点炸穿
        """
        compound = {}

        # 从Top30开始, 如某位置候选不足3个则展宽
        pool_size = 30
        top_n = top_scored[:pool_size]

        for _ in range(3):  # 最多展宽3轮
            bai_cnt = Counter(d['digits'][0] for d in top_n)
            shi_cnt = Counter(d['digits'][1] for d in top_n)
            ge_cnt = Counter(d['digits'][2] for d in top_n)

            bai_pool = sorted(bai_cnt, key=lambda d: -bai_cnt[d])[:8]
            shi_pool = sorted(shi_cnt, key=lambda d: -shi_cnt[d])[:8]
            ge_pool = sorted(ge_cnt, key=lambda d: -ge_cnt[d])[:8]

            if len(bai_pool) < 3 or len(shi_pool) < 3 or len(ge_pool) < 3:
                pool_size += 20
                top_n = top_scored[:min(pool_size, len(top_scored))]
                continue
            break

        # 即使展宽后仍不足3个,则用全数字集强制补全
        def _ensure_min3(pool, pos_counts):
            if len(pool) >= 3:
                return pool
            # 从Counter中取频率最高的3个数字
            top_digits = sorted(pos_counts, key=lambda d: -pos_counts[d])
            for d in top_digits:
                if d not in pool:
                    pool.append(d)
                if len(pool) >= 3:
                    break
            return pool

        bai_pool = _ensure_min3(list(bai_pool), bai_cnt)
        shi_pool = _ensure_min3(list(shi_pool), shi_cnt)
        ge_pool = _ensure_min3(list(ge_pool), ge_cnt)

        compound['直选复式'] = {
            'bai': sorted(bai_pool),
            'shi': sorted(shi_pool),
            'ge': sorted(ge_pool),
            'bets': f"{len(bai_pool)}×{len(shi_pool)}×{len(ge_pool)}="
                    f"{len(bai_pool)*len(shi_pool)*len(ge_pool)}注",
        }

        all_digits = Counter()
        for bet in top_n:
            for d in bet['digits']:
                all_digits[d] += 1
        pool = sorted(all_digits, key=lambda d: -all_digits[d])[:8]
        if len(pool) > 3:
            compound['组选复式'] = {
                'numbers': sorted(pool),
                'count': len(pool),
                'desc': f"从{len(pool)}个数字中选3个的组合",
            }

        return compound

    # ================================================================
    # 回测（V1.3.1: 完全复用 _compute_layers，消除代码重复和评分偏差）
    # ================================================================

    def _backtest_build_score_tables(self, seq: List[int], prev_val: int) -> Dict:
        """
        回测用：构建单个位置的L1评分表
        与 _build_score_tables 逻辑一致（但基于滚动训练集）
        返回: (score_dict, trans1, trans2, miss_dict)
        """
        miss = {}
        for i in range(len(seq) - 1, -1, -1):
            d = seq[i]
            if d not in miss:
                miss[d] = len(seq) - 1 - i
        for d in range(10):
            if d not in miss:
                miss[d] = len(seq)

        total = len(seq)

        # 指数衰减频率（与 predict 路径一致）
        import math as _m
        half_life = 20.0
        lam = _m.log(2) / half_life
        freq = {d: 0.01 for d in range(10)}
        total_w = 0.0
        for i in range(max(0, len(seq) - 200), len(seq)):
            age = len(seq) - 1 - i
            w = _m.exp(-lam * age)
            freq[seq[i]] += w
            total_w += w
        if total_w > 0:
            for d in range(10):
                freq[d] /= total_w

        full_cnt = Counter(seq)

        # 贝叶斯收缩系数
        def _ba(tc): return 10.0 / (1.0 + tc / 20.0)

        # 一阶转移（贝叶斯收缩）
        trans1 = [[0] * 10 for _ in range(10)]
        cnt1 = [[0] * 10 for _ in range(10)]
        for i in range(len(seq) - 1):
            cnt1[seq[i]][seq[i + 1]] += 1
        for p in range(10):
            tc = sum(cnt1[p])
            alpha = _ba(tc)
            for c in range(10):
                trans1[p][c] = (cnt1[p][c] + alpha) / (tc + alpha * 10)

        # 二阶转移（贝叶斯收缩）
        trans2 = [[0] * 10 for _ in range(100)]
        cnt2 = [[0] * 10 for _ in range(100)]
        for i in range(len(seq) - 2):
            p1, p2, c2 = seq[i], seq[i + 1], seq[i + 2]
            cnt2[p1 * 10 + p2][c2] += 1
        for idx in range(100):
            tc = sum(cnt2[idx])
            alpha = _ba(tc)
            for c2 in range(10):
                trans2[idx][c2] = (cnt2[idx][c2] + alpha) / (tc + alpha * 10)

        # 三阶转移 1000×10（贝叶斯收缩）
        trans3 = [[0] * 10 for _ in range(1000)]
        cnt3 = [[0] * 10 for _ in range(1000)]
        for i in range(len(seq) - 3):
            p1, p2, p3, c3 = seq[i], seq[i + 1], seq[i + 2], seq[i + 3]
            cnt3[p1 * 100 + p2 * 10 + p3][c3] += 1
        for idx in range(1000):
            tc = sum(cnt3[idx])
            alpha = _ba(tc)
            for c3 in range(10):
                trans3[idx][c3] = (cnt3[idx][c3] + alpha) / (tc + alpha * 10)

        from modules.p3_poisson_model import P3PoissonModel
        poisson_fn = P3PoissonModel.poisson_anomaly

        # 融合分 = freq*0.5 + poisson*0.2（trans部分在调用方 + trans*0.3）
        score = {}
        for d in range(10):
            fs = freq.get(d, 0.01)
            p_d = full_cnt.get(d, 0) / max(total, 1)
            ps = poisson_fn(miss.get(d, total), p_d) * 0.2
            score[d] = fs * 0.5 + ps
        return score, trans1, trans2, trans3, miss

    def train_irl(self, n_periods: int = 50, verbose: bool = True):
        """【3】独立IRL训练 — 不触发预测, 模拟回放历史数据更新RL权重"""
        if len(self.draws) < n_periods + 10:
            if verbose:
                print(f"[P3-IRL] ⚠️ 数据不足 (需要{n_periods+10}, 现有{len(self.draws)})")
            return

        trained = self.draws[-n_periods:]
        history = self.draws[:-n_periods]

        for i in range(n_periods):
            # 模拟predict: 使用历史数据, 传入actual_draw触发_meta_learn_update
            sim = Pick3FusionComplete.__new__(Pick3FusionComplete)
            sim.draws = history + trained[:i] if i > 0 else history
            if len(sim.draws) < 10:
                continue
            sim.prev_draw = list(sim.draws[-1])
            sim._init_modules(sim.draws)
            sim._build_position_stats(sim.draws)
            from modules.p3_pattern_recognizer import P3PatternRecognizer
            sim.recognizer = P3PatternRecognizer(sim.draws)
            sim.recognizer.build_distributions()
            sim._compute_pair_freq(sim.draws)
            sim._multi_period_mc(sim.prev_draw, 100)
            sim._z3_group_miss = getattr(self, '_z3_group_miss', {})
            # NN/RF模型已移除(在随机数据上无效)
            from p3_game_theory import P3GameTheoryAnalyzer as _P3GT
            try:
                self._game_theory_test = _P3GT(sim.draws)
            except:
                pass
            try:
                sim._game_theory = _P3GT(sim.draws)
            except:
                sim._game_theory = self.game_theory

            actual = list(trained[i])
            try:
                sim.predict(_silent=True, actual_draw=actual)
            except Exception:
                pass

            if verbose and (i+1) % 10 == 0:
                print(f"[P3-IRL] 🏋️ 训练进度: {i+1}/{n_periods}")

        # 从sim中获取更新后的rl_weights
        try:
            self._rl_weights = sim._rl_weights
        except Exception:
            pass

        if verbose:
            print(f"[P3-IRL] ✅ 【3】IRL训练完成 ({n_periods}期, rl_weights={self._rl_weights})")

    def update_from_actual(self, period: str, actual_draw: List[int]) -> Dict[str, Any]:
        """
        P5: IRL在线更新接口 — 外部喂入实际开奖结果，更新预测权重和IRL反馈
        参数:
            period: 期号字符串(如"26167")
            actual_draw: 实际开奖 [百位, 十位, 个位]
        返回:
            更新后的权重和命中统计
        """
        if len(actual_draw) != 3:
            return {'error': 'actual_draw必须是3元素列表'}

        # 更新draws
        self.draws.append(tuple(actual_draw))
        self.prev_draw = actual_draw
        self.last_period = period

        # 调用IRL更新
        irl_result = self._online_irl_update()

        # 清除mc_cache，下次predict重新计算
        self._mc_cache = None

        print(f"[P3-IRL] ✅ 已更新期号{period} 开奖{actual_draw[0]}{actual_draw[1]}{actual_draw[2]}")

        return {
            'period': period,
            'actual': actual_draw,
            'rl_weights': self._rl_weights,
            'total_draws': len(self.draws),
        }

    def _online_irl_update(self):
        """【P3-5】开奖后自动IRL更新 — 检测新数据并触发_meta_learn_update"""
        try:
            from p3_data_updater import check_and_update
            result = check_and_update()
            if result.get('updated') and result.get('new_count', 0) > 0:
                # 有新开奖数据，重新加载draws
                from p3_data_updater import get_last_period, get_total
                last = get_last_period()
                # 更新数据
                old_n = len(self.draws)
                self.draws = load_data(self.data_path)
                new_n = len(self.draws)
                if new_n > old_n and last:
                    # 获取最新的开奖
                    actual = list(self.draws[-1])
                    self.prev_draw = list(self.draws[-1])
                    self.last_period = str(int(self._get_last_period()) + 1)
                    print(f"[P3-IRL] \U0001f504 【5】新数据({actual}), 触发IRL更新")
                    # 重新构建位置统计
                    self._build_position_stats(self.draws)
                    try:
                        self.pattern_recognizer = P3PatternRecognizer(self.draws)
                        self.pattern_recognizer.build_distributions()
                    except Exception:
                        pass
        except Exception as e:
            print(f"[P3-IRL] \u26a0\ufe0f 【5】IRL更新跳过: {e}")

    def _estimate_confidence(self, zx_all):
        """【P3-4】不确定性估计 — 基于Top3分数的gap"""
        if len(zx_all) < 3:
            return 1.0
        top3 = zx_all[:3]
        scores = [c.get('final_score', 0) for c in top3]
        gap = scores[0] - scores[2] if len(scores) >= 3 else 1.0
        # gap < 0.05 → 低置信度, gap > 0.15 → 高置信度
        confidence = min(max((gap - 0.05) / 0.10, 0.0), 1.0)
        return confidence

    def _ensure_mi_weights(self):
        """确保MI权重已计算(首次predict/backtest/benchmark触发)"""
        if not hasattr(self, '_mi_weights') or self._mi_weights is None:
            print("\n  [P3-MI] 置换检验校准权重...")
            self._mi_weights = self._compute_mi_adjusted_weights()

    def backtest(self, n_periods: int = 50) -> Dict[str, Any]:
        """
        回测：滚动训练集 → 构建类似 _build_score_tables 的结构
        """
        if len(self.draws) < n_periods + 10:
            n_periods = max(len(self.draws) - 10, 10)
        # 确保MI权重已计算(与predict一致)
        self._ensure_mi_weights()

        total = 0
        zx_hits = 0
        z3_hits = 0
        z6_hits = 0
        any_hit = 0
        details = []

        print(f"[P3-Backtest] 回测 {n_periods} 期...")

        for idx in range(len(self.draws) - n_periods - 1, len(self.draws) - 1):
            train_draws = self.draws[:idx + 1]
            actual = list(self.draws[idx + 1])
            actual_sorted = sorted(actual)
            prev_draw = list(train_draws[-1])

            bai_s = [d[0] for d in train_draws]
            shi_s = [d[1] for d in train_draws]
            ge_s = [d[2] for d in train_draws]

            l1_bai, t1_bai, t2_bai, t3_bai, _ = self._backtest_build_score_tables(bai_s, prev_draw[0])
            l1_shi, t1_shi, t2_shi, t3_shi, _ = self._backtest_build_score_tables(shi_s, prev_draw[1])
            l1_ge, t1_ge, t2_ge, t3_ge, _ = self._backtest_build_score_tables(ge_s, prev_draw[2])

            # 融入 trans（同 P3PoissonModel.get_trans_score alpha=0.35）
            _a = 0.35
            l1_tables = {}
            # bai: 只用一阶
            scores_bai = {}
            for d in range(10):
                base = l1_bai.get(d, 0.01)
                p1 = t1_bai[prev_draw[0]][d]
                scores_bai[d] = base + p1 * 0.3
            l1_tables['bai'] = scores_bai
            # shi: 一阶+二阶
            scores_shi = {}
            for d in range(10):
                base = l1_shi.get(d, 0.01)
                p1 = t1_shi[prev_draw[1]][d]
                p2 = t2_shi[prev_draw[0] * 10 + prev_draw[1]][d]
                trans_s = (1 - _a) * p1 + _a * p2
                scores_shi[d] = base + trans_s * 0.3
            l1_tables['shi'] = scores_shi
            # ge: 一阶+二阶+三阶
            scores_ge = {}
            for d in range(10):
                base = l1_ge.get(d, 0.01)
                p1 = t1_ge[prev_draw[2]][d]
                p2 = t2_ge[prev_draw[1] * 10 + prev_draw[2]][d]
                p3 = t3_ge[prev_draw[0] * 100 + prev_draw[1] * 10 + prev_draw[2]][d]
                trans_s = (1 - _a) * p1 + (_a * 0.5) * p2 + (_a * 0.5) * p3
                scores_ge[d] = base + trans_s * 0.3
            l1_tables['ge'] = scores_ge

            # 构建临时模式识别器（用训练数据，避免前视偏差）
            bt_recognizer = P3PatternRecognizer(train_draws)
            # 训练集位置对概率（避免前视偏差）
            _bt_pair_bs = Counter((d[0], d[1]) for d in train_draws)
            _bt_pair_bg = Counter((d[0], d[2]) for d in train_draws)
            _bt_pair_sg = Counter((d[1], d[2]) for d in train_draws)
            _bt_total = len(train_draws)
            bt_pair_bs = {k: v / _bt_total for k, v in _bt_pair_bs.items()}
            bt_pair_bg = {k: v / _bt_total for k, v in _bt_pair_bg.items()}
            bt_pair_sg = {k: v / _bt_total for k, v in _bt_pair_sg.items()}
            # 临时替换 self 的 pair probs（避免改每个调用点）
            _saved_pairs = (self._pair_prob_bs, self._pair_prob_bg, self._pair_prob_sg)
            self._pair_prob_bs, self._pair_prob_bg, self._pair_prob_sg = bt_pair_bs, bt_pair_bg, bt_pair_sg

            # ZX 全枚举 + _compute_layers（与 predict 完全一致）
            w1_zx, w2_zx, w3_zx, w4_zx = self._get_group_weights('zx')
            zx_all = []
            l1s, l2s, l3s, gts = [], [], [], []
            for bai in range(10):
                for shi in range(10):
                    for ge in range(10):
                        digits = [bai, shi, ge]
                        ps = bt_recognizer.score_pattern(digits, prev_draw)
                        ss = P3MathFilter.smart_score(digits)
                        gt = self._game_theory.expected_value(digits)
                        l1, l2, l3 = self._compute_layers(bai, shi, ge, prev_draw, l1_tables, ps, ss)
                        cs = self._cross_seq_score(digits, prev_draw)
                        zx_all.append({'digits': digits, 'final_score': 0,
                                       'l1': l1, 'l2': l2, 'l3': l3, 'gt': gt, 'cross_seq': cs})
                        l1s.append(l1)
                        l2s.append(l2)
                        l3s.append(l3)
                        gts.append(gt)

            l1n = self._normalize(l1s)
            l2n = self._normalize(l2s)
            l3n = self._normalize(l3s)
            gtn = self._normalize(gts)

            for i, c in enumerate(zx_all):
                c['final_score'] = l1n[i]*w1_zx + l2n[i]*w2_zx + l3n[i]*w3_zx + gtn[i]*w4_zx
            zx_all.sort(key=lambda x: -x['final_score'])
            # 方案D: K-Medoids覆盖选择
            top5 = self._kmedoids_cover(zx_all, 10)

            w1_z3, w2_z3, w3_z3, w4_z3 = self._get_group_weights('z3')
            w1_z6, w2_z6, w3_z6, w4_z6 = self._get_group_weights('z6')

            # 训练集数字总频率（用于Z3数字对评分，避免前视偏差）
            bt_digit_cnt = Counter(b for d in train_draws for b in d)
            bt_total = sum(bt_digit_cnt.values())
            bt_z3_freqs = {d: bt_digit_cnt.get(d, 0) / max(bt_total, 1) for d in range(10)}
            # Z3组级遗漏（训练集内计算）
            from modules.p3_poisson_model import P3PoissonModel as _PPM
            bt_z3_miss = {}
            for i in range(len(train_draws) - 1, -1, -1):
                td = train_draws[i]
                if len(set(td)) == 2:
                    if td[0] == td[1]: _pd, _sd = td[0], td[2]
                    elif td[0] == td[2]: _pd, _sd = td[0], td[1]
                    else: _pd, _sd = td[1], td[0]
                    _k = (_pd, _sd)
                    if _k not in bt_z3_miss:
                        bt_z3_miss[_k] = len(train_draws) - 1 - i
            for _pd in range(10):
                for _sd in range(10):
                    if _pd != _sd and (_pd, _sd) not in bt_z3_miss:
                        bt_z3_miss[(_pd, _sd)] = len(train_draws)

            # Z3（归一化后加权，与 predict 一致）
            z3_candidates = {}
            for d in range(10):
                for diff in range(10):
                    if diff == d:
                        continue
                    for perm in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
                        b = d if perm[0] == 0 else diff
                        s2 = d if perm[1] == 0 else diff
                        g = d if perm[2] == 0 else diff
                        digits = [b, s2, g]
                        ps = bt_recognizer.score_pattern(digits, prev_draw)
                        ss = P3MathFilter.smart_score(digits)
                        gt = self._game_theory.expected_value(digits)
                        # Z3组级遗漏泊松分
                        _miss = bt_z3_miss.get((d, diff), len(train_draws))
                        _z3_prob = (2 * bt_z3_freqs.get(d, 0.1) + bt_z3_freqs.get(diff, 0.1)) / 3.0
                        _miss_score = _PPM.poisson_anomaly(_miss, _z3_prob)
                        l1, l2, l3 = self._compute_layers(b, s2, g, prev_draw, l1_tables, ps, ss, is_z3=True, z3_freqs=bt_z3_freqs, z3_miss_score=_miss_score)
                        key = tuple(sorted([b, s2, g]))
                        if key not in z3_candidates or (l1*w1_z3+l2*w2_z3+l3*w3_z3+gt*w4_z3) > z3_candidates[key][4]:
                            z3_candidates[key] = (l1, l2, l3, gt, l1*w1_z3+l2*w2_z3+l3*w3_z3+gt*w4_z3)
            z3_keys = list(z3_candidates.keys())
            z3_l1s = [z3_candidates[k][0] for k in z3_keys]
            z3_l2s = [z3_candidates[k][1] for k in z3_keys]
            z3_l3s = [z3_candidates[k][2] for k in z3_keys]
            z3_gts = [z3_candidates[k][3] for k in z3_keys]
            z3_l1n = self._normalize(z3_l1s)
            z3_l2n = self._normalize(z3_l2s)
            z3_l3n = self._normalize(z3_l3s)
            z3_gtn = self._normalize(z3_gts)
            z3_scores = {k: z3_l1n[i]*w1_z3+z3_l2n[i]*w2_z3+z3_l3n[i]*w3_z3+z3_gtn[i]*w4_z3
                         for i, k in enumerate(z3_keys)}
            top_z3 = [list(k) for k in sorted(z3_scores, key=lambda k: -z3_scores[k])[:10]]

            # Z6（归一化后加权）
            z6_candidates = {}
            perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
            for d1 in range(10):
                for d2 in range(d1 + 1, 10):
                    for d3 in range(d2 + 1, 10):
                        for p in perms:
                            b = d1 if p[0] == 0 else d2 if p[0] == 1 else d3
                            s2 = d1 if p[1] == 0 else d2 if p[1] == 1 else d3
                            g = d1 if p[2] == 0 else d2 if p[2] == 1 else d3
                            digits = [b, s2, g]
                            ps = bt_recognizer.score_pattern(digits, prev_draw)
                            ss = P3MathFilter.smart_score(digits)
                            gt = self._game_theory.expected_value(digits)
                            l1, l2, l3 = self._compute_layers(b, s2, g, prev_draw, l1_tables, ps, ss)
                            key = (d1, d2, d3)
                            if key not in z6_candidates or (l1*w1_z6+l2*w2_z6+l3*w3_z6+gt*w4_z6) > z6_candidates[key][4]:
                                z6_candidates[key] = (l1, l2, l3, gt, l1*w1_z6+l2*w2_z6+l3*w3_z6+gt*w4_z6)
            z6_keys = list(z6_candidates.keys())
            z6_l1s = [z6_candidates[k][0] for k in z6_keys]
            z6_l2s = [z6_candidates[k][1] for k in z6_keys]
            z6_l3s = [z6_candidates[k][2] for k in z6_keys]
            z6_gts = [z6_candidates[k][3] for k in z6_keys]
            z6_l1n = self._normalize(z6_l1s)
            z6_l2n = self._normalize(z6_l2s)
            z6_l3n = self._normalize(z6_l3s)
            z6_gtn = self._normalize(z6_gts)
            z6_scores = {k: z6_l1n[i]*w1_z6+z6_l2n[i]*w2_z6+z6_l3n[i]*w3_z6+z6_gtn[i]*w4_z6
                         for i, k in enumerate(z6_keys)}
            top_z6 = [list(k) for k in sorted(z6_scores, key=lambda k: -z6_scores[k])[:8]]
            # 恢复原始 pair probs
            self._pair_prob_bs, self._pair_prob_bg, self._pair_prob_sg = _saved_pairs

            zx_match = any(c['digits'] == actual for c in top5)
            z3_match = any(sorted(c) == actual_sorted for c in top_z3)
            z6_match = any(sorted(c) == actual_sorted for c in top_z6)

            period_num = self._get_last_period_for_backtest(idx + 1)

            if zx_match:
                zx_hits += 1
            if z3_match:
                z3_hits += 1
            if z6_match:
                z6_hits += 1
            if zx_match or z3_match or z6_match:
                any_hit += 1
            total += 1

            if total % 10 == 0:
                print(f"[P3-Backtest] 进度 {total}/{n_periods} ...")

            details.append({
                'period': period_num,
                'actual': actual,
                'zx_match': zx_match,
                'z3_match': z3_match,
                'z6_match': z6_match,
                'any_match': any_hit,
            })

        print(f"[P3-Backtest] 完成: ZX={zx_hits} Z3={z3_hits} Z6={z6_hits} any={any_hit}")

        return {
            'total_periods': total,
            'zx_hits': zx_hits,
            'z3_hits': z3_hits,
            'z6_hits': z6_hits,
            'zx_hit_rate': round(zx_hits / total * 100, 2) if total else 0,
            'z3_hit_rate': round(z3_hits / total * 100, 2) if total else 0,
            'z6_hit_rate': round(z6_hits / total * 100, 2) if total else 0,
            'overall_hit_rate': round(any_hit / total * 100, 2) if total else 0,
            'details': details[-20:],
        }

    def _get_last_period_for_backtest(self, idx: int) -> str:
        try:
            if self._df_periods is None:
                df = pd.read_excel(self.data_path, engine='openpyxl')
                self._df_periods = df['期号'].tolist()
            return str(int(self._df_periods[idx]))
        except Exception:
            return "?"

    def benchmark(self, n_periods: int = 100) -> Dict[str, Any]:
        """
        基准对比: 模型(Top10 ZX) vs 纯随机(Top10 ZX)
        对比精确命中率和和值±2命中率
        """
        if len(self.draws) < n_periods + 10:
            return {'error': f'数据不足({len(self.draws)}期)'}

        import random as rnd
        rnd.seed(42)
        self._ensure_mi_weights()

        model_exact = 0
        random_exact = 0
        model_sum = 0
        random_sum = 0

        for idx in range(len(self.draws) - n_periods - 1, len(self.draws) - 1):
            train_draws = self.draws[:idx + 1]
            actual = list(self.draws[idx + 1])
            actual_sum = sum(actual)
            prev_draw = list(train_draws[-1])

            # 模型预测 (直接用缓存枚举或快速枚举)
            zx_all = self._enumerate_direct(prev_draw, mc_counts={})
            zx_all.sort(key=lambda x: x['final_score'], reverse=True)
            model_top10 = [c['digits'] for c in zx_all[:10]]

            if actual in model_top10:
                model_exact += 1
            if any(abs(sum(c) - actual_sum) <= 2 for c in model_top10):
                model_sum += 1

            # 随机基线
            rnd.seed(idx)
            random_top10 = []
            for _ in range(10):
                random_top10.append([rnd.randint(0, 9) for _ in range(3)])

            if actual in random_top10:
                random_exact += 1
            if any(abs(sum(c) - actual_sum) <= 2 for c in random_top10):
                random_sum += 1

        m_exact_r = model_exact / n_periods * 100
        r_exact_r = random_exact / n_periods * 100
        m_sum_r = model_sum / n_periods * 100
        r_sum_r = random_sum / n_periods * 100

        # Wilson score 95% CI
        def _wilson_ci(p, n, z=1.96):
            if n == 0:
                return 0, 0
            p = p / n if isinstance(p, int) else p
            denom = 1 + z**2/n
            centre = (p + z**2/(2*n)) / denom
            margin = z * (p*(1-p)/n + z**2/(4*n**2))**0.5 / denom
            return centre - margin, centre + margin

        m_ci = _wilson_ci(m_sum_r/100, n_periods)
        r_ci = _wilson_ci(r_sum_r/100, n_periods)

        # 卡方检验(和值±2)
        from scipy.stats import chi2_contingency
        table = [[model_sum, n_periods - model_sum],
                 [random_sum, n_periods - random_sum]]
        chi2, p_value = chi2_contingency(table, correction=True)[:2]

        delta = m_sum_r - r_sum_r
        delta_se = (m_sum_r*(100-m_sum_r)/n_periods + r_sum_r*(100-r_sum_r)/n_periods)**0.5
        delta_ci = (delta - 1.96*delta_se, delta + 1.96*delta_se)

        print(f"\n{'='*55}")
        print(f"  排列3 基准对比 ({n_periods}期, Top10直选, 和值±2)")
        print(f"{'='*55}")
        print(f"  模型  精确命中: {model_exact}/{n_periods} = {m_exact_r:.2f}%")
        print(f"  模型  和值±2:   {model_sum}/{n_periods} = {m_sum_r:.1f}%")
        print(f"        95%CI: [{m_ci[0]*100:.1f}%, {m_ci[1]*100:.1f}%]")
        print(f"  随机  精确命中: {random_exact}/{n_periods} = {r_exact_r:.2f}%")
        print(f"  随机  和值±2:   {random_sum}/{n_periods} = {r_sum_r:.1f}%")
        print(f"        95%CI: [{r_ci[0]*100:.1f}%, {r_ci[1]*100:.1f}%]")
        print(f"  差值  Δ={delta:+.1f}%  95%CI=[{delta_ci[0]:.1f}%, {delta_ci[1]:.1f}%]")
        print(f"  卡方检验: χ²={chi2:.3f}, p={p_value:.4f}")

        if p_value < 0.05:
            if delta > 0:
                print(f"  ✅ 模型显著优于随机")
            else:
                print(f"  ❌ 随机显著优于模型")
        else:
            print(f"  ⚠️ 模型与随机无显著差异 (p={p_value:.4f})")

        return {
            'n_periods': n_periods,
            'model_exact_%': round(m_exact_r, 2),
            'random_exact_%': round(r_exact_r, 2),
            'model_sum_match_%': round(m_sum_r, 2),
            'random_sum_match_%': round(r_sum_r, 2),
            'delta_%': round(delta, 2),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05,
        }

    def info(self) -> Dict[str, Any]:
        stats = {
            'sum_stats': self.stats.sum_stats(),
            'span_stats': self.stats.span_stats(),
            'parity_stats': self.stats.parity_stats(),
            'group_type_stats': self.stats.group_type_stats(),
        }
        return {
            'skill': '排列3 (Pick3) 全量枚举+三层评分',
            'version': VERSION,
            'release_date': RELEASE_DATE,
            'data_periods': len(self.draws),
            'last_draw': self.prev_draw,
            'stats_overview': stats,
        }
