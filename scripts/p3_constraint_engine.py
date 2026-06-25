#!/usr/bin/env python3
"""
排列3 统一约束引擎 (P3ConstraintEngine)

约束分类:
1. 硬约束 — 必须满足（合法号码范围）
2. 策略约束 — 策略特定条件
3. 软约束 — 评分扣分项
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

DIGIT_RANGE = (0, 9)
DIGIT_COUNT = 3
PRIME_DIGITS = {2, 3, 5, 7}


@dataclass
class P3HardConstraints:
    """硬约束配置"""
    sum_range: Tuple[int, int] = (1, 26)      # 和值范围
    span_range: Tuple[int, int] = (0, 9)      # 跨度范围
    group_types: List[str] = None              # 允许的组类型

    def __post_init__(self):
        if self.group_types is None:
            self.group_types = ['组三', '组六', '豹子']


@dataclass
class P3SoftConstraints:
    """软约束配置 — FIX P2：删除无区分度的全偏好"""
    sum_prefer: Tuple[int, int] = (8, 19)       # 偏好和值区间
    parity_prefer: List[str] = None             # 偏好奇偶比（仅高频比给高分）
    span_prefer: Tuple[int, int] = (2, 8)       # 偏好跨度区间
    route_012_prefer: List[str] = None           # 偏好012路形态

    def __post_init__(self):
        if self.parity_prefer is None:
            # 仅2:1和1:2为高频（组六约70%），3:0和0:3低频
            self.parity_prefer = ['2:1', '1:2']
        if self.route_012_prefer is None:
            # 保留最均衡的3种形态，其他形态降分
            self.route_012_prefer = ['1:1:1', '2:1:0', '2:0:1', '1:2:0', '1:0:2']


class P3ConstraintEngine:
    """排列3约束引擎"""

    def __init__(self, hard: Optional[P3HardConstraints] = None,
                 soft: Optional[P3SoftConstraints] = None):
        self.hard = hard or P3HardConstraints()
        self.soft = soft or P3SoftConstraints()

    def validate_hard(self, digits: List[int]) -> bool:
        """硬约束验证"""
        if len(digits) != 3:
            return False
        if not all(DIGIT_RANGE[0] <= d <= DIGIT_RANGE[1] for d in digits):
            return False

        s = sum(digits)
        if not (self.hard.sum_range[0] <= s <= self.hard.sum_range[1]):
            return False

        sp = max(digits) - min(digits)
        if not (self.hard.span_range[0] <= sp <= self.hard.span_range[1]):
            return False

        # 组类型验证
        if self.hard.group_types:
            gtype = {1: '豹子', 2: '组三', 3: '组六'}[len(set(digits))]
            if gtype not in self.hard.group_types:
                return False

        return True

    def score_soft(self, digits: List[int]) -> float:
        """
        软约束评分 (0.0 ~ 1.0，越高越好)
        """
        scores = []
        s = sum(digits)
        sp = max(digits) - min(digits)

        # 和值偏好
        if self.soft.sum_prefer[0] <= s <= self.soft.sum_prefer[1]:
            scores.append(1.0)
        else:
            # 偏离越远分越低
            center = (self.soft.sum_prefer[0] + self.soft.sum_prefer[1]) / 2
            dist = abs(s - center)
            scores.append(max(0, 1.0 - dist * 0.08))

        # 跨度偏好
        if self.soft.span_prefer[0] <= sp <= self.soft.span_prefer[1]:
            scores.append(1.0)
        else:
            center = (self.soft.span_prefer[0] + self.soft.span_prefer[1]) / 2
            dist = abs(sp - center)
            scores.append(max(0, 1.0 - dist * 0.1))

        # 【FIX P2】奇偶比偏好（仅高频2:1/1:2给1.0，极端比降分）
        odd_cnt = sum(1 for d in digits if d % 2 == 1)
        parity_key = f"{odd_cnt}:{3-odd_cnt}"
        if parity_key in ['2:1', '1:2']:
            scores.append(1.0)
        elif parity_key == '3:0':
            scores.append(0.5)  # 全奇降分
        elif parity_key == '0:3':
            scores.append(0.4)  # 全偶降得更低

        # 【FIX P2】012路形态偏好（极端分布降分）
        r0 = sum(1 for d in digits if d % 3 == 0)
        r1 = sum(1 for d in digits if d % 3 == 1)
        r2 = sum(1 for d in digits if d % 3 == 2)
        route_key = f"{r0}:{r1}:{r2}"
        if route_key in ['1:1:1', '2:1:0', '2:0:1', '1:2:0', '1:0:2']:
            scores.append(1.0)
        elif route_key in ['3:0:0', '0:3:0', '0:0:3']:
            scores.append(0.2)  # 全同路极罕见
        else:
            scores.append(0.6)  # 其他2:0:1 变体

        if not scores:
            return 0.5
        return sum(scores) / len(scores)

    def validate_and_score(self, digits: List[int]) -> Tuple[bool, float]:
        """组合验证+评分"""
        if not self.validate_hard(digits):
            return False, 0.0
        return True, self.score_soft(digits)


# 预配置的策略
STRATEGY_CONFIGS = {
    1: {  # 均衡策略
        'name': '均衡策略',
        'hard_sum': (3, 24),
        'soft_sum': (9, 18),
        'soft_spans': (3, 7),
        'parity_prefer': ['2:1', '1:2'],
        'group_types': ['组三', '组六'],
    },
    2: {  # 热号追踪策略
        'name': '热号追踪策略',
        'hard_sum': (1, 26),
        'soft_sum': (8, 19),
        'soft_spans': (2, 8),
        'parity_prefer': ['2:1', '1:2', '3:0', '0:3'],
        'group_types': ['组三', '组六', '豹子'],
    },
    3: {  # 冷号补缺策略
        'name': '冷号补缺策略',
        'hard_sum': (1, 26),
        'soft_sum': (10, 17),
        'soft_spans': (4, 9),
        'parity_prefer': ['1:2', '2:1', '3:0'],
        'group_types': ['组三', '组六'],
    },
}
