#!/usr/bin/env python3
"""
排列3 (Pick3) 彩票预测技能 — OpenClaw CLI Adapter
委托给 Pick3FusionComplete 执行实际预测，本层只做参数解析+格式化输出。
"""

import argparse
import sys
from typing import Dict, List, Any

from p3_fusion_complete import Pick3FusionComplete, VERSION


def _fmt_digits(digits: List[int]) -> str:
    return ' '.join(str(d) for d in digits)


def _print_predict(result: Dict[str, Any]) -> None:
    """格式化预测输出"""
    zx = result.get('zx_bets', [])
    z3 = result.get('z3_bets', [])
    z6 = result.get('z6_bets', [])
    compound = result.get('compound_bets', {})
    period = result.get('period', '?')

    print(f"\n{'='*55}")
    print(f"  排列3 第{period}期 多策略融合预测")
    print(f"{'='*55}")

    # 直选推荐
    if zx:
        print(f"\n🎯 直选方案 (Top {len(zx)})")
        print(f"{'-'*40}")
        for i, bet in enumerate(zx, 1):
            digits = bet.get('digits', [])
            score = bet.get('final_score', 0)
            prob = bet.get('hit_probability', 0)
            print(f"  {i:2d}. [{_fmt_digits(digits)}]  score={score:.4f}  p={prob:.1f}%")

    # 组三推荐
    if z3:
        print(f"\n📋 组三方案 (Top {len(z3)})")
        print(f"{'-'*40}")
        for i, bet in enumerate(z3, 1):
            digits = sorted(bet.get('digits', []))
            score = bet.get('final_score', 0)
            prob = bet.get('hit_probability', 0)
            print(f"  {i:2d}. [{_fmt_digits(digits)}]  score={score:.4f}  p={prob:.1f}%")

    # 组六推荐
    if z6:
        print(f"\n📋 组六方案 (Top {len(z6)})")
        print(f"{'-'*40}")
        for i, bet in enumerate(z6, 1):
            digits = sorted(bet.get('digits', []))
            score = bet.get('final_score', 0)
            prob = bet.get('hit_probability', 0)
            print(f"  {i:2d}. [{_fmt_digits(digits)}]  score={score:.4f}  p={prob:.1f}%")

    # 复式方案
    if compound:
        print(f"\n📊 复式方案")
        print(f"{'-'*40}")
        for ctype, info in compound.items():
            if ctype == '直选复式':
                bai = _fmt_digits(info['bai'])
                shi = _fmt_digits(info['shi'])
                ge = _fmt_digits(info['ge'])
                print(f"  直选复式: 百[{bai}] 十[{shi}] 个[{ge}]  {info['bets']}")
            elif ctype == '组选复式':
                nums = _fmt_digits(info['numbers'])
                print(f"  组选复式: [{nums}]  ({info['desc']})")

    # 存储预测
    try:
        from prediction_store import store_prediction
        store_prediction(
            period=period,
            zx_bets=zx,
            z3_bets=z3 if z3 else None,
            z6_bets=z6 if z6 else None,
            compound_bets=compound if compound else None,
        )
    except Exception as e:
        print(f"  [Store] ⚠️ 存储失败: {e}")

    print(f"\n⚠️  仅供参考娱乐，请理性投注！")


def _print_backtest(result: Dict[str, Any]) -> None:
    """格式化回测输出"""
    if 'error' in result:
        print(f"❌ 回测失败: {result['error']}")
        return

    print(f"\n{'='*55}")
    print(f"  排列3 回测结果 (近{result['total_periods']}期)")
    print(f"{'='*55}")
    print(f"  回测期数:    {result['total_periods']}")
    print(f"  直选命中率:  {result['zx_hit_rate']:.2f}% ({result['zx_hits']}次)")
    print(f"  组三命中率:  {result['z3_hit_rate']:.2f}% ({result['z3_hits']}次)")
    print(f"  组六命中率:  {result['z6_hit_rate']:.2f}% ({result['z6_hits']}次)")
    print(f"  综合命中率:  {result['overall_hit_rate']:.2f}%")

    # 最近详情
    details = result.get('details', [])
    if details:
        print(f"\n  最近{len(details)}期详情:")
        for d in details:
            markers = []
            if d['zx_match']:
                markers.append('🎯直')
            if d['z3_match']:
                markers.append('📋三')
            if d['z6_match']:
                markers.append('📋六')
            flag = ' '.join(markers) if markers else '❌'
            print(f"    {d['period']}: [{_fmt_digits(d['actual'])}]  {flag}")

    print()


def _print_info(info: Dict[str, Any]) -> None:
    """格式化技能信息"""
    print(f"\n{'='*55}")
    print(f"  {info['skill']}")
    print(f"{'='*55}")
    print(f"  版本:      V{info['version']}")
    print(f"  发布日期:  {info['release_date']}")
    print(f"  历史数据:  {info['data_periods']}期")
    print(f"  最新开奖:  [{_fmt_digits(info['last_draw'])}]")

    stats = info.get('stats_overview', {})
    if stats:
        ss = stats.get('sum_stats', {})
        print(f"\n  📊 和值统计: 均值={ss.get('mean','-')}  "
              f"Top5={_fmt_list(ss.get('top_5',[]))}")
        ps = stats.get('parity_stats', {})
        dist = ps.get('distribution', {})
        if dist:
            print(f"  奇偶分布:    {' | '.join(f'{k}={v}%' for k,v in sorted(dist.items()))}")
        gts = stats.get('group_type_stats', {})
        gdist = gts.get('distribution', {})
        if gdist:
            print(f"  组类型分布:  {' | '.join(f'{k}={v}%' for k,v in sorted(gdist.items()))}")

    print()


def _fmt_list(lst: List) -> str:
    return ' '.join(str(x) for x in lst)


def main():
    parser = argparse.ArgumentParser(
        description='排列3 (Pick3) 预测技能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  pick3 predict              # 预测下一期（直选+组三+组六+复式）
  pick3 predict --zx 5       # 只推荐5注直选
  pick3 backtest --periods 50  # 回测50期
  pick3 info                 # 查看技能信息
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # predict
    p_predict = subparsers.add_parser('predict', help='预测下一期排列3号码')
    p_predict.add_argument('--zx', type=int, default=10, help='直选推荐注数')
    p_predict.add_argument('--z3', type=int, default=5, help='组三推荐注数')
    p_predict.add_argument('--z6', type=int, default=5, help='组六推荐注数')
    p_predict.add_argument('--no-compound', action='store_true', help='跳过复式方案')

    # backtest
    p_bt = subparsers.add_parser('backtest', help='回测模型表现')
    p_bt.add_argument('--periods', type=int, default=50, help='回测期数')

    # info
    p_info = subparsers.add_parser('info', help='查看技能信息')

    args = parser.parse_args()

    try:
        fusion = Pick3FusionComplete(auto_update=True)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)

    if args.command == 'predict' or args.command is None:
        result = fusion.predict(
            n_zx=args.zx if hasattr(args, 'zx') else 10,
            n_z3=args.z3 if hasattr(args, 'z3') else 5,
            n_z6=args.z6 if hasattr(args, 'z6') else 5,
            include_compound=not (args.no_compound if hasattr(args, 'no_compound') else False),
        )
        _print_predict(result)

    elif args.command == 'backtest':
        bt_result = fusion.backtest(n_periods=args.periods if hasattr(args, 'periods') else 50)
        _print_backtest(bt_result)

    elif args.command == 'info':
        info = fusion.info()
        _print_info(info)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
