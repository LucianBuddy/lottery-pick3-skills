#!/usr/bin/env python3
"""
排列3 (Pick3) 历史数据自动更新器 v1
数据源: 体彩数据API (webapi.sporttery.cn)
"""

import os
import sys
import json
import re
import urllib.request
import ssl
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np

SKILL_DIR = Path(__file__).resolve().parent
DATA_PATH = SKILL_DIR.parent / 'assets' / 'data' / '排列3历史数据.xlsx'

# 体彩数据API — gameNo=35 是排列3
API_URL = 'https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry'

# 500彩票网数据源（备选）
API_URL_500COM = 'https://datachart.500.com/pls/history/newinc/history.php'


def _build_url(page_no: int = 1, page_size: int = 30) -> str:
    return (f'{API_URL}?gameNo=35&provinceId=0&pageSize={page_size}'
            f'&isPc=true&pageNo={page_no}')


def _http_get(url: str, timeout: int = 15, referer: str = 'https://www.lottery.gov.cn/') -> str:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
        'Accept': 'application/json, text/plain, */*',
        'Referer': referer,
    })
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return resp.read().decode('utf-8', errors='replace')


def fetch_draws_from_api(page_no: int = 1, page_size: int = 50) -> List[dict]:
    """
    从体彩数据API获取排列3开奖数据
    Returns: [{'期号': int, '百位': int, '十位': int, '个位': int}, ...]
    """
    url = _build_url(page_no, page_size)
    try:
        raw = _http_get(url)
        data = json.loads(raw)
        if not data.get('success'):
            print(f"[P3-Updater] API返回失败: {data.get('errorMessage', '未知错误')}")
            return []
        draw_list = data.get('value', {}).get('list', [])
        results = []
        for draw in draw_list:
            draw_num = draw.get('lotteryDrawNum', '')
            draw_result = draw.get('lotteryDrawResult', '')
            if not draw_num or not draw_result:
                continue
            # 排列3格式: "1 2 3" 或 "1,2,3"
            nums = re.split(r'[\s,]+', draw_result.strip())
            if len(nums) != 3:
                continue
            try:
                period = int(draw_num)
                results.append({
                    '期号': period,
                    '百位': int(nums[0]),
                    '十位': int(nums[1]),
                    '个位': int(nums[2]),
                })
            except (ValueError, IndexError):
                continue
        return results
    except Exception as e:
        print(f"[P3-Updater] API请求失败: {e}")
        return []


# ——— 500彩票网数据源（备选） ———

def _http_get_500com(timeout: int = 20) -> str:
    url = f'{API_URL_500COM}?start=00001&end=99999'  # 排列3 500彩票网数据
    return _http_get(url, timeout=timeout, referer='https://www.500.com/')


def parse_500com_html(html: str) -> List[dict]:
    """解析500彩票网HTML表格，提取排列3数据"""
    clean = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    results = []
    row_pattern = re.compile(r'<tr\s+class="t_tr1">(.*?)</tr>', re.DOTALL)
    td_pattern = re.compile(r'<td[^>]*>(\d+)</td>')
    for row in row_pattern.finditer(clean):
        cells = td_pattern.findall(row.group(1))
        if len(cells) < 4:
            continue
        try:
            period = int(cells[0])
            bai = int(cells[1])
            shi = int(cells[2])
            ge = int(cells[3])
            if not all(0 <= n <= 9 for n in [bai, shi, ge]):
                continue
            results.append({
                '期号': period,
                '百位': bai,
                '十位': shi,
                '个位': ge,
            })
        except (ValueError, IndexError):
            continue
    if results:
        results.sort(key=lambda x: x['期号'])
        seen = set()
        unique = []
        for d in results:
            pid = d['期号']
            if pid not in seen:
                seen.add(pid)
                unique.append(d)
        results = unique
    return results


def fetch_draws_from_500com() -> List[dict]:
    try:
        html = _http_get_500com()
        draws = parse_500com_html(html)
        if draws:
            print(f"[P3-Updater] 500彩票网获取到 {len(draws)} 期数据 "
                  f"({draws[0]['期号']}~{draws[-1]['期号']})")
        return draws
    except Exception as e:
        print(f"[P3-Updater] 500彩票网请求失败: {e}")
        return []


# ——— 主要获取逻辑 ———

def fetch_new_draws(last_period: int) -> List[dict]:
    """从体彩数据API获取更新的所有期号"""
    all_draws = []
    page = 1
    max_pages = 5
    while page <= max_pages:
        draws = fetch_draws_from_api(page_no=page, page_size=50)
        if not draws:
            break
        all_draws.extend(draws)
        if draws and draws[-1]['期号'] <= last_period:
            break
        page += 1
    if not all_draws:
        return []
    seen = set()
    unique = []
    for d in all_draws:
        pid = d['期号']
        if pid not in seen:
            seen.add(pid)
            unique.append(d)
    new_draws = [d for d in unique if d['期号'] > last_period]
    new_draws.sort(key=lambda x: x['期号'])
    if new_draws:
        print(f"[P3-Updater] 发现 {len(new_draws)} 期新数据: "
              f"{new_draws[0]['期号']}~{new_draws[-1]['期号']}")
    else:
        current_max = max(d['期号'] for d in unique)
        print(f"[P3-Updater] 数据已是最新 (最新期号: {current_max})")
    return new_draws


def fetch_new_500com(last_period: int) -> List[dict]:
    all_draws = fetch_draws_from_500com()
    if not all_draws:
        return []
    new_draws = [d for d in all_draws if d['期号'] > last_period]
    if new_draws:
        print(f"[P3-Updater] 500彩票网发现 {len(new_draws)} 期新数据: "
              f"{new_draws[0]['期号']}~{new_draws[-1]['期号']}")
    return new_draws


def get_last_period() -> int:
    if not DATA_PATH.exists():
        return 0
    try:
        df = pd.read_excel(str(DATA_PATH), engine='openpyxl')
        return int(df['期号'].iloc[-1])
    except Exception:
        return 0


def get_first_period() -> int:
    if not DATA_PATH.exists():
        return 0
    try:
        df = pd.read_excel(str(DATA_PATH), engine='openpyxl')
        return int(df['期号'].iloc[0])
    except Exception:
        return 0


def get_total() -> int:
    if not DATA_PATH.exists():
        return 0
    try:
        df = pd.read_excel(str(DATA_PATH), engine='openpyxl')
        return len(df)
    except Exception:
        return 0


def append_draws(new_draws: List[dict]) -> int:
    if not new_draws:
        return 0
    clean = [{k: v for k, v in d.items() if not k.startswith('_')} for d in new_draws]
    new_df = pd.DataFrame(clean)
    columns = ['期号', '百位', '十位', '个位']
    for col in columns:
        if col not in new_df.columns:
            new_df[col] = 0
    new_df = new_df[columns]

    if DATA_PATH.exists():
        existing = pd.read_excel(str(DATA_PATH), engine='openpyxl')
        for col in columns:
            if col not in existing.columns:
                existing[col] = 0
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=['期号'], keep='last')
    combined = combined.sort_values('期号').reset_index(drop=True)
    combined.to_excel(str(DATA_PATH), index=False)
    print(f"[P3-Updater] 数据文件已更新: {len(combined)} 期 "
          f"({combined['期号'].iloc[0]} ~ {combined['期号'].iloc[-1]})")
    return len(new_draws)


def check_and_update() -> dict:
    last = get_last_period()
    first = get_first_period()
    print(f"[P3-Updater] 当前数据文件: {first} ~ {last} ({get_total()}期)")

    new_draws = fetch_new_draws(last)
    source = 'sporttery'
    if not new_draws:
        print(f"[P3-Updater] ⚠️ 体彩数据API无数据，尝试500彩票网...")
        new_draws = fetch_new_500com(last)
        source = '500com'

    if not new_draws:
        return {
            'updated': False, 'new_count': 0, 'last_period': last,
            'first_period': first, 'total': get_total(),
            'new_periods': [], 'source': source,
        }

    count = append_draws(new_draws)
    new_periods = [d['期号'] for d in new_draws]
    return {
        'updated': True, 'new_count': count,
        'last_period': new_periods[-1] if new_periods else last,
        'first_period': first, 'total': get_total(),
        'new_periods': new_periods, 'source': source,
    }


def verify_latest_period() -> dict:
    draws = fetch_draws_from_api(page_no=1, page_size=1)
    if draws:
        d = draws[0]
        return {
            'period': d['期号'],
            'result': f"{d['百位']} {d['十位']} {d['个位']}",
        }
    return {}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='排列3数据更新器')
    parser.add_argument('--check', action='store_true', help='检查新数据')
    parser.add_argument('--latest', action='store_true', help='查看最新开奖')
    args = parser.parse_args()

    if args.latest:
        info = verify_latest_period()
        if info:
            print(f"最新期号: {info['period']}")
            print(f"开奖号码: {info['result']}")
        else:
            print("未获取到数据")
    elif args.check:
        last = get_last_period()
        print(f"本地最新: {last}")
        new = fetch_new_draws(last)
        if new:
            print(f"有 {len(new)} 期新数据:")
            for d in new:
                print(f"  {d['期号']}: {d['百位']} {d['十位']} {d['个位']}")
        else:
            print("没有新数据")
    else:
        result = check_and_update()
        print(json.dumps(result, ensure_ascii=False, indent=2))
