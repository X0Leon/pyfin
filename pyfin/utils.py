# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def format_percent(number):
    """
    数字格式化成百分比，例如0.234 -> 23.40%
    """
    if np.isnan(number):
        return '-'
    return format(number, '.2%')


def format_100p(number):
    """
    数字格式化成百分数，例如0.234 -> 23.40
    """
    if np.isnan(number):
        return '-'
    return format(number * 100, '.2f')


def format_2float(number):
    """
    数字格式化成浮点，例如0.234 -> 0.2340
    """
    if np.isnan(number):
        return '-'
    return format(number, '.2f')


def get_period_name(period):
    """
    时间周期
    """
    period = period.upper()
    periods = {
        'B': 'business day',
        'C': 'custom business day',
        'D': 'daily',
        'W': 'weekly',
        'M': 'monthly',
        'BM': 'business month end',
        'CBM': 'custom business month end',
        'MS': 'month start',
        'BMS': 'business month start',
        'CBMS': 'custom business month start',
        'Q': 'quarterly',
        'BQ': 'business quarter end',
        'QS': 'quarter start',
        'BQS': 'business quarter start',
        'Y': 'yearly',
        'A': 'yearly',
        'BA': 'business year end',
        'AS': 'year start',
        'BAS': 'business year start',
        'H': 'hourly',
        'T': 'minutely',
        'S': 'secondly',
        'L': 'milliseonds',
        'U': 'microseconds'}

    if period in periods:
        return periods[period]
    else:
        return None


def scale(val, src, dst):
    """
    将范围在src内的val等比率转换到dst的范围
    如果src超出src范围，则取上界或下界

    示例：
        scale(0, (0.0, 99.0), (-1.0, 1.0)) == -1.0
        scale(-5, (0.0, 99.0), (-1.0, 1.0)) == -1.0
    """
    if val < src[0]:
        return dst[0]
    if val > src[1]:
        return dst[1]

    return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]


def as_percent(self, digits=2):
    return as_format(self, '.%s%%' % digits)


def as_format(item, format_str='.2f'):
    """
    对pandas对象（Series或DataFrame）map一个格式化字符串
    """
    if isinstance(item, pd.Series):
        return item.map(lambda x: format(x, format_str))
    elif isinstance(item, pd.DataFrame):
        return item.applymap(lambda x: format(x, format_str))


def parse_arg(arg):
    """
    解析参数成列表
    参数
    arg: list, tuple, string或csv list ('a,b,c')其中之一
    返回
    list
    """
    if type(arg) == str:
        arg = arg.strip()
        if ',' in arg:
            arg = arg.split(',')
            arg = [x.strip() for x in arg]
        else:
            arg = [arg]

    return arg


def clean_symbol(symbol):
    return symbol


def clean_symbols(symbols):
    return symbols
