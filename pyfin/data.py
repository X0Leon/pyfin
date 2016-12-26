# -*- coding: utf-8 -*-

import datetime

import pandas as pd
import requests

import pyfin
import pyfin.utils as utils


def get(symbols, provider=None, common_dates=False, forward_fill=True,
        clean_symbols=True, column_names=None, symbol_field_sep=':',
        existing=None, **kwargs):
    """
    获取数据，返回DataFrame
    参数：
    symbols: list, string, csv string
    provider (function): 下载数据的函数，默认为pyfin.DEFAULT_PROVIDER
    common_dates (bool): 是否保存相同是否，如果是，剔除NaN
    forward_fill (bool): 是否向前填充NaN
    clean_symbols (bool): 使用pyfin.utils.clean_symbols标准化代码
    column_names (list): 列名
    symbol_field_sep (char): symbol和field分隔符，如'600008:open'
    existing (DataFrame): 多数据源下载时用于合并df
    kwargs: 传给provider
    """

    if provider is None:
        provider = DEFAULT_PROVIDER

    symbols = utils.parse_arg(symbols)

    data = {}
    for symbol in symbols:
        s = symbol
        f = None

        bits = symbol.split(symbol_field_sep, 1)
        if len(bits) == 2:
            s = bits[0]
            f = bits[1]

        data[symbol] = provider(s, field=f, **kwargs)

    df = pd.DataFrame(data)
    df = df[symbols]

    if existing is not None:
        df = pyfin.merge(existing, df)

    if common_dates:
        df = df.dropna()

    if forward_fill:
        df = df.fillna(method='ffill')

    if column_names:
        cnames = utils.parse_arg(column_names)
        if len(cnames) != len(df.columns):
            raise ValueError('Column names must be of same length as symbols!')
        df.columns = cnames
    elif clean_symbols:
        df.columns = map(utils.clean_symbol, df.columns)

    return df


def web(symbol, field=None, start=None, end=None, source='netease'):
    """
    web数据源，可选：netease
    """
    tmp = None
    if source == 'netease':
        tmp = _get_netease(symbol, start=start, end=end)
    if tmp is None:
        raise ValueError('Failed to retrieve data for %s:%s' % (symbol, field))

    if field is not None:
        return tmp[field]
    else:
        return tmp['close']


def _get_netease(symbol, start='', end=''):
    """
    网易财经数据源，获得日线数据
    示例：http://quotes.money.163.com/service/chddata.html?code=600008&start=20150508&end=20150512
    """
    if not start:
        start = (datetime.datetime.now().date() + datetime.timedelta(days=-300)).strftime('%Y-%m-%d')
    if not end:
        end = datetime.datetime.now().date().strftime('%Y-%m-%d')
    start = start.replace('-', '')
    end = end.replace('-', '')
    data_url = "http://quotes.money.163.com/service/chddata.html?code=0" + symbol + "&start=" + start + "&end=" + end
    r = requests.get(data_url, stream=True)
    lines = r.content.decode('gb2312').split("\n")
    lines = lines[1:len(lines) - 1]
    bars = []
    for line in lines[::-1]:
        stock_info = line.split(",", 14)
        s_date = stock_info[0]
        s_close = float(stock_info[3])
        s_high = float(stock_info[4])
        s_low = float(stock_info[5])
        s_open = float(stock_info[6])
        s_volume = float(stock_info[11])
        bars.append([s_date, s_open, s_high, s_low, s_close, s_volume])
    bars = pd.DataFrame(bars, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    bars.index = pd.to_datetime(bars['datetime'], format='%Y-%m-%d')

    return bars


def csv(symbol, path='data.csv', field='', **kwargs):
    """
    本地csv数据源
    """
    if 'index_col' not in kwargs:
        kwargs['index_col'] = 0
    if 'parse_dates' not in kwargs:
        kwargs['parse_dates'] = True

    df = pd.read_csv(path, **kwargs)

    syb = symbol
    if field is not '' and field is not None:
        syb = '%s:%s' % (syb, field)

    if syb not in df:
        raise ValueError('Symbol(field) not present in csv file!')

    return df[syb]


DEFAULT_PROVIDER = web
