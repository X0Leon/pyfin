# -*- coding: utf-8 -*-

import pyfin
import pyfin.utils as utils
import pandas as pd
import tushare as ts


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
        t = symbol
        f = None

        bits = symbol.split(symbol_field_sep, 1)
        if len(bits) == 2:
            t = bits[0]
            f = bits[1]

        data[symbol] = provider(symbol=t, field=f, **kwargs)

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


def web(symbol, field=None, start=None, end=None, source='tushare'):
    """
    TuShare数据源
    """
    if source == 'tushare' and field is None:
        field = 'close'

    tmp = ts.get_k_data(code=symbol, start=start, end=end)
    tmp.index = pd.to_datetime(tmp['date'], format='%Y-%m-%d')
    tmp.index.name = 'datetime'

    if tmp is None:
        raise ValueError('Failed to retrieve data for %s:%s' % (symbol, field))

    if field:
        return tmp[field]
    else:
        return tmp


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
