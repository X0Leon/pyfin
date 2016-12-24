# -*- coding: utf-8 -*-

import random
from . import utils
from .utils import format_percent, format_2float, format_100p, get_period_name
import numpy as np
import pandas as pd
from pandas.core.base import PandasObject
from tabulate import tabulate
from matplotlib import pyplot as plt
import sklearn.manifold
import sklearn.cluster
import sklearn.covariance
from scipy.optimize import minimize
import scipy.stats
from scipy.stats import t


def set_riskfree_rate(rf, update_all=False):

    """
    设置年化无风险利率，供PerformanceStats使用。
    影响所有PerformanceStats实例，除非其属性已被覆盖
    参数：
    rf (float): 年化利率
    update_all (bool): 更新所以实例的值
    """
    PerformanceStats._yearly_rf = rf
    PerformanceStats._monthly_rf = (np.power(1+rf, 1./12.) - 1.) * 12
    PerformanceStats._daily_rf = (np.power(1+rf, 1./252.) - 1.) * 252

    if update_all:
        from gc import get_objects
        for obj in get_objects():
            if isinstance(obj, PerformanceStats):
                obj.set_riskfree_rate(rf)


class PerformanceStats(object):

    """
    价格序列表现评估的类，提供绘图和统计的方法
    参数：
    price (Series): 价格Series

    属性:
    name (str): 即price series的name
    return_table (DataFrame): 月度和年度收益
    lookback_returns (Series): 不同时间周期的收益(1m, 3m, 6m, ytd...)，ytd: year to date
    stats (Series): 包含所有统计量的Series
    """

    # 用于计算夏普率的无风险收益
    _yearly_rf = 0.
    _monthly_rf = 0.
    _daily_rf = 0.

    def __init__(self, prices):
        super(PerformanceStats, self).__init__()
        self.prices = prices
        self.name = self.prices.name
        self._start = self.prices.index[0]
        self._end = self.prices.index[-1]

        self._update(self.prices)

    def set_riskfree_rate(self, rf):

        """
        设置年化无风险利率，更新全部统计计算结果
        参数：
        rf (float): 年化利率
        """

        self._yearly_rf = rf
        self._monthly_rf = (np.power(1+rf, 1./12.) - 1.) * 12
        self._daily_rf = (np.power(1+rf, 1./252.) - 1.) * 252

        self._update(self.prices)

    def _update(self, obj):
        self._calculate(obj)
        self.return_table = pd.DataFrame(self.return_table).T
        if len(self.return_table.columns) == 13:
            self.return_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                                         'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                                         'Nov', 'Dec', 'YTD']

        self.lookback_returns = pd.Series(
            [self.mtd, self.three_month, self.six_month, self.ytd,
             self.one_year, self.three_year, self.five_year,
             self.ten_year, self.cagr],
            ['mtd', '3m', '6m', 'ytd', '1y', '3y', '5y', '10y', 'incep'])
        self.lookback_returns.name = self.name

        st = self._stats()
        self.stats = pd.Series(
            [getattr(self, x[0]) for x in st if x[0] is not None],
            [x[0] for x in st if x[0] is not None]).drop_duplicates()

    def _calculate(self, obj):
        # default values
        self.daily_mean = np.nan
        self.daily_vol = np.nan
        self.daily_sharpe = np.nan
        self.best_day = np.nan
        self.worst_day = np.nan
        self.total_return = np.nan
        self.cagr = np.nan
        self.incep = np.nan
        self.drawdown = np.nan
        self.max_drawdown = np.nan
        self.drawdown_details = np.nan
        self.daily_skew = np.nan
        self.daily_kurt = np.nan
        self.monthly_returns = np.nan
        self.avg_drawdown = np.nan
        self.avg_drawdown_days = np.nan
        self.monthly_mean = np.nan
        self.monthly_vol = np.nan
        self.monthly_sharpe = np.nan
        self.best_month = np.nan
        self.worst_month = np.nan
        self.mtd = np.nan
        self.three_month = np.nan
        self.pos_month_perc = np.nan
        self.avg_up_month = np.nan
        self.avg_down_month = np.nan
        self.monthly_skew = np.nan
        self.monthly_kurt = np.nan
        self.six_month = np.nan
        self.yearly_returns = np.nan
        self.ytd = np.nan
        self.one_year = np.nan
        self.yearly_mean = np.nan
        self.yearly_vol = np.nan
        self.yearly_sharpe = np.nan
        self.best_year = np.nan
        self.worst_year = np.nan
        self.three_year = np.nan
        self.win_year_perc = np.nan
        self.twelve_month_win_perc = np.nan
        self.yearly_skew = np.nan
        self.yearly_kurt = np.nan
        self.five_year = np.nan
        self.ten_year = np.nan

        self.return_table = {}

        if len(obj) is 0:
            return

        self.start = obj.index[0]
        self.end = obj.index[-1]

        self.daily_prices = obj
        self.monthly_prices = obj.resample('M').last()
        self.yearly_prices = obj.resample('A').last()

        p = obj
        mp = self.monthly_prices
        yp = self.yearly_prices

        if len(p) is 1:
            return

        self.returns = p.to_returns()
        self.log_returns = p.to_log_returns()
        r = self.returns

        if len(r) < 2:
            return

        self.daily_mean = r.mean() * 252
        self.daily_vol = r.std() * np.sqrt(252)
        self.daily_sharpe = (self.daily_mean - self._daily_rf) / self.daily_vol
        self.best_day = r.max()
        self.worst_day = r.min()

        self.total_return = obj[-1] / obj[0] - 1
        self.ytd = self.total_return
        self.cagr = calc_cagr(p)
        self.incep = self.cagr

        self.drawdown = p.to_drawdown_series()
        self.max_drawdown = self.drawdown.min()
        self.drawdown_details = drawdown_details(self.drawdown)
        if self.drawdown_details is not None:
            self.avg_drawdown = self.drawdown_details['drawdown'].mean()
            self.avg_drawdown_days = self.drawdown_details['days'].mean()

        if len(r) < 4:
            return

        self.daily_skew = r.skew()

        if len(r[(~np.isnan(r)) & (r != 0)]) > 0:
            self.daily_kurt = r.kurt()

        self.monthly_returns = self.monthly_prices.to_returns()
        mr = self.monthly_returns

        if len(mr) < 2:
            return

        self.monthly_mean = mr.mean() * 12
        self.monthly_vol = mr.std() * np.sqrt(12)
        self.monthly_sharpe = ((self.monthly_mean - self._monthly_rf) /
                               self.monthly_vol)
        self.best_month = mr.max()
        self.worst_month = mr.min()

        self.mtd = p[-1] / mp[-2] - 1

        self.pos_month_perc = len(mr[mr > 0]) / float(len(mr) - 1)
        self.avg_up_month = mr[mr > 0].mean()
        self.avg_down_month = mr[mr <= 0].mean()

        for idx in mr.index:
            if idx.year not in self.return_table:
                self.return_table[idx.year] = {1: 0, 2: 0, 3: 0,
                                               4: 0, 5: 0, 6: 0,
                                               7: 0, 8: 0, 9: 0,
                                               10: 0, 11: 0, 12: 0}
            if not np.isnan(mr[idx]):
                self.return_table[idx.year][idx.month] = mr[idx]

        fidx = mr.index[0]
        try:
            self.return_table[fidx.year][fidx.month] = float(mp[0]) / p[0] - 1
        except ZeroDivisionError:
            self.return_table[fidx.year][fidx.month] = 0

        for idx in self.return_table:
            arr = np.array(list(self.return_table[idx].values()))
            self.return_table[idx][13] = np.prod(arr + 1) - 1

        if len(mr) < 3:
            return

        denom = p[:p.index[-1] - pd.DateOffset(months=3)]
        if len(denom) > 0:
            self.three_month = p[-1] / denom[-1] - 1

        if len(mr) < 4:
            return

        self.monthly_skew = mr.skew()

        if len(mr[(~np.isnan(mr)) & (mr != 0)]) > 0:
            self.monthly_kurt = mr.kurt()

        denom = p[:p.index[-1] - pd.DateOffset(months=6)]
        if len(denom) > 0:
            self.six_month = p[-1] / denom[-1] - 1

        self.yearly_returns = self.yearly_prices.to_returns()
        yr = self.yearly_returns

        if len(yr) < 2:
            return

        self.ytd = p[-1] / yp[-2] - 1

        denom = p[:p.index[-1] - pd.DateOffset(years=1)]
        if len(denom) > 0:
            self.one_year = p[-1] / denom[-1] - 1

        self.yearly_mean = yr.mean()
        self.yearly_vol = yr.std()
        self.yearly_sharpe = ((self.yearly_mean - self._yearly_rf) /
                              self.yearly_vol)
        self.best_year = yr.max()
        self.worst_year = yr.min()

        self.three_year = calc_cagr(p[p.index[-1] - pd.DateOffset(years=3):])

        self.win_year_perc = len(yr[yr > 0]) / float(len(yr) - 1)

        tot = 0
        win = 0
        for i in range(11, len(mr)):
            tot += 1
            if mp[i] / mp[i - 11] > 1:
                win += 1
        self.twelve_month_win_perc = float(win) / tot

        if len(yr) < 4:
            return

        self.yearly_skew = yr.skew()

        if len(yr[(~np.isnan(yr)) & (yr != 0)]) > 0:
            self.yearly_kurt = yr.kurt()

        self.five_year = calc_cagr(p[p.index[-1] - pd.DateOffset(years=5):])
        self.ten_year = calc_cagr(p[p.index[-1] - pd.DateOffset(years=10):])

        return

    def _stats(self):
        stats = [('start', 'Start', 'dt'),
                 ('end', 'End', 'dt'),
                 ('_yearly_rf', 'Risk-free rate', 'p'),
                 (None, None, None),
                 ('total_return', 'Total Return', 'p'),
                 ('daily_sharpe', 'Daily Sharpe', 'n'),
                 ('cagr', 'CAGR', 'p'),
                 ('max_drawdown', 'Max Drawdown', 'p'),
                 (None, None, None),
                 ('mtd', 'MTD', 'p'),
                 ('three_month', '3m', 'p'),
                 ('six_month', '6m', 'p'),
                 ('ytd', 'YTD', 'p'),
                 ('one_year', '1Y', 'p'),
                 ('three_year', '3Y (ann.)', 'p'),
                 ('five_year', '5Y (ann.)', 'p'),
                 ('ten_year', '10Y (ann.)', 'p'),
                 ('incep', 'Since Incep. (ann.)', 'p'),
                 (None, None, None),
                 ('daily_sharpe', 'Daily Sharpe', 'n'),
                 ('daily_mean', 'Daily Mean (ann.)', 'p'),
                 ('daily_vol', 'Daily Vol (ann.)', 'p'),
                 ('daily_skew', 'Daily Skew', 'n'),
                 ('daily_kurt', 'Daily Kurt', 'n'),
                 ('best_day', 'Best Day', 'p'),
                 ('worst_day', 'Worst Day', 'p'),
                 (None, None, None),
                 ('monthly_sharpe', 'Monthly Sharpe', 'n'),
                 ('monthly_mean', 'Monthly Mean (ann.)', 'p'),
                 ('monthly_vol', 'Monthly Vol (ann.)', 'p'),
                 ('monthly_skew', 'Monthly Skew', 'n'),
                 ('monthly_kurt', 'Monthly Kurt', 'n'),
                 ('best_month', 'Best Month', 'p'),
                 ('worst_month', 'Worst Month', 'p'),
                 (None, None, None),
                 ('yearly_sharpe', 'Yearly Sharpe', 'n'),
                 ('yearly_mean', 'Yearly Mean', 'p'),
                 ('yearly_vol', 'Yearly Vol', 'p'),
                 ('yearly_skew', 'Yearly Skew', 'n'),
                 ('yearly_kurt', 'Yearly Kurt', 'n'),
                 ('best_year', 'Best Year', 'p'),
                 ('worst_year', 'Worst Year', 'p'),
                 (None, None, None),
                 ('avg_drawdown', 'Avg. Drawdown', 'p'),
                 ('avg_drawdown_days', 'Avg. Drawdown Days', 'n'),
                 ('avg_up_month', 'Avg. Up Month', 'p'),
                 ('avg_down_month', 'Avg. Down Month', 'p'),
                 ('win_year_perc', 'Win Year %', 'p'),
                 ('twelve_month_win_perc', 'Win 12m %', 'p')]

        return stats

    def set_date_range(self, start=None, end=None):
        """
        更新统计、绘图等的日期范围。如果start和end都是None则重置到原始日期

        参数:
        start (date): 起始日期
        end (end): 终止日期
        """
        if start is None:
            start = self._start
        else:
            start = pd.to_datetime(start)

        if end is None:
            end = self._end
        else:
            end = pd.to_datetime(end)

        self._update(self.prices.ix[start:end])

    def display(self):
        """
        展示统计结果概览
        """
        print('Stats for %s from %s - %s' % (self.name, self.start, self.end))
        print('Annual risk-free rate considered: %s' % (format_percent(self._yearly_rf)))
        print('Summary:')
        data = [[format_percent(self.total_return), format_2float(self.daily_sharpe),
                 format_percent(self.cagr), format_percent(self.max_drawdown)]]
        print(tabulate(data, headers=['Total Return', 'Sharpe',
                                      'CAGR', 'Max Drawdown']))

        print('\nAnnualized Returns:')
        data = [[format_percent(self.mtd), format_percent(self.three_month), format_percent(self.six_month),
                 format_percent(self.ytd), format_percent(self.one_year), format_percent(self.three_year),
                 format_percent(self.five_year), format_percent(self.ten_year),
                 format_percent(self.incep)]]
        print(tabulate(data,
                       headers=['mtd', '3m', '6m', 'ytd', '1y',
                                '3y', '5y', '10y', 'incep.']))

        print('\nPeriodic:')
        data = [
            ['sharpe', format_2float(self.daily_sharpe), format_2float(self.monthly_sharpe),
             format_2float(self.yearly_sharpe)],
            ['mean', format_percent(self.daily_mean), format_percent(self.monthly_mean),
             format_percent(self.yearly_mean)],
            ['vol', format_percent(self.daily_vol), format_percent(self.monthly_vol),
             format_percent(self.yearly_vol)],
            ['skew', format_2float(self.daily_skew), format_2float(self.monthly_skew),
             format_2float(self.yearly_skew)],
            ['kurt', format_2float(self.daily_kurt), format_2float(self.monthly_kurt),
             format_2float(self.yearly_kurt)],
            ['best', format_percent(self.best_day), format_percent(self.best_month),
             format_percent(self.best_year)],
            ['worst', format_percent(self.worst_day), format_percent(self.worst_month),
             format_percent(self.worst_year)]]
        print(tabulate(data, headers=['daily', 'monthly', 'yearly']))

        print('\nDrawdowns:')
        data = [
            [format_percent(self.max_drawdown), format_percent(self.avg_drawdown),
             format_2float(self.avg_drawdown_days)]]
        print(tabulate(data, headers=['max', 'avg', '# days']))

        print('\nMisc:')
        data = [['avg. up month', format_percent(self.avg_up_month)],
                ['avg. down month', format_percent(self.avg_down_month)],
                ['up year %', format_percent(self.win_year_perc)],
                ['12m up %', format_percent(self.twelve_month_win_perc)]]
        print(tabulate(data))

    def display_monthly_returns(self):
        """
        展示月度收益率和过去一年收益（YTD）
        """
        data = [['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'YTD']]
        for k in self.return_table.index:
            r = self.return_table.ix[k].values
            data.append([k] + [format_100p(x) for x in r])
        print(tabulate(data, headers='firstrow'))

    def display_lookback_returns(self):
        """
        展示过去一段时间的收益
        """
        return self.lookback_returns.map('{:,.2%}'.format)

    def plot(self, period='m', figsize=(15, 5), title=None,
             logy=False, **kwargs):
        """
        绘图

        参数:
        period (str): 绘图的时间周期，沿用pandas的记号
        figsize ((x,y)): 图片尺寸
        title (str): 图题
        logy (bool): y周对数坐标
        kwargs: 其他pandas可用的绘图参数
        """
        if title is None:
            title = '%s %s price series' % (self.name, get_period_name(period))

        ser = self._get_series(period)
        ser.plot(figsize=figsize, title=title, logy=logy, **kwargs)

    def plot_histogram(self, period='m', figsize=(15, 5), title=None,
                       bins=20, **kwargs):
        """
        给定周期，绘制直方图

        参数:
        period (str): 绘图的时间周期，沿用pandas的记号
        figsize ((x,y)): 图片尺寸
        title (str): 图题
        bins (int): 直方图条柱数目
        kwargs: 其他pandas hist方法的参数
        """
        if title is None:
            title = '%s %s return histogram' % (
                self.name, get_period_name(period))

        ser = self._get_series(period).to_returns().dropna()

        plt.figure(figsize=figsize)
        ax = ser.hist(bins=bins, figsize=figsize, normed=True, **kwargs)
        ax.set_title(title)
        plt.axvline(0, linewidth=4)
        ser.plot(kind='kde')

    def _get_series(self, per):
        if per == 'y':
            per = 'a'
        return self.daily_prices.asfreq(per, 'ffill')

    def to_csv(self, sep=',', path=None):
        """
        返回csv字符串，如果path不为None，则保存至该路径

        参数:
        sep (char): 分隔符
        path (str): 如果None, 返回CSV字符串，否则保存至该路径
        """
        stats = self._stats()

        data = []
        first_row = ['Stat', self.name]
        data.append(sep.join(first_row))

        for stat in stats:
            k, n, f = stat

            if k is None:
                row = [''] * len(data[0])
                data.append(sep.join(row))
                continue

            row = [n]
            raw = getattr(self, k)
            if f is None:
                row.append(raw)
            elif f == 'p':
                row.append(format_percent(raw))
            elif f == 'n':
                row.append(format_2float(raw))
            elif f == 'dt':
                row.append(raw.strftime('%Y-%m-%d'))
            else:
                raise NotImplementedError('unsupported format %s' % f)

            data.append(sep.join(row))

        res = '\n'.join(data)

        if path is not None:
            with open(path, 'w') as fl:
                fl.write(res)
        else:
            return res


class GroupStats(dict):

    """
    GroupStats用于比较多个序列
    基于{price.name: PerformanceStats}的字典结构，并提供一些便捷方法

    序列次序保持，单个PerformanceStats对象可以通过index为止或者[]获取

    参数:
    prices (Series): 用于比较的多个序列

    属性:
    stats (DataFrame): 不同序列各一列，统计量在不同行
    lookback_returns (DataFrame): 不同回顾周期的收益率 (1m, 3m, 6m, ytd...)
    prices (DataFrame): 合并调整后的价格
    """

    def __init__(self, *prices):
        names = []
        for p in prices:
            if isinstance(p, pd.DataFrame):
                names.extend(p.columns)
            elif isinstance(p, pd.Series):
                names.append(p.name)
            else:
                print('else')
                names.append(getattr(p, 'name', 'n/a'))
        self._names = names

        self._prices = merge(*prices).dropna()

        self._prices = self._prices[self._names]

        if len(self._prices.columns) != len(set(self._prices.columns)):
            raise ValueError('One or more data series provided',
                             'have same name! Please provide unique names')

        self._start = self._prices.index[0]
        self._end = self._prices.index[-1]

        self._update(self._prices)

    def __getitem__(self, key):
        if type(key) == int:
            return self[self._names[key]]
        else:
            return self.get(key)

    def _update(self, data):
        self._calculate(data)

        self.lookback_returns = pd.DataFrame(
            {x.lookback_returns.name: x.lookback_returns for x in
             self.values()})

        self.stats = pd.DataFrame(
            {x.name: x.stats for x in self.values()})

    def _calculate(self, data):
        self.prices = data
        for c in data.columns:
            prc = data[c]
            self[c] = PerformanceStats(prc)

    def _stats(self):
        stats = [('start', 'Start', 'dt'),
                 ('end', 'End', 'dt'),
                 ('_yearly_rf', 'Risk-free rate', 'p'),
                 (None, None, None),
                 ('total_return', 'Total Return', 'p'),
                 ('daily_sharpe', 'Daily Sharpe', 'n'),
                 ('cagr', 'CAGR', 'p'),
                 ('max_drawdown', 'Max Drawdown', 'p'),
                 (None, None, None),
                 ('mtd', 'MTD', 'p'),
                 ('three_month', '3m', 'p'),
                 ('six_month', '6m', 'p'),
                 ('ytd', 'YTD', 'p'),
                 ('one_year', '1Y', 'p'),
                 ('three_year', '3Y (ann.)', 'p'),
                 ('five_year', '5Y (ann.)', 'p'),
                 ('ten_year', '10Y (ann.)', 'p'),
                 ('incep', 'Since Incep. (ann.)', 'p'),
                 (None, None, None),
                 ('daily_sharpe', 'Daily Sharpe', 'n'),
                 ('daily_mean', 'Daily Mean (ann.)', 'p'),
                 ('daily_vol', 'Daily Vol (ann.)', 'p'),
                 ('daily_skew', 'Daily Skew', 'n'),
                 ('daily_kurt', 'Daily Kurt', 'n'),
                 ('best_day', 'Best Day', 'p'),
                 ('worst_day', 'Worst Day', 'p'),
                 (None, None, None),
                 ('monthly_sharpe', 'Monthly Sharpe', 'n'),
                 ('monthly_mean', 'Monthly Mean (ann.)', 'p'),
                 ('monthly_vol', 'Monthly Vol (ann.)', 'p'),
                 ('monthly_skew', 'Monthly Skew', 'n'),
                 ('monthly_kurt', 'Monthly Kurt', 'n'),
                 ('best_month', 'Best Month', 'p'),
                 ('worst_month', 'Worst Month', 'p'),
                 (None, None, None),
                 ('yearly_sharpe', 'Yearly Sharpe', 'n'),
                 ('yearly_mean', 'Yearly Mean', 'p'),
                 ('yearly_vol', 'Yearly Vol', 'p'),
                 ('yearly_skew', 'Yearly Skew', 'n'),
                 ('yearly_kurt', 'Yearly Kurt', 'n'),
                 ('best_year', 'Best Year', 'p'),
                 ('worst_year', 'Worst Year', 'p'),
                 (None, None, None),
                 ('avg_drawdown', 'Avg. Drawdown', 'p'),
                 ('avg_drawdown_days', 'Avg. Drawdown Days', 'n'),
                 ('avg_up_month', 'Avg. Up Month', 'p'),
                 ('avg_down_month', 'Avg. Down Month', 'p'),
                 ('win_year_perc', 'Win Year %', 'p'),
                 ('twelve_month_win_perc', 'Win 12m %', 'p')]

        return stats

    def set_riskfree_rate(self, rf):

        """
        设置年化无风险收益率，计算年度、月度、天收益率，衡量表现得统计量也重新计算
        影响GroupStats对象包含的那些PerformanceStats实例

        参数:
        rf (float): 年化无风险利率
        """
        for key in self._names:
            self[key].set_riskfree_rate(rf)

    def set_date_range(self, start=None, end=None):
        """
        更新统计、绘图等的日期范围。如果start和end都是None则重置到原始日期

        参数:
        start (date): 起始日期
        end (end): 终止日期
        """
        if start is None:
            start = self._start
        else:
            start = pd.to_datetime(start)

        if end is None:
            end = self._end
        else:
            end = pd.to_datetime(end)

        self._update(self._prices.ix[start:end])

    def display(self):
        """
        展示统计结果
        """
        data = []
        first_row = ['Stat']
        first_row.extend(self._names)
        data.append(first_row)

        stats = self._stats()

        for stat in stats:
            k, n, f = stat

            if k is None:
                row = [''] * len(data[0])
                data.append(row)
                continue

            row = [n]
            for key in self._names:
                raw = getattr(self[key], k)
                if f is None:
                    row.append(raw)
                elif f == 'p':
                    row.append(format_percent(raw))
                elif f == 'n':
                    row.append(format_2float(raw))
                elif f == 'dt':
                    row.append(raw.strftime('%Y-%m-%d'))
                else:
                    raise NotImplementedError('unsupported format %s' % f)
            data.append(row)

        print(tabulate(data, headers='firstrow'))

    def display_lookback_returns(self):
        """
        展示当前回顾周期下各价格序列的收益率
        """
        return self.lookback_returns.apply(
            lambda x: x.map('{:,.2%}'.format), axis=1)

    def plot(self, period='m', figsize=(15, 5), title=None,
             logy=False, **kwargs):
        """
        绘图

        参数:
        period (str): 绘图的时间周期，沿用pandas的记号
        figsize ((x,y)): 图片尺寸
        title (str): 图题
        logy (bool): y周对数坐标
        kwargs: 其他pandas可用的绘图参数
        """

        if title is None:
            title = '%s equity progression' % get_period_name(period)
        ser = self._get_series(period).rebase()
        ser.plot(figsize=figsize, logy=logy,
                 title=title, **kwargs)

    def plot_scatter_matrix(self, period='m', title=None,
                            figsize=(10, 10), **kwargs):
        """
        绘图，封装pandas的scatter_matrix

        参数:
        period (str): 绘图的时间周期，沿用pandas的记号
        figsize ((x,y)): 图片尺寸
        title (str): 图题
        kwargs: 其他pandas scatter_matrix可用的绘图参数
        """
        if title is None:
            title = '%s return scatter matrix' % get_period_name(period)

        plt.figure()
        ser = self._get_series(period).to_returns().dropna()
        pd.scatter_matrix(ser, figsize=figsize, **kwargs)
        plt.suptitle(title)

    def plot_histograms(self, period='m', title=None,
                        figsize=(10, 10), **kwargs):
        """
        绘图，封装pandas的hist

        参数:
        period (str): 绘图的时间周期，沿用pandas的记号
        figsize ((x,y)): 图片尺寸
        title (str): 图题
        kwargs: 其他pandas hist可用的绘图参数
        """
        if title is None:
            title = '%s return histogram matrix' % get_period_name(period)

        plt.figure()
        ser = self._get_series(period).to_returns().dropna()
        ser.hist(figsize=figsize, **kwargs)
        plt.suptitle(title)

    def plot_correlation(self, period='m', title=None,
                         figsize=(12, 6), **kwargs):
        """
        绘制相关系数的热度图

        参数:
        period (str): 绘图的时间周期，沿用pandas的记号
        figsize ((x,y)): 图片尺寸
        title (str): 图题
        kwargs: 其他pandas plot_corr_heatmap可用的绘图参数
        """
        if title is None:
            title = '%s return correlation matrix' % get_period_name(period)

        rets = self._get_series(period).to_returns().dropna()
        return rets.plot_corr_heatmap(title=title, figsize=figsize, **kwargs)

    def _get_series(self, per):
        if per == 'y':
            per = 'a'
        return self.prices.asfreq(per, 'ffill')

    def to_csv(self, sep=',', path=None):
        """
        返回csv字符串，如果path不为None，则保存至该路径

        参数:
        sep (char): 分隔符
        path (str): 如果None, 返回CSV字符串，否则保存至该路径
        """
        data = []
        first_row = ['Stat']
        first_row.extend(self._names)
        data.append(sep.join(first_row))

        stats = self._stats()

        for stat in stats:
            k, n, f = stat

            if k is None:
                row = [''] * len(data[0])
                data.append(sep.join(row))
                continue

            row = [n]
            for key in self._names:
                raw = getattr(self[key], k)
                if f is None:
                    row.append(raw)
                elif f == 'p':
                    row.append(format_percent(raw))
                elif f == 'n':
                    row.append(format_2float(raw))
                elif f == 'dt':
                    row.append(raw.strftime('%Y-%m-%d'))
                else:
                    raise NotImplementedError('unsupported format %s' % f)
            data.append(sep.join(row))

        res = '\n'.join(data)

        if path is not None:
            with open(path, 'w') as fl:
                fl.write(res)
        else:
            return res


def to_returns(prices):
    """
    计算价格序列的算术收益率，(p1 / p0) - 1

    参数:
    prices: price series
    """
    return prices / prices.shift(1) - 1


def to_log_returns(prices):
    """
    计算价格的对数收益率，ln(p1/p0)

    参数:
    prices: price series
    """
    return np.log(prices / prices.shift(1))


def to_price_index(returns, start=100):
    """
    由收益率序列计算价格指数，cumprod(1+r)

    参数:
    returns (Series): 算术收益率series
    start (number): 起始价格
    """
    return (returns.replace(to_replace=np.nan, value=0) + 1).cumprod() * start


def rebase(prices, value=100):
    """
    调整所有series到相同的起始价格，便于比较和绘图不同价格序列

    参数:
    prices: price series
    value (number): 起始价格
    """
    return prices / prices.ix[0] * value


def calc_perf_stats(obj):
    """
    计算对象的表现，返回PerformanceStats对象，其包含所有统计信息

    参数:
    obj: prices series
    """
    return PerformanceStats(obj)


def calc_stats(obj):
    """
    计算对象的表现，可以接受Series或DataFrame
    """
    if isinstance(obj, pd.Series):
        return PerformanceStats(obj)
    elif isinstance(obj, pd.DataFrame):
        return GroupStats(*[obj[x] for x in obj.columns])
    else:
        raise NotImplementedError('Unsupported type')


def to_drawdown_series(prices):
    """
    计算回撤序列，drawdown = current / hwm - 1

    参数:
    prices (Series or DataFrame): Series of prices.
    """
    drawdown = prices.copy()
    drawdown = drawdown.fillna(method='ffill')

    drawdown[np.isnan(drawdown)] = -np.Inf

    roll_max = np.maximum.accumulate(drawdown)
    drawdown = drawdown / roll_max - 1.
    return drawdown


def calc_max_drawdown(prices):
    """
    计算最大回撤
    """
    return (prices / prices.expanding(min_periods=1).max()).min() - 1


def drawdown_details(drawdown):
    """
    回撤详情，返回的DataFrame包括列：start, end, days (duration) and drawdown
    注意，其中days是实际的时间 (calendar days)，而非交易日 (trading days)


    参数:
    drawdown (pandas.Series): drawdown Series
    """
    # TODO: 此函数需要改进

    is_zero = drawdown == 0
    start = ~is_zero & is_zero.shift(1)
    start = list(start[start == True].index)

    end = is_zero & (~is_zero).shift(1)
    end = list(end[end == True].index)

    if len(start) is 0:
        return None
    if len(end) is 0:
        end.append(drawdown.index[-1])
    if start[0] > end[0]:
        start.insert(0, drawdown.index[0])
    if start[-1] > end[-1]:
        end.append(drawdown.index[-1])

    result = pd.DataFrame(columns=('start', 'end', 'days', 'drawdown'),
                          index=range(0, len(start)))

    for i in range(0, len(start)):
        dd = drawdown[start[i]:end[i]].min()
        result.ix[i] = (start[i], end[i], (end[i] - start[i]).days, dd)

    return result


def calc_cagr(prices):
    """
    计算价格序列的复合年化收益率CAGR (compound annual growth rate)

    参数:
    prices (pandas.Series): prices Series

    返回:
    CAGR (float)
    """
    start = prices.index[0]
    end = prices.index[-1]
    return (prices.ix[-1] / prices.ix[0]) ** (1 / year_frac(start, end)) - 1


def calc_risk_return_ratio(price):
    """
    计算return/risk，基本上等于未经无风险利率调整的夏普率
    """
    return price.mean() / price.std()


def calc_information_ratio(returns, benchmark_returns):
    """
    信息率：IR = E(r_p - r_b) / std(r_p - r_b)
    http://en.wikipedia.org/wiki/Information_ratio
    """
    diff_rets = returns - benchmark_returns
    diff_std = diff_rets.std()

    if np.isnan(diff_std) or diff_std == 0:
        return 0.0

    return diff_rets.mean() / diff_std


def calc_prob_mom(returns, other_returns):
    """
    Probabilistic momentum
    一项资产表现优于另一资产的概率或置信程度

    Source:
    http://cssanalytics.wordpress.com/2014/01/28/are-simple-momentum-strategies-too-dumb-introducing-probabilistic-momentum/
    """
    return t.cdf(returns.calc_information_ratio(other_returns), len(returns) - 1)


def calc_total_return(prices):
    """
    计算series总收益，last / first - 1
    """
    return (prices.ix[-1] / prices.ix[0]) - 1


def year_frac(start, end):
    """
    计算两个日期间的时间长度，以年为单位

    参数:
    start (datetime): start date
    end (datetime): end date

    """
    if start > end:
        raise ValueError('start cannot be larger than end')

    return (end - start).total_seconds() / 31557600


def merge(*series):
    """
    Merge Series and/or DataFrames
    """
    dfs = []
    for s in series:
        if isinstance(s, pd.DataFrame):
            dfs.append(s)
        elif isinstance(s, pd.Series):
            tmpdf = pd.DataFrame({s.name: s})
            dfs.append(tmpdf)
        else:
            raise NotImplementedError('Unsupported merge type')

    return pd.concat(dfs, axis=1)


def drop_duplicate_cols(df):
    """
    移除冗余列
    """
    names = set(df.columns)
    for n in names:
        if len(df[n].shape) > 1:
            sub = df[n]
            sub.columns = ['%s-%s' % (n, x) for x in range(sub.shape[1])]
            keep = sub.count().idxmax()
            del df[n]
            df[n] = sub[keep]

    return df


def to_monthly(series, method='ffill', how='end'):
    """
    将时间序列转换成月频，使用asfreq_actual
    """
    return series.asfreq_actual('M', method=method, how=how)


def asfreq_actual(series, freq, method='ffill', how='end', normalize=False):
    """
    将时间序列转换成某频率，类似于pandas的asfreq函数，但是保证真实的日期
    例如，如果最后一个数据点在一月为29号，则用29号，而不是31号
    """
    orig = series
    is_series = False
    if isinstance(series, pd.Series):
        is_series = True
        name = series.name if series.name else 'data'
        orig = pd.DataFrame({name: series})

    t = pd.concat([orig, pd.DataFrame({'dt': orig.index.values},
                                      index=orig.index.values)], axis=1)
    dts = t.asfreq(freq=freq, method=method, how=how,
                   normalize=normalize)['dt']

    res = orig.ix[dts.values]

    if is_series:
        return res[name]
    else:
        return res


def calc_inv_vol_weights(returns):
    """
    计算每列波动率的倒数作为权重，这样通过控制每个品种的头寸可以使之波动率都在同一水平

    已剔除全零或NaN的列

    Returns:
        Series {col_name: weight}
    """
    vol = 1.0 / returns.std()
    vol[np.isinf(vol)] = np.NaN
    vols = vol.sum()
    return vol / vols


def calc_mean_var_weights(returns, weight_bounds=(0., 1.),
                          rf=0.,
                          covar_method='ledoit-wolf'):
    """
    计算DataFrame每列收益的均值-方差权重

    参数:
    returns (DataFrame): 多资产的收益率
    weight_bounds ((low, high)): 用于优化的权重界限
    rf (float): 无风险收益率
    covar_method (str): 协方差矩阵的估计方法，目前支持ledoit-wolf, standard

    返回:
        Series {col_name: weight}

    """
    def fitness(weights, exp_rets, covar, rf):
        mean = sum(exp_rets * weights)
        var = np.dot(np.dot(weights, covar), weights)
        util = (mean - rf) / np.sqrt(var)  # Sharpe ratio
        return -util

    n = len(returns.columns)

    exp_rets = returns.mean()

    if covar_method == 'ledoit-wolf':
        covar = sklearn.covariance.ledoit_wolf(returns)[0]
    elif covar_method == 'standard':
        covar = returns.cov()
    else:
        raise NotImplementedError('covar_method not implemented')

    weights = np.ones([n]) / n
    bounds = [weight_bounds for i in range(n)]
    constraints = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})
    optimized = minimize(fitness, weights, (exp_rets, covar, rf),
                         method='SLSQP', constraints=constraints,
                         bounds=bounds)

    if not optimized.success:
        raise Exception(optimized.message)

    return pd.Series({returns.columns[i]: optimized.x[i] for i in range(n)})


def get_num_days_required(offset, period='d', perc_required=0.90):
    """
    估计需要的交易日数，用以检查给定周期内数据集是否正常

    参数:
    offset (DateOffset): Offset (lookback) period.
    period (str): Period string, 'd', 'm', 'y'
    perc_required (float): percentage of number of days expected required.
    """
    x = pd.to_datetime('2010-01-01')
    delta = x - (x - offset)
    # 粗略估计交易日数
    days = delta.days * 0.69

    if period == 'd':
        req = days * perc_required
    elif period == 'm':
        req = (days / 20) * perc_required
    elif period == 'y':
        req = (days / 252) * perc_required
    else:
        raise NotImplementedError(
            'Period not supported. Supported periods are d, m, y')

    return req


def calc_clusters(returns, n=None, plot=False):
    """
    基于k-means聚类

    参数:
    returns (pd.DataFrame): DataFrame of returns
    n (int): 聚类数，如果没有指定则自动识别
    plot (bool): 是否绘图

    返回:
    dict: {cluster No.: [col names]}
    """
    corr = returns.corr()
    diss = 1 - corr # dissimilarity matrix

    # scale down to 2 dimensions using MDS (multi-dimensional scaling)
    mds = sklearn.manifold.MDS(dissimilarity='precomputed')
    xy = mds.fit_transform(diss)

    def routine(k):
        km = sklearn.cluster.KMeans(n_clusters=k)
        km_fit = km.fit(xy)
        labels = km_fit.labels_
        centers = km_fit.cluster_centers_
        mappings = dict(zip(returns.columns, labels))

        totss = 0
        withinss = 0
        avg = np.array([np.mean(xy[:, 0]), np.mean(xy[:, 1])])
        for idx, lbl in enumerate(labels):
            withinss += sum((xy[idx] - centers[lbl]) ** 2)
            totss += sum((xy[idx] - avg) ** 2)
        pvar_expl = 1.0 - withinss / totss

        return mappings, pvar_expl, labels

    if n:
        result = routine(n)
    else:
        n = len(returns.columns)
        n1 = int(np.ceil(n * 0.6666666666))
        for i in range(2, n1 + 1):
            result = routine(i)
            if result[1] > 0.9:
                break

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(xy[:, 0], xy[:, 1], c=result[2], s=90)
        for i, txt in enumerate(returns.columns):
            ax.annotate(txt, (xy[i, 0], xy[i, 1]), size=14)

    # sanitize return value
    tmp = result[0]
    # {cluster: [list of symbols], cluster2: [...]}
    inv_map = {}
    for k, v in tmp.iteritems():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)

    return inv_map


def calc_ftca(returns, threshold=0.5):
    """
    Implementation of David Varadi's Fast Threshold Clustering Algorithm (FTCA).

    http://cssanalytics.wordpress.com/2013/11/26/fast-threshold-clustering-algorithm-ftca/

    比使用k-means聚类更稳定，如果需要得到更多类，threshold设置大些

    参数:
    returns (DataFrame)：每列对应一种资产
    threshold (float): 用于控制序列相关程度的阈值，从而获得不同数量的聚类

    返回:
    dict: {cluster No.: [col names]}

    """
    # cluster index (name)
    i = 0
    # correlation matrix
    corr = returns.corr()
    # remaining securities to cluster
    remain = list(corr.index.copy())
    n = len(remain)
    res = {}

    while n > 0:
        # if only one left then create cluster and finish
        if n == 1:
            i += 1
            res[i] = remain
            n = 0
        # if not then we have some work to do
        else:
            # filter down correlation matrix to current remain
            cur_corr = corr[remain].ix[remain]
            # get mean correlations, ordered
            mc = cur_corr.mean().order()
            # get lowest and highest mean correlation
            low = mc.index[0]
            high = mc.index[-1]

            # case if corr(high,low) > threshold
            if corr[high][low] > threshold:
                i += 1

                # new cluster for high and low
                res[i] = [low, high]
                remain.remove(low)
                remain.remove(high)

                rmv = []
                for x in remain:
                    avg_corr = (corr[x][high] + corr[x][low]) / 2.0
                    if avg_corr > threshold:
                        res[i].append(x)
                        rmv.append(x)
                [remain.remove(x) for x in rmv]

                n = len(remain)

            # otherwise we are creating two clusters - one for high
            # and one for low
            else:
                # add cluster with HC
                i += 1
                res[i] = [high]
                remain.remove(high)
                remain.remove(low)

                rmv = []
                for x in remain:
                    if corr[x][high] > threshold:
                        res[i].append(x)
                        rmv.append(x)
                [remain.remove(x) for x in rmv]

                i += 1
                res[i] = [low]

                rmv = []
                for x in remain:
                    if corr[x][low] > threshold:
                        res[i].append(x)
                        rmv.append(x)
                [remain.remove(x) for x in rmv]

                n = len(remain)

    return res


def limit_weights(weights, limit=0.1):
    """
    根据权重上限，重新按比例分配超限资产的头寸至其他资产

    例:
    weights = {a: 0.7, b: 0.2, c: 0.1}
    limit = 0.5
    a (0.7-0.5) -> b, c => {a: 0.5, b: 0.33, c: 0.167}

    参数:
    weights (Series): 描述权重的序列
    limit (float): 权重上限
    """
    if 1.0 / limit > len(weights):
        raise ValueError('invalid limit -> 1 / limit must be <= len(weights)')

    if isinstance(weights, dict):
        weights = pd.Series(weights)

    if np.round(weights.sum(), 1) != 1.0:
        raise ValueError('Expecting weights (that sum to 1) - sum is %s'
                         % weights.sum())

    res = np.round(weights.copy(), 4)
    to_rebalance = (res[res > limit] - limit).sum()

    ok = res[res < limit]
    ok += (ok / ok.sum()) * to_rebalance

    res[res > limit] = limit
    res[res < limit] = ok

    if not np.all([x <= limit for x in res]):
        return limit_weights(res, limit=limit)

    return res


def random_weights(n, bounds=(0., 1.), total=1.0):
    """
    生成伪随机权重，用于benchmark

    参数:
    n (int): 随机权重的数目
    bounds ((low, high)): 每个权重的边界
    total (float): 权重加和

    """
    low = bounds[0]
    high = bounds[1]

    if high < low:
        raise ValueError('Higher bound must be greater or equal to lower bound')

    if n * high < total or n * low > total:
        raise ValueError('solution not possible with given n and bounds')

    w = [0] * n
    tgt = -float(total)

    for i in range(n):
        rn = n - i - 1
        rhigh = rn * high
        rlow = rn * low

        lowb = max(-rhigh - tgt, low)
        highb = min(-rlow - tgt, high)

        rw = random.uniform(lowb, highb)
        w[i] = rw

        tgt += rw

    random.shuffle(w)
    return w


def plot_heatmap(data, title='Heatmap', show_legend=True,
                 show_labels=True, label_fmt='.2f',
                 vmin=None, vmax=None,
                 figsize=None,
                 cmap='RdBu', **kwargs):
    """
    绘制热度图，基于matplotlib的pcolor.

    参数:
    data (DataFrame): 如相关矩阵
    title (string): 图题
    show_legend (bool): color bar
    show_labels (bool): value labels
    label_fmt (str): Label format string
    vmin (float): Min value for scale
    vmax (float): Max value for scale
    cmap (string): Color map
    kwargs: matplotlib pcolor的其他参数
    """
    fig, ax = plt.subplots(figsize=figsize)

    heatmap = ax.pcolor(data, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.invert_yaxis()

    plt.title(title)

    if show_legend:
        fig.colorbar(heatmap)

    if show_labels:
        vals = data.values
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                plt.text(x + 0.5, y + 0.5, format(vals[y, x], label_fmt),
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='w')

    plt.yticks(np.arange(0.5, len(data.index), 1), data.index)
    plt.xticks(np.arange(0.5, len(data.columns), 1), data.columns)

    plt.show()


def plot_corr_heatmap(data, **kwargs):
    """
    相关系数的热度图
    """
    return plot_heatmap(data.corr(), vmin=-1, vmax=1, **kwargs)


def rollapply(data, window, fn):
    """
    对滚动窗口window内的data应用函数fn计算

    参数:
    data (Series/DataFrame): Series或DataFrame
    window (int): 滚动窗口大小
    fn (function): 应用于窗口的函数

    返回：
    依据data类型返回计算结果
    """
    # TODO：pandas自身也有几种实现，看是否有必要
    res = data.copy()
    res[:] = np.nan
    n = len(data)

    if window > n:
        return res

    for i in range(window - 1, n):
        res.iloc[i] = fn(data.iloc[i - window + 1:i + 1])

    return res


def _winsorize_wrapper(x, limits):
    """
    封装scipy winsorize函数以剔除NaN
    """
    if hasattr(x, 'dropna'):
        if len(x.dropna()) == 0:
            return x

        x[~np.isnan(x)] = scipy.stats.mstats.winsorize(x[~np.isnan(x)],
                                                       limits=limits)
        return x
    else:
        return scipy.stats.mstats.winsorize(x, limits=limits)


def winsorize(x, axis=0, limits=0.01):
    """
    Winsorize values based on limits
    """
    x = x.copy()

    if isinstance(x, pd.DataFrame):
        return x.apply(_winsorize_wrapper, axis=axis, args=(limits, ))
    else:
        return pd.Series(_winsorize_wrapper(x, limits).values,
                         index=x.index)


def rescale(x, min=0., max=1., axis=0):
    """
    调整值到[min, max]范围
    """
    def innerfn(x, min, max):
        return np.interp(x, [np.min(x), np.max(x)], [min, max])

    if isinstance(x, pd.DataFrame):
        return x.apply(innerfn, axis=axis, args=(min, max,))
    else:
        return pd.Series(innerfn(x, min, max), index=x.index)


def annualize(returns, durations, one_year=365.):
    """
    年化收益率，(1 + returns) ** (1 / (durations / one_year)) - 1
    """
    return (1. + returns) ** (1. / (durations / one_year)) - 1.


def extend_pandas():
    """
    扩展pandas的PandasObject (Series, DataFrame)，用上述定义的函数

    例如:
        prices_df.to_returns().dropna().calc_clusters()
    """
    PandasObject.to_returns = to_returns
    PandasObject.to_log_returns = to_log_returns
    PandasObject.to_price_index = to_price_index
    PandasObject.rebase = rebase
    PandasObject.calc_perf_stats = calc_perf_stats
    PandasObject.to_drawdown_series = to_drawdown_series
    PandasObject.calc_max_drawdown = calc_max_drawdown
    PandasObject.calc_cagr = calc_cagr
    PandasObject.calc_total_return = calc_total_return
    PandasObject.as_percent = utils.as_percent
    PandasObject.as_format = utils.as_format
    PandasObject.to_monthly = to_monthly
    PandasObject.asfreq_actual = asfreq_actual
    PandasObject.drop_duplicate_cols = drop_duplicate_cols
    PandasObject.calc_information_ratio = calc_information_ratio
    PandasObject.calc_prob_mom = calc_prob_mom
    PandasObject.calc_risk_return_ratio = calc_risk_return_ratio
    PandasObject.calc_inv_vol_weights = calc_inv_vol_weights
    PandasObject.calc_mean_var_weights = calc_mean_var_weights
    PandasObject.calc_clusters = calc_clusters
    PandasObject.calc_ftca = calc_ftca
    PandasObject.calc_stats = calc_stats
    PandasObject.plot_heatmap = plot_heatmap
    PandasObject.plot_corr_heatmap = plot_corr_heatmap
    PandasObject.rollapply = rollapply
    PandasObject.winsorize = winsorize
    PandasObject.rescale = rescale
