# pyfin

Financial toolkit for quantitative investments of China stock.

用于量化投资和金融市场分析的常用工具库。当前版本：Version 1.0a (2016/12)

特性：提供收益评估、数据处理、绘图等功能；主要依赖numpy/scipy/pandas等。

## 安装

* 方式1：python setup.py install
* 方式2：pip install pyfin (推荐)
    
## 使用

示例：
    
    import pyfin
    closes_df = pyfin.get(['600008','600018','600028'], start='2015-01-01', end='2016-12-22')
    stats = closes_df.calc_stats()
    print(stats.display())
    
## Changelog

2016年12月24日，Version 1.0a

* 收益评估、数据处理和绘图等诸多方法
    
Focus on Chinese market, refer to Philippe's [ffn](https://github.com/pmorissette/ffn) for American market.

Copyright (c) 2016 X0Leon (Leon Zhang) Email: pku09zl[at]gmail[dot]com
