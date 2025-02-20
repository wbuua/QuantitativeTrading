import matplotlib

matplotlib.use('Agg')
import tushare as ts
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 多字体回退
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示


# ====================
# 数据获取与处理
# ====================
def get_data(token, code, start, end):
    ts.set_token(token)
    pro = ts.pro_api()

    # 获取股票日线数据
    df = pro.daily(
        ts_code=code,
        start_date=start,
        end_date=end,
        adj='qfq'
    )

    # 处理数据格式
    df = df.sort_values('trade_date')  # 确保日期升序排列
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'vol': 'Volume'
    }, inplace=True)
    print(df.head())

    return df


# ====================
# 回测引擎与策略类
# ====================
class SmaCrossStrategy(bt.Strategy):
    params = (
        ('fast', 5),  # 短期均线周期
        ('slow', 10)  # 长期均线周期
    )

    def __init__(self):
        # 初始化参数，创建移动平均线指标
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.p.fast)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        self.trade_count = 0
        self.total_commission = 0

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)  # 使用当前数据时间
        print(f'[{dt}] {txt}')  # 输出到控制台

    def notify_order(self, order):
        """订单状态处理"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f'买入执行 价格:{order.executed.price:.2f}')
            elif order.issell():
                self.log(f'卖出执行 价格:{order.executed.price:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'订单异常 状态:{order.getstatusname()}')

        self.order = None

    def next(self):
        if not self.position:  # 空仓状态
            if self.crossover > 0:  # 金叉信号
                self.order = self.buy()  # 全仓买入
        else:  # 持仓状态
            if self.crossover < 0:  # 死叉信号
                self.order = self.sell()  # 清仓卖出


# ====================
# 分析与可视化
# ====================
def analyze_and_visualize(result):
    # 提取带时间戳的收益率序列
    returns = pd.Series(
        result[0].analyzers._TimeReturn.get_analysis()
    ).sort_index()

    # 生成QuantStats报告
    qs.reports.html(
        returns,
        output='./Output.html',
        title='策略回测报告',
        figfmt='svg',
        rf=0.0,
        periods='daily'
    )


# ====================
# 主程序
# ====================
if __name__ == '__main__':
    # 配置参数
    TOKEN = '1bfd3d06307626a68c17cccda5adfab9b74cff94a91ec6467551975a'  # tushare token
    CODE = '000627.SZ'  # 天茂
    START = '20220101'
    END = '20221230'
    INIT_CASH = 1000000

    # ========== 获取并处理数据 ==========
    df = get_data(token=TOKEN, code=CODE, start=START, end=END)

    # ========== 初始化回测引擎 ==========
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INIT_CASH)
    cerebro.broker.setcommission(commission=0.0001, stocklike=True)  # 佣金0.01%

    # 添加数据
    data = bt.feeds.PandasData(
        dataname=df,
        timeframe=bt.TimeFrame.Days
    )
    cerebro.adddata(data)

    # 添加策略和分析器
    cerebro.addstrategy(SmaCrossStrategy)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')

    # ========== 执行回测 ==========
    print('初始资金: %.2f' % cerebro.broker.getvalue())
    result = cerebro.run()
    print('最终资金: %.2f' % cerebro.broker.getvalue())

    # ========== 分析和可视化 ==========
    analyze_and_visualize(result)
