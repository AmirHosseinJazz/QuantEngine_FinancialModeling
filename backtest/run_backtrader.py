import os

# Set MPLBACKEND to Agg before importing any matplotlib or Backtrader modules
os.environ["MPLBACKEND"] = "Agg"
import matplotlib

matplotlib.use("Agg")

import backtrader as bt
import datetime
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from preprocess import get_backtrader_data
import backtrader.analyzers as btanalyzers

load_dotenv()


class SimpleStrategy(bt.Strategy):
    def __init__(self):
        self.counter = 0
        self.order = None
        self.buy_price = None
        self.buy_comm = None

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} {txt}")

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.status == order.Completed:
                if order.isbuy():
                    self.buy_price = order.executed.price
                    self.buy_comm = order.executed.comm
                else:
                    self.log(
                        f"SELL EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm: {order.executed.comm}"
                    )
            self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f"TRADE PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}")

    def next(self):
        self.counter += 1
        if self.order:
            return

        if self.counter % 5 == 0:
            self.order = self.buy()
        elif self.position:
            self.order = self.sell()


def store_metrics(data_points, trade_points):
    conn = psycopg2.connect(
        host=os.getenv("TIMESCALE_HOST_NAME"),
        port=os.getenv("LOCAL_TIMESCALE_PORT"),
        user=os.getenv("TIMESCALE_USER"),
        password=os.getenv("TIMESCALE_PASSWORD"),
        dbname=os.getenv("TIMESCALE_DATABASE"),
    )
    cursor = conn.cursor()

    metrics_query = """
        INSERT INTO "BTCUSDT".strategy_metrics (time, close, position, cash, value)
        VALUES %s
        ON CONFLICT (time) DO NOTHING
    """
    execute_values(cursor, metrics_query, data_points)

    trades_query = """
        INSERT INTO "BTCUSDT".strategy_trades (time, trade_type, price, pnl, pnlcomm)
        VALUES %s
        ON CONFLICT (time) DO NOTHING
    """
    execute_values(cursor, trades_query, trade_points)

    conn.commit()
    cursor.close()
    conn.close()


class MetricsLogger(bt.Analyzer):
    def start(self):
        self.data_points = []
        self.trade_points = []

    def stop(self):
        store_metrics(self.data_points, self.trade_points)

    def next(self):
        time = self.strategy.data.datetime.datetime()
        close = self.strategy.data.close[0]
        position = self.strategy.position.size
        cash = self.strategy.broker.get_cash()
        value = self.strategy.broker.get_value()
        self.data_points.append((time, close, position, cash, value))

    def notify_trade(self, trade):
        if trade.isclosed:
            time = trade.close_datetime()
            trade_type = "BUY" if trade.size > 0 else "SELL"
            price = trade.price
            pnl = trade.pnl
            pnlcomm = trade.pnlcomm
            self.trade_points.append((time, trade_type, price, pnl, pnlcomm))

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SimpleStrategy)
    cerebro.addanalyzer(MetricsLogger)
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="mysharpe")

    data = bt.feeds.PandasData(
        dataname=get_backtrader_data("BTCUSDT", "1685904000000", "1715904000000", "1d")
    )
    cerebro.adddata(data)

    thestrats = cerebro.run()
    thestrat = thestrats[0]
    print("Sharpe Ratio:", thestrat.analyzers.mysharpe.get_analysis())
