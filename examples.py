from backtesting import trading_env, trade, trade_benchmark
from utils import get_ohlc_data

wait_period = 15
sell_thresh = 0.2
stop_loss_thresh = 0.2
min_prob_buy = 0.4
n_stocks_per_day = 5
single_stock_exposure = 6000
min_preds_rank = 10

data_bt = get_ohlc_data()

# Define trading enviroment which has access to the price data through data_bt
env = trading_env(data_bt, commission=0, slippage=0.1)
period = len(env.date_index)

# trade implements a specific trading strategy. It interacts with env (trading environment)
# place buy/sell orders.
trade(env, period = None, wait_period = wait_period, sell_thresh = sell_thresh, 
      stop_loss_thresh = stop_loss_thresh, min_prob_buy = min_prob_buy, n_stocks_per_day = n_stocks_per_day, 
      single_stock_exposure = single_stock_exposure, min_preds_rank = min_preds_rank, verbose=False)

# Compute the performance metrics
env.compute_test_results()

# For running the benchmark strategy of buy and hold
env = trading_env(data_bt, commission=0, slippage=0.0)
trade_benchmark(env)
env.compute_test_results()
