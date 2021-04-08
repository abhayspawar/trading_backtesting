from utils import get_pct_change, get_SR, get_sortino, get_max_drawdown
from datetime import date
from seaborn import lineplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class trading_env():
    """Trading environment which has access to daily price data and
    can be interacted with to place orders"""

    def __init__(self, backtesting_data, start_value=100000,
                 commission=0, slippage=0.1, verbose=False):
        """
        Parameters
        ----------
        backtesting_data : OHLC data for all stocks
        start_value : Starting amount in the account
        commission : Commission percentage
        slippage : Slippage when buying and selling stocks
        verbose : If buying and selling orders should be printed
        """

        # Pre-process the OHLC price data. This is written for daily data
        backtesting_data = backtesting_data.reset_index(drop=True)
        backtesting_data['date'] = backtesting_data['date'].apply(
            lambda x: x.date())

        # Add an index column which is the day_number for each stock
        self.date_index = backtesting_data[['date']].drop_duplicates(
        ).sort_values('date').reset_index(drop=True)
        self.date_index['index'] = self.date_index.index
        self.backtesting_data = backtesting_data.merge(
            self.date_index, on='date', how='left')

        # Initialize various variables to keep track of different things during backtesting
        (self.index, self.curr_buy_number, self.curr_sell_number, self.buy_order_id,
            self.invested, self.sold) = 0, 0, 0, 0, 0, 0
        self.commission, self.slippage = commission*0.01, slippage*0.01
        self.assets, self.orders, self.settled_orders = {}, {}, {}
        self.start_value, self.value, self.cash = start_value, start_value, start_value

        # Track daily quantities to compute metrics later
        (self.daily_cash, self.daily_values, self.daily_investments,
            self.daily_divestments, self.daily_buy_number,
            self.daily_sell_number) = [], [], [], [], [], []
        self.dates, self.indices = [], []
        self.ohlc_cols = ['open', 'high', 'close', 'low']
        self.verbose = verbose

    def get_date_given_index(self, input_index):
        """Returns index (day_number) given date"""
        return self.date_index.loc[input_index, 'date']

    def get_index_given_date(self, input_date):
        """Returns date given index (day_number)"""
        return self.date_index.loc[self.date_index['date'].astype('str') == str(input_date),
                                   'index'].values[0]

    def get_price(self, stock, price_col='open'):
        """ Returns price on current day during backtesting.
        Current day is determined by self.index"""
        return (self.backtesting_data.loc[(self.backtesting_data['stock'] == stock) &
                                          (self.backtesting_data['index']
                                           == self.index),
                                          price_col]).values[0]

    def buy_stocks(self, buy_dict, buy_price=None):
        """Buy stocks in buy_dict. buy_dict has stocks as key and qty as values"""
        for stock in buy_dict:
            # Get price is not passed. Default behavior is to look up price
            if not buy_price:
                buy_price = self.get_price(stock)*(1+self.slippage)

            # Compute commission and update self.cash
            qty = buy_dict[stock]
            total_buy_price = buy_price*qty
            commission_amt = self.commission*total_buy_price
            self.cash = self.cash - commission_amt

            # Make an entry into self.orders and update self.assets
            self.orders[self.buy_order_id] = [
                stock, qty, self.index, buy_price]
            if stock in self.assets:
                self.assets[stock][0] += qty
                self.assets[stock][1] += total_buy_price

            elif qty > 0:
                self.assets[stock] = [qty, total_buy_price]

            if qty > 0:
                # Increment buy_order_id for next buy order
                # Update other variables
                self.buy_order_id += 1
                self.invested += total_buy_price
                self.cash -= total_buy_price
                self.curr_buy_number += 1

            # Your trading code ideally shouldn't place orders which will lead to this
            if self.cash < 0:
                raise ValueError('Cash has become negative!')

            if self.verbose:
                print('Bought', stock)

    def sell_stocks(self, order_ids, sell_price=None):
        """Sell stocks based on buy_order_id."""
        for order_id in order_ids:
            # Get info on stock and qty from buy_order_id
            stock = self.orders[order_id][0]
            qty = self.orders[order_id][1]

            # Get today's open price
            open_price = self.get_price(stock)
            if not sell_price:
                sell_price = open_price*(1-self.slippage)

            # Update self.assets and self.settled_orders
            total_sell_price = sell_price*qty
            left_stock_qty = self.assets[stock][0] - qty

            self.settled_orders[order_id] = self.orders[order_id] + \
                [self.index, sell_price]
            del self.orders[order_id]

            if left_stock_qty == 0:
                del self.assets[stock]
            elif left_stock_qty < 0:
                raise ValueError(
                    'Selling too many stocks than it is possible!')
            elif left_stock_qty > 0:
                self.assets[stock][0] = left_stock_qty
                self.assets[stock][1] = left_stock_qty*open_price

            self.sold += total_sell_price
            self.cash += total_sell_price
            self.curr_sell_number += 1

            if self.verbose:
                print('Sold', stock)

    def get_performance_metrics(self, return_df, row_duration=1):
        return_df['value_prev_row'] = return_df['value'].shift()
        return_df.loc[0, 'value_prev_row'] = self.start_value
        return_df['return_rate'] = get_pct_change(
            return_df['value_prev_row'], return_df['value'])
        return_rate_mean = return_df['return_rate'].mean()
        sharpe_ratio = get_SR(
            return_df['return_rate'].to_list(), row_duration=row_duration)
        sortino_ratio = get_sortino(
            return_df['return_rate'].to_list(), row_duration=row_duration)
        max_dd = get_max_drawdown(self.daily_values)

        return return_df, return_rate_mean, sharpe_ratio, sortino_ratio, max_dd

    def get_todays_value(self, stock, column):
        # Seems redundant with get_price
        cond = (self.backtesting_data['stock'] == stock) & (
            self.backtesting_data['index'] == self.index)
        today_value = self.backtesting_data[cond][column].values[0]

        return today_value

    def daily_wrap_up(self):
        """This function needs to run after placing all orders for the day during backtesting.
            It appends data like today's investments, cash, etc."""
        self.daily_investments.append(self.invested)
        self.daily_divestments.append(self.sold)
        self.daily_cash.append(self.cash)
        self.sold, self.invested = 0, 0

        self.daily_buy_number.append(self.curr_buy_number)
        self.daily_sell_number.append(self.curr_sell_number)
        self.curr_buy_number, self.curr_sell_number = 0, 0

        invested_value = 0
        for stock in self.assets:
            close_price = self.backtesting_data.loc[(self.backtesting_data['stock'] == stock) &
                                                    (self.backtesting_data['index']
                                                     == self.index),
                                                    'close'].values[0]
            self.assets[stock][1] = self.assets[stock][0]*close_price
            invested_value += self.assets[stock][0]*close_price

        self.value = invested_value + self.cash
        self.daily_values.append(self.value)

        curr_date = self.get_date_given_index(self.index)
        self.dates.append(curr_date)
        self.indices.append(self.index)

    def compute_test_results(self, verbose_end_results=True):
        """This function needs to be run at the end of the backtesting period to compute
        all the final metrics"""
        if self.settled_orders != {}:
            settled_orders_df = pd.DataFrame(self.settled_orders).transpose()
            settled_orders_df.columns = [
                'stock', 'qty', 'index_buy', 'buy_price', 'index_sell', 'sell_price']
            settled_orders_df['return'] = (settled_orders_df['sell_price'] -
                                           settled_orders_df['buy_price'])/settled_orders_df['buy_price']
            settled_orders_df['profit'] = settled_orders_df['qty']*(settled_orders_df['sell_price'] -
                                                                    settled_orders_df['buy_price'])
            return_rate_per_trade = settled_orders_df['return'].mean()
            return_amt_per_trade = settled_orders_df['profit'].mean()
        else:
            return_rate_per_trade = 0
            return_amt_per_trade = 0

        daily_return_df = pd.DataFrame()
        daily_return_df['value'] = self.daily_values
        daily_return_df['date'] = self.dates
        daily_return_df['index'] = self.indices
        month_duration = 20
        monthly_return_df = daily_return_df.loc[(daily_return_df['index'] % month_duration == 0) |
                                                (daily_return_df['index'] == daily_return_df['index'].max())]

        # Compute daily and monthly metrics
        daily_return_df, return_rate_daily, \
            sharpe_ratio_daily, sortino_ratio_daily, max_drawdown = self.get_performance_metrics(
                daily_return_df, row_duration=1)
        monthly_return_df, return_rate_monthly, \
            sharpe_ratio_monthly, sortino_ratio_monthly, max_dd_not_required = self.get_performance_metrics(
                monthly_return_df, row_duration=20)

        num_days = len(self.indices)
        num_buys = sum(self.daily_buy_number)
        num_buy_days = len([val for val in self.daily_buy_number if val != 0])
        neg_profit_days = daily_return_df[daily_return_df['return_rate'] <= 0].shape[0]
        pos_profit_days = daily_return_df[daily_return_df['return_rate'] > 0].shape[0]
        self.daily_return_df = daily_return_df
        self.monthly_return_df = monthly_return_df

        perc_profitable_days = pos_profit_days / \
            (pos_profit_days+neg_profit_days)
        perc_trading_days = num_buy_days/num_days
        avg_buys_per_tading_day = num_buys/num_buy_days
        avg_overall_buys_per_day = num_buys/num_days
        full_period_return = get_pct_change(
            self.daily_values[0], self.daily_values[-1])

        # Put all the metrics into a dataframe.
        self.test_results = pd.DataFrame({'start_date': self.dates[0], 'end_date': [self.dates[-1]],
                                          # 'prob_buy_thresh': [prob_buy_thresh], 'sell_threshold': [sell_threshold],
                                          # 'stoploss_threshold': [stoploss_threshold], 'sell_window': [window],
                                          'buy_trades': [num_buys], 'total_days': [num_days], 'neg_profit_days': [neg_profit_days],
                                          'pos_profit_days': [pos_profit_days],
                                          'perc_profitable_days': [perc_profitable_days],
                                          'num_days_buy': [num_buy_days],
                                          'perc_days_buy': [perc_trading_days],
                                          'buys_per_tading_day': [avg_buys_per_tading_day],
                                          'overall_buys_per_day': [avg_overall_buys_per_day],
                                          'return_per_trade': [return_rate_per_trade],
                                          'avg_daily_return': [return_rate_daily],
                                          'avg_monthly_return': [return_rate_monthly],
                                          'full_period_return': [full_period_return],
                                          'sharpe_ratio_daily': [sharpe_ratio_daily],
                                          'sortino_ratio_daily': [sortino_ratio_daily],
                                          'sharpe_ratio_monthly': [sharpe_ratio_monthly],
                                          'sortino_ratio_monthly': [sortino_ratio_monthly],
                                          'max_drawdwon': [max_drawdown]
                                          })

        # Print all the plots to visualize what the trading strategy did
        if verbose_end_results:
            print('Avg investments daily', np.mean(self.daily_investments))
            print('Duration in years:', num_days/245)
            print('Returns per trade %1.4f' % return_rate_per_trade)
            print('Avg monthly return:', return_rate_monthly)  # per sell trade
            print('Full period return:', full_period_return)
            print(month_duration, 'day Sharpe Ratio:', sharpe_ratio_monthly)
            print(month_duration, 'day Sortino Ratio:', sortino_ratio_monthly)
            print('Max drawdown:', max_drawdown)

            plt.figure()
            plt.title('Daily cash')
            lineplot(x=range(len(self.daily_cash)), y=self.daily_cash)
            plt.show()

            plt.figure()
            plt.title('Daily investments')
            lineplot(x=range(len(self.daily_investments)),
                     y=self.daily_investments)
            plt.show()

            profits_daily = (
                daily_return_df['value']-daily_return_df['value_prev_row']).to_list()
            plt.title('Daily profits')
            lineplot(x=range(len(profits_daily)), y=profits_daily)
            plt.show()

            plt.figure()
            plt.title('Value of portfolio')
            lineplot(x=range(len(self.daily_values)), y=self.daily_values)
            plt.show()

            plt.figure()
            plt.title('Avg. daily returns')
            lineplot(
                x=range(len(daily_return_df['return_rate'])), y=daily_return_df['return_rate'])
            plt.show()

            plt.figure()
            plt.title('Avg. monthly returns')
            lineplot(
                x=range(len(monthly_return_df['return_rate'])), y=monthly_return_df['return_rate'])
            plt.show()

        return self.test_results


def trade(env, period=None, wait_period=10, sell_thresh=0.1, stop_loss_thresh=0.1,
          min_prob_buy=0.25, n_stocks_per_day=5, single_stock_exposure=10000, min_preds_rank=10,
          daily_limit=20000, verbose=False):
    """Function that interacts with above trading environment
    and actually implements the trading strategy"""
    if not period:
        period = len(env.date_index)

    # Iterate over all the days in backtesting price data
    for index in range(period):
        env.index = index

        # Check if any of the bought stocks need to be sold
        sell_orders = []
        curr_orders = env.orders.copy()
        for order in curr_orders:
            stock = curr_orders[order][0]
            buy_price = curr_orders[order][3]

            # Check if they hit the stop_loss or take profit limits. If yes, then sell
            if index > 0:
                stock_index_cond = ((env.backtesting_data['index'] == index-1)
                                    & (env.backtesting_data['stock'] == stock))
                min_price = env.backtesting_data[stock_index_cond][[
                    'open', 'close']].min(axis=1).values[0]

                stock_index_cond = (
                    env.backtesting_data['index'] == index-1) & (env.backtesting_data['stock'] == stock)
                max_price = env.backtesting_data[stock_index_cond][[
                    'open', 'close']].max(axis=1).values[0]

                if min_price < (1-stop_loss_thresh)*buy_price:
                    env.sell_stocks([order], sell_price=(
                        1-stop_loss_thresh)*buy_price)
                    sell_orders.append(order)
                    if verbose:
                        print('Sold at stop loss')
                elif max_price >= (1+sell_thresh)*buy_price:
                    env.sell_stocks([order], sell_price=(
                        1+sell_thresh)*buy_price)
                    sell_orders.append(order)
                    if verbose:
                        print('Sold at threshold profit')

            # Check if bought stocks are at the end of hold period. If yes, sell.
            if curr_orders[order][2]+wait_period == index and order not in sell_orders:
                env.sell_stocks([order])
                if verbose:
                    print('Sold at end of period')

        # Code to determine today's investment based on strategy
        num_last_10_day_invested = wait_period - sum([1 for val in env.daily_investments[-wait_period:]
                                                      if val == 0])

        if num_last_10_day_invested < 4:
            num_last_10_day_invested = 4

        todays_investment = (env.value/num_last_10_day_invested)*1.25

        if todays_investment > env.cash:
            todays_investment = env.cash
        elif todays_investment > daily_limit:
            todays_investment = daily_limit

        high_investment_stocks = [
            key for key in env.assets if env.assets[key][1] > single_stock_exposure]
        eligible_stock_cond = (env.backtesting_data['index'] == index-1) & (env.backtesting_data['y_pred'] >= min_prob_buy)\
            & (~env.backtesting_data['stock'].isin(high_investment_stocks))
        eligible_stocks = env.backtesting_data[eligible_stock_cond].sort_values(
            'y_pred', ascending=False)
        n_stocks_today = min(n_stocks_per_day, len(eligible_stocks))
        stock_bought = 0
        stock_rank = 0

        # Buy stocks
        for stock in eligible_stocks['stock'].to_list():
            if stock_rank >= min_preds_rank:
                break
            today_price = env.get_todays_value(stock, 'open')
            if stock_bought < n_stocks_today:
                investment_per_stock = todays_investment / \
                    (n_stocks_today-stock_bought)
            else:
                investment_per_stock = todays_investment

            actual_today_price = today_price*(1+env.commission+env.slippage)
            num_to_buy = int(investment_per_stock/actual_today_price)
            if num_to_buy > 0:
                env.buy_stocks({stock: num_to_buy})

                stock_bought += 1
                todays_investment -= num_to_buy*actual_today_price
                if verbose:
                    print(stock, actual_today_price, num_to_buy,
                          todays_investment, env.cash)

            stock_rank += 1

        # buy list:
        # 1. top stocks above a threshold
        # 2. decide how much to invest today
        # 3. go through stocks one by one and invest today_amt/n value
        # 4. skip above if threshold amt invested in that stock
        # 5. stop when top_amt is exhausted

        # Append metrics for the day in the trading environment
        env.daily_wrap_up()
    return env


def trade_parallelize(inputs):
    """Wrapper around trade() for easy parallelization when testing for different sets of parameters"""
    # for HPT of trading parameters
    wait_period = inputs[0]
    sell_thresh = inputs[1]
    stop_loss_thresh = inputs[2]
    min_prob_buy = inputs[3]
    n_stocks_per_day = inputs[4]
    single_stock_exposure = inputs[5]
    min_preds_rank = inputs[6]
    data_bt = inputs[7]

    env = trading_env(data_bt, commission=0, slippage=0.1)
    period = len(env.date_index)  # len(env.date_index)

    trade(env, period=period, wait_period=wait_period, sell_thresh=sell_thresh,
          stop_loss_thresh=stop_loss_thresh, min_prob_buy=min_prob_buy, n_stocks_per_day=n_stocks_per_day,
          single_stock_exposure=single_stock_exposure, min_preds_rank=min_preds_rank, verbose=False)  # slippage

    results = env.compute_test_results(verbose_end_results=False)
    results['wait_period'] = inputs[0]
    results['sell_thresh'] = inputs[1]
    results['stop_loss_thresh'] = inputs[2]
    results['min_prob_buy'] = inputs[3]
    results['n_stocks_per_day'] = inputs[4]
    results['single_stock_exposure'] = inputs[5]
    results['min_preds_rank'] = inputs[6]

    return results


def trade_benchmark(env, period=None, wait_period=10, sell_thresh=0.1, stop_loss_thresh=0.1,
                    min_prob_buy=0.25, n_stocks_per_day=5, single_stock_exposure=10000, min_preds_rank=10,
                    daily_limit=20000, verbose=False):
    """Benchmark trading strategy of buying and holding till end of period"""
    if not period:
        period = len(env.date_index)

    for index in range(period):
        env.index = index

        if index == 0:
            todays_data = env.backtesting_data[env.backtesting_data['index'] == index].sort_values(
                'open', ascending=False)
            num_stocks_left = len(todays_data)
            for stock in todays_data['stock'].to_list():
                cash_left = env.cash
                per_stock_amt = (cash_left/num_stocks_left)
                open_price = todays_data[todays_data['stock']
                                         == stock]['open'].values[0]
                num_to_buy = int(per_stock_amt/open_price)

                # if num_to_buy>0:
                env.buy_stocks({stock: num_to_buy})
                num_stocks_left -= 1

        env.daily_wrap_up()
