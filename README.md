# trading_backtesting
Example of writing a backtesting framework from scratch.

### Why?
Implementing backtesting is seen as something very complicated, but actually it is not! 
My main motivation for writing this backtesting framework from scratch was flexibility and optimization. 
I found it very difficult to work with existing backtesting for mutli-stocks strategies and they 
were pretty slow. 

### Implementation details
This code implements a general purpose trading environment through `trading_env` in backtesting.py.
This trading enviroment has access to the OHLC price dataframe passed during initialization. You interact with it 
to place buy/sell orders. The enviroment keeps track of all the data during backtesting that is required for computing
the final performance metrics.

The `trade` function in backtesting.py implements my trading strategy which is a multi-stock strategy.

### Usage Details
examples.py shows how to use these to run backtesting on my trading strategy and the benchmark strategy.
