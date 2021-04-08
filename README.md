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

### Some random thoughts
I wouldn't recommend using this code as is for your backtesting purpose. I would definitely try out existing
frameworks and if they don't work, only then I would go for writing one from scratch like this. I landed on this piece of
code after going through several iterations. I tried out several ways to optimize the code by vectorization and
parallelization. But, ultimately the simple idea of iterating through each trading day worked the best. The final
piece of code doesn't seem that complicated, but my thought process took a fairly long and winding road to reach there :D.

It was fun though! And that is what matters the most.
