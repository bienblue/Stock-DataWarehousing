SELECT *,
    CASE WHEN ISNULL(LAG([Close]) OVER (PARTITION BY Ticker ORDER BY DateKey ASC), 0) = 0 THEN 0.0
         ELSE ([Close] - LAG([Close]) OVER (PARTITION BY Ticker ORDER BY DateKey ASC)) / LAG([Close]) OVER (PARTITION BY Ticker ORDER BY DateKey ASC)
    END AS percent_change
INTO Trading
FROM fact_trade_history
