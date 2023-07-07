#%%
from api_services import *
import pandas as pd
import numpy as np
import os 
from mssql_services import *

#%%
# companies = pd.read_excel('convertcsv.xlsx')
# companies_overviews = pd.DataFrame()
# for ticker in companies.ticker:
#     _information = ticker_overview(ticker)
#     if companies_overviews.empty:
#         companies_overviews = _information
#     else:
#         companies_overviews = pd.concat([companies_overviews, _information], axis=0)
#         companies_overviews = companies_overviews.reset_index(drop=True)
#     print(companies_overviews.shape)


# companies_overviews = pd.merge(companies, companies_overviews, on='ticker', how='inner')
# companies_overviews.to_csv('companies_overviews.csv', index=False, encoding='utf-8-sig')

#%%
# companies = pd.read_csv('companies_overviews.csv')
# historical_stocks = pd.DataFrame()
# for _, company in companies.iterrows():
#     ticker = company['ticker']
#     establishedYear = company['establishedYear']
#     print(f'{_}: {ticker} - {establishedYear}')
#     if establishedYear >= 2000 or np.isnan(establishedYear):
#         continue
#     _historical_stock = stock_historical_data(ticker, '2013-01-01', '2022-12-31')
#     if _historical_stock.empty or _historical_stock.shape[0] < 2400:
#         continue
#     print(_historical_stock.shape)
#     _historical_stock['ticker'] = ticker
#     if historical_stocks.empty:
#         historical_stocks = _historical_stock
#     else:
#         historical_stocks = pd.concat([historical_stocks, _historical_stock], axis=0)
#         historical_stocks = historical_stocks.reset_index(drop=True)

# historical_stocks.to_csv('historical_stocks.csv', index=False, encoding='utf-8-sig')

# %%
# tickers = pd.read_csv('historical_stocks.csv').ticker.unique()
# _x = tickers

# financial_ratios = pd.DataFrame()
# for ticker in tickers:
#     _financial_ratio = financial_ratio(ticker, 'quarterly', True)
#     if _financial_ratio.empty:
#         continue
#     _financial_ratio['ticker'] = ticker
#     print(f'{ticker} {_financial_ratio.shape}')
#     _financial_ratio = _financial_ratio[(_financial_ratio['year'] >= 2013) & (_financial_ratio['year'] <= 2022)]
#     if financial_ratios.empty:
#         financial_ratios = _financial_ratio
#     else:
#         financial_ratios = pd.concat([financial_ratios, _financial_ratio], axis=0)
#         financial_ratios = financial_ratios.reset_index(drop=True)

# financial_ratios = financial_ratios[['ticker', 'quarter', 'year', 'priceToEarning', 'priceToBook', 'roe', 'roa', 'earningPerShare', 'bookValuePerShare', 'equityOnTotalAsset', 'equityOnLiability', 'epsChange', 'bookValuePerShareChange']]
# print(financial_ratios.shape)
# financial_ratios.to_csv('financial_ratios.csv', index=False, encoding='utf-8-sig')

# %%
files = os.listdir()
for file in files:
    if file.endswith('.csv'):
        df = pd.read_csv(file)
        create_table(file, df)
