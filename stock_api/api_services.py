import pandas as pd
import requests
from pandas import json_normalize
from datetime import datetime, timedelta
import time
from io import BytesIO
import json


headers = {
        'Connection': 'keep-alive',
        'sec-ch-ua': '"Not A;Brand";v="99", "Chromium";v="98", "Google Chrome";v="98"',
        'DNT': '1',
        'sec-ch-ua-mobile': '?0',
        'X-Fiin-Key': 'KEY',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Fiin-User-ID': 'ID',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
        'X-Fiin-Seed': 'SEED',
        'sec-ch-ua-platform': 'Windows',
        'Origin': 'https://iboard.ssi.com.vn',
        'Sec-Fetch-Site': 'same-site',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
        'Referer': 'https://iboard.ssi.com.vn/',
        'Accept-Language': 'en-US,en;q=0.9,vi-VN;q=0.8,vi;q=0.7'
        }

def api_request(url, headers=headers):
    r = requests.get(url, headers).json()
    return r

def ticker_overview (symbol):
    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/ticker/{}/overview'.format(symbol)).json()
    df = json_normalize(data)
    return df

## STOCK TRADING HISTORICAL DATA
def stock_historical_data (symbol, start_date, end_date):
    fd = int(time.mktime(time.strptime(start_date, "%Y-%m-%d")))
    td = int(time.mktime(time.strptime(end_date, "%Y-%m-%d")))
    data = requests.get('https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={}&type=stock&resolution=D&from={}&to={}'.format(symbol, fd, td)).json()
    df = json_normalize(data['data'])
    if df.empty:
        return df
    df['tradingDate'] = pd.to_datetime(df.tradingDate.str.split("T", expand=True)[0])
    df.columns = df.columns.str.title()
    df.rename(columns={'Tradingdate':'TradingDate'}, inplace=True)
    return df


# TRADING INTELLIGENT
today_val = datetime.now()

def today():
    today = today_val.strftime('%Y-%m-%d')
    return today

def last_xd (day_num): # return the date of last x days
    last_xd = (today_val - timedelta(day_num)).strftime('%Y-%m-%d')
    return last_xd

def start_xm (period): # return the start date of x months
    date = pd.date_range(end=today, periods=period+1, freq='MS')[0].strftime('%Y-%m-%d')
    return date

def stock_intraday_data (symbol, page_num, page_size):
    d = datetime.now()
    if d.weekday() > 4: #today is weekend
        data = requests.get('https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/{}/his/paging?page={}&size={}&headIndex=-1'.format(symbol, page_num, page_size)).json()
    else: #today is weekday
        data = requests.get('https://apipubaws.tcbs.com.vn/stock-insight/v1/intraday/{}/his/paging?page={}&size={}'.format(symbol, page_num, page_size)).json()
    df = json_normalize(data['data']).rename(columns={'p':'price', 'v':'volume', 't': 'time'})
    return df

# COMPANY OVERVIEW
def company_overview (symbol):
    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/ticker/{}/overview'.format(symbol)).json()
    df = json_normalize(data)
    return df


# FINANCIAL REPORT
def financial_report (symbol, report_type, frequency, headers=headers): # Quarterly, Yearly
    url = 'https://fiin-fundamental.ssi.com.vn/FinancialStatement/Download{}?language=vi&OrganCode={}&Skip=0&Frequency={}'.format(report_type, symbol, frequency)
    r = requests.get(url, headers=headers)
    df = pd.read_excel(BytesIO(r.content), skiprows=7).dropna()
    return df

def financial_ratio_compare (symbol_ls, industry_comparison, frequency, start_year, headers=headers): 
    global timeline
    if frequency == 'Yearly':
      timeline = str(start_year) + '_5'
    elif frequency == 'Quarterly':
      timeline = str(start_year) + '_4'

    for i in range(len(symbol_ls)):
      if i == 0:
        company_join = '&CompareToCompanies={}'.format(symbol_ls[i])
        url = 'https://fiin-fundamental.ssi.com.vn/FinancialAnalysis/DownloadFinancialRatio2?language=vi&OrganCode={}&CompareToIndustry={}{}&Frequency={}&Ratios=ryd21&Ratios=ryd25&Ratios=ryd14&Ratios=ryd7&Ratios=rev&Ratios=isa22&Ratios=ryq44&Ratios=ryq14&Ratios=ryq12&Ratios=rtq51&Ratios=rtq50&Ratios=ryq48&Ratios=ryq47&Ratios=ryq45&Ratios=ryq46&Ratios=ryq54&Ratios=ryq55&Ratios=ryq56&Ratios=ryq57&Ratios=nob151&Ratios=casa&Ratios=ryq58&Ratios=ryq59&Ratios=ryq60&Ratios=ryq61&Ratios=ryd11&Ratios=ryd3&TimeLineFrom={}'.format(symbol_ls[i], industry_comparison, '', frequency, timeline)
      elif i > 0:
        company_join = '&'.join([company_join, 'CompareToCompanies={}'.format(symbol_ls[i])])
        url = 'https://fiin-fundamental.ssi.com.vn/FinancialAnalysis/DownloadFinancialRatio2?language=vi&OrganCode={}&CompareToIndustry={}{}&Frequency={}&Ratios=ryd21&Ratios=ryd25&Ratios=ryd14&Ratios=ryd7&Ratios=rev&Ratios=isa22&Ratios=ryq44&Ratios=ryq14&Ratios=ryq12&Ratios=rtq51&Ratios=rtq50&Ratios=ryq48&Ratios=ryq47&Ratios=ryq45&Ratios=ryq46&Ratios=ryq54&Ratios=ryq55&Ratios=ryq56&Ratios=ryq57&Ratios=nob151&Ratios=casa&Ratios=ryq58&Ratios=ryq59&Ratios=ryq60&Ratios=ryq61&Ratios=ryd11&Ratios=ryd3&TimeLineFrom=2017_5'.format(symbol_ls[i], industry_comparison, company_join, frequency, timeline)
    r = requests.get(url, headers=headers)
    df = pd.read_excel(BytesIO(r.content), skiprows=7)
    return df


# STOCK FILTERING

def financial_ratio (symbol, report_range, is_all): #TCBS source
    if report_range == 'yearly':
        x = 1
    elif report_range == 'quarterly':
        x = 0
    
    if is_all == True:
      y = 'true'
    else:
      y = 'false'

    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{}/financialratio?yearly={}&isAll={}'.format(symbol, x, y)).json()
    df = json_normalize(data)
    return df

def financial_flow(symbol, report_type, report_range): # incomestatement, balancesheet, cashflow | report_range: 0 for quarterly, 1 for yearly
    if report_range == 'yearly':
        x = 1
    elif report_range == 'quarterly':
        x = 0
    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/finance/{}/{}'.format(symbol, report_type), params={'yearly': x, 'isAll':'true'}).json()
    df = json_normalize(data)
    df[['year', 'quarter']] = df[['year', 'quarter']].astype(str)
    df['index'] = df['year'].str.cat('-Q' + df['quarter'])
    df = df.set_index('index').drop(columns={'year', 'quarter'})
    return df

def dividend_history (symbol):
    """
    This function returns the dividend historical data of the seed stock symbol.
    Args:
        symbol (:obj:`str`, required): 3 digits name of the desired stock.
    """
    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/company/{}/dividend-payment-histories?page=0&size=20'.format(symbol)).json()
    df = json_normalize(data['listDividendPaymentHis']).drop(columns=['no', 'ticker'])
    return df


## STOCK RATING
def  general_rating (symbol):
    """
    This function returns a dataframe with all rating aspects for the desired stock symbol.
    Args:
        symbol (:obj:`str`, required): 3 digits name of the desired stock.
    """
    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/rating/{}/general?fType=TICKER'.format(symbol)).json()
    df = json_normalize(data).drop(columns='stockRecommend')
    return df

def biz_model_rating (symbol):
    """
    This function returns the business model rating for the desired stock symbol.
    Args:
        symbol (:obj:`str`, required): 3 digits name of the desired stock.
    """
    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/rating/{}/business-model?fType=TICKER'.format(symbol)).json()
    df = json_normalize(data)
    return df

def biz_operation_rating (symbol):
    """
    This function returns the business operation rating for the desired stock symbol.
    Args:
        symbol (:obj:`str`, required): 3 digits name of the desired stock.
    """
    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/rating/{}/business-operation?fType=TICKER'.format(symbol)).json()
    df = json_normalize(data)
    return df

def financial_health_rating (symbol):
    """
    This function returns the financial health rating for the desired stock symbol.
    Args:
        symbol (:obj:`str`, required): 3 digits name of the desired stock.
    """
    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/rating/{}/financial-health?fType=TICKER'.format(symbol)).json()
    df = json_normalize(data)
    return df


def valuation_rating (symbol):
    """
    This function returns the valuation rating for the desired stock symbol.
    Args:
        symbol (:obj:`str`, required): 3 digits name of the desired stock.
    """
    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/rating/{}/valuation?fType=TICKER'.format(symbol)).json()
    df = json_normalize(data)
    return df


def industry_financial_health (symbol):
    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/rating/{}/financial-health?fType=INDUSTRY'.format(symbol)).json()
    df = json_normalize(data)
    return df

## STOCK COMPARISON

def industry_analysis (symbol):
    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/rating/detail/council?tickers={}&fType=INDUSTRIES'.format(symbol)).json()
    df = json_normalize(data)
    data1 = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/rating/detail/single?ticker={}&fType=TICKER'.format(symbol)).json()
    df1 = json_normalize(data1)
    df = pd.concat([df1, df]).reset_index(drop=True)
    return df

def stock_ls_analysis (symbol_ls):
    data = requests.get('https://apipubaws.tcbs.com.vn/tcanalysis/v1/rating/detail/council?tickers={}&fType=TICKERS'.format(symbol_ls)).json()
    df = json_normalize(data).dropna(axis=1)
    return df

## MARKET WATCH

def market_top_mover (report_name): #Value, Losers, Gainers, Volume, ForeignTrading, NewLow, Breakout, NewHigh
    ls1 = ['Gainers', 'Losers', 'Value', 'Volume']
    # ls2 = ['ForeignTrading', 'NewLow', 'Breakout', 'NewHigh']
    if report_name in ls1:
        url = 'https://fiin-market.ssi.com.vn/TopMover/GetTop{}?language=vi&ComGroupCode=All'.format(report_name)
    elif report_name == 'ForeignTrading':
        url = 'https://fiin-market.ssi.com.vn/TopMover/GetTopForeignTrading?language=vi&ComGroupCode=All&Option=NetBuyVol'
    elif report_name == 'NewLow':
        url = 'https://fiin-market.ssi.com.vn/TopMover/GetTopNewLow?language=vi&ComGroupCode=All&TimeRange=ThreeMonths'
    elif report_name == 'Breakout':
        url = 'https://fiin-market.ssi.com.vn/TopMover/GetTopBreakout?language=vi&ComGroupCode=All&TimeRange=OneWeek&Rate=OnePointFive'
    elif report_name == 'NewHigh':
        url = 'https://fiin-market.ssi.com.vn/TopMover/GetTopNewHigh?language=vi&ComGroupCode=All&TimeRange=ThreeMonths'
    r = api_request(url)
    df = pd.DataFrame(r['items'])
    return df


def fr_trade_heatmap (exchange, report_type): 
    url = 'https://fiin-market.ssi.com.vn/HeatMap/GetHeatMap?language=vi&Exchange={}&Criteria={}'.format(exchange, report_type)
    r = api_request(url)
    concat_ls = []
    for i in range(len(r['items'])):
        for j in range(len(r['items'][i]['sectors'])):
            name = r['items'][i]['sectors'][j]['name']
            rate = r['items'][i]['sectors'][j]['rate']
            df = json_normalize(r['items'][i]['sectors'][j]['tickers'])
            df['industry_name'] = name
            df['rate'] = rate
            concat_ls.append(df)
    combine_df = pd.concat(concat_ls)
    return combine_df

# GET MARKET IN DEPT DATA - INDEX SERIES

def get_index_series(index_code='VNINDEX', time_range='OneYear', headers=headers):
    url = f"https://fiin-market.ssi.com.vn/MarketInDepth/GetIndexSeries?language=vi&ComGroupCode={index_code}&TimeRange={time_range}&id=1"
    payload={}
    response = requests.request("GET", url, headers=headers, data=payload)
    result = json_normalize(response.json()['items'])
    return result

def get_latest_indices(headers=headers):
    """
    Retrieve the latest indices values
    """
    url = "https://fiin-market.ssi.com.vn/MarketInDepth/GetLatestIndices?language=vi&pageSize=999999&status=1"
    payload={}
    response = requests.request("GET", url, headers=headers, data=payload)
    result = json_normalize(response.json()['items'])
    return result



