from langchain.callbacks import StreamlitCallbackHandler
from langchain_core.messages import HumanMessage
from utils.summarizer import chain
from utils.agent import agent_chain
import streamlit as st
from yahooquery import Ticker
import yfinance as yf
from datetime import date, timedelta
from datetime import datetime
import finnhub
import requests
import json

finnhub_token = st.secrets['X-FINNHUB-TOKEN']


async def get_stock_symbols(company):
    st_callback = StreamlitCallbackHandler(st.container())
    symbol_list = agent_chain.invoke({"input": [HumanMessage(content=f"{company}")]}, callbacks=[st_callback])
    symbols = symbol_list['output'].split(" ")
    return symbols


async def get_fin_statements(symbol):
    df = Ticker(symbol)
    df1 = df.income_statement().reset_index(drop=True).transpose()
    df2 = df.balance_sheet().reset_index(drop=True).transpose()
    df3 = df.cash_flow().reset_index(drop=True).transpose()
    return df1, df2, df3


async def get_stock_history(symbol):
    curr_day = date.today().strftime("%Y-%m-%d")
    ticker = yf.Ticker(symbol)
    data = ticker.history(start="2019-01-01", end=curr_day)
    return data


async def get_latest_price(data):
    latest_price = data['Close'].iloc[-1]
    return latest_price


def get_news_search(symbol):
    """
        get outsider sentiment
        get news with sentiment score about company with stock symbol passed as input. \n
        news will be used for sentiment analysis for finance & stock trade decisions
        output: json containing the metadata and news
        """
    now = datetime.now() - timedelta(days=30)
    formatted_time = now.strftime("%Y%m%dT%H%M")
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&time_from={formatted_time}&tickers={symbol}&sort=LATEST&apikey={st.secrets["AV_KEY"]}'
    r = requests.get(url)
    data = r.json()

    txt = json.dumps(data)
    return chain.invoke({"topic": "company news with sentiment score", "symbol": symbol, "json": txt})


def insider_sentiment(symbol: str):
    """
  Get json containing the insider sentiment towards the particular company for 3 months time from current date
  input: Stock symbol
  output: json containing the insider sentiments for 3 month time
  """
    today = date.today()
    start = today - timedelta(days=3 * 30)
    finnhub_client = finnhub.Client(api_key=finnhub_token)

    sentiment = finnhub_client.stock_insider_sentiment(symbol, start, today)

    data = json.dumps(sentiment)
    return chain.invoke({"topic": "company insider sentiment", "symbol": symbol, "json": data})


def insider_transactions(symbol: str):
    """
  Get json containing the insider transaction towards the particular company stock for 2 months time from current date
  input: Stock symbol
  output: json containing the insider transactions for 3 month time
  """
    today = date.today()
    start = today - timedelta(days=2 * 30)
    finnhub_client = finnhub.Client(api_key=finnhub_token)

    transactions = finnhub_client.stock_insider_transactions(symbol, start, today)

    data = json.dumps(transactions)
    return chain.invoke({"topic": "company insider stock transactions", "symbol": symbol, "json": data})


async def get_gnews_api_spec(search_term):
    url = f"https://gnews.io/api/v4/search?q={search_term}&token={st.secrets['GNEWS_KEY']}"
    response = requests.get(url)
    news = response.json()
    return news


def get_article_jina(search_url):
    url = f"https://r.jina.ai/{search_url}"
    headers = {
        'Accept': 'application/json',
    }

    response = requests.get(url, headers=headers)
    news = response.json()
    return news
