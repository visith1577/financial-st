import numpy as np
import streamlit as st
from datetime import datetime
from langchain_core.messages import ChatMessage
from utils.tools import get_stock_symbols, get_stock_history
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio

st.set_page_config(
    page_title="Technical Analysis Report",
    page_icon='🦈',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title("Technical Analytics")


async def run():
    date_now = datetime.now()
    date_year = date_now.year
    date_month = date_now.month
    date_day = date_now.day
    date_day_ = date_now.strftime("%A")

    date_d = "{}-{}-{}".format(date_year, date_month, date_day)

    st.subheader(f" _{date_d}_")
    st.subheader(f" :orange[_{date_day_}_]", divider='rainbow')

    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="")]

    with st.form(key='company_search_form'):
        company_name = st.text_input("Enter a company name:")
        submit_button = st.form_submit_button("Search", type="primary")

    if submit_button and company_name:
        symbols = await get_stock_symbols(company_name)

        for symbol in symbols:
            left_column, _ = st.columns(2)
            with left_column:
                st.header(symbol)
            with st.expander(f"{symbol} :: Technical Analysis", expanded=True):
                plot_placeholder = st.empty()
                st.markdown("""---""")

            df = await get_stock_history(symbol)
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=20, append=True)

            data_date = df.index.to_numpy()
            data_open_price = df['Open'].to_numpy()
            data_close_price = df['Close'].to_numpy()
            data_high_price = df['High'].to_numpy()
            data_low_price = df['Low'].to_numpy()

            # indicators
            rsi = df['RSI_14'].to_numpy()
            macd = df['MACD_12_26_9'].to_numpy()
            macdh = df['MACDh_12_26_9'].to_numpy()
            macds = df['MACDs_12_26_9'].to_numpy()
            bbl = df['BBL_20_2.0'].to_numpy()
            bbm = df['BBM_20_2.0'].to_numpy()
            bbu = df['BBU_20_2.0'].to_numpy()

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                                row_heights=[0.8, 0.1, 0.1, 0.1])
            fig.add_trace(go.Candlestick(
                x=data_date,
                open=data_open_price,
                high=data_high_price,
                low=data_low_price,
                close=data_close_price,
                name='candlestick',
            ), row=1, col=1)

            fig.add_trace(
                go.Scatter(x=data_date, y=bbl, mode='lines', name='BBL', line=dict(color='red'), legendgroup='1'),
                row=1, col=1)
            fig.add_trace(go.Scatter(x=data_date, y=bbm, mode='lines', name='BBM', line=dict(color='pink', width=4),
                                     legendgroup='1'), row=1, col=1)
            fig.add_trace(
                go.Scatter(x=data_date, y=bbu, mode='lines', name='BBU', line=dict(color='blue'), legendgroup='1'),
                row=1, col=1)

            high_line = np.ones(data_date.shape[0]) * 70
            low_line = np.ones(data_date.shape[0]) * 30

            fig.add_trace(
                go.Scatter(x=data_date, y=high_line, mode='lines', name='70%', line=dict(width=1, dash='dash'),
                           legendgroup='3'),
                row=3, col=1)

            fig.add_trace(go.Scatter(x=data_date, y=rsi, mode='lines', name='RSI',
                                     legendgroup='3'), row=3, col=1)
            fig.add_trace(go.Scatter(x=data_date, y=low_line, mode='lines', name='30%',
                                     line=dict(color='green', width=1, dash='dash'),
                                     legendgroup='3'), row=3, col=1)

            fig.add_trace(go.Scatter(x=data_date, y=macd, mode='lines', name='MACD', line=dict(width=2),
                                     legendgroup='4'),
                          row=4, col=1)
            fig.add_trace(go.Scatter(x=data_date, y=macds, mode='lines', name='signal', line=dict(width=2),
                                     legendgroup='4'),
                          row=4, col=1)
            colors = np.where(macdh > 0, 'green', 'red')
            fig.add_trace(go.Bar(x=data_date, y=macdh, name='MACD histogram', marker={
                'color': colors
            }, legendgroup='4'), row=4, col=1)

            fig.update_layout(xaxis_rangeslider_visible=False, title_text='Stock Price Chart', title_x=0.5, height=1000)
            plot_placeholder.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    asyncio.run(run())
