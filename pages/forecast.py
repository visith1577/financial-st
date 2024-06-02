import asyncio

import streamlit as st
from datetime import datetime

from langchain_core.messages import ChatMessage
from torchview import draw_graph
import plotly.graph_objects as go

from utils.data import data_on_percent, fetch_data, normalized_data
from utils.model import Model, train_model, launch_model, graph_preds
from utils.tools import get_stock_symbols, get_stock_history

st.set_page_config(
    page_title="Forecasting AI",
    page_icon='🦈',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title("Forecasting AI")
agree = st.checkbox('Show Model')


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
            left_column, right_column = st.columns(2)
            with left_column:
                st.header(symbol)
            with right_column:
                plot_placeholder = st.empty()
            with left_column:
                plot_placeholder_daily = st.empty()
                price_pred_placeholder = st.empty()
            with right_column:
                table_placeholder_daily = st.empty()
            st.markdown("""---""")

            df = await get_stock_history(symbol)
            stock_data = await fetch_data(df, symbol)

            num_data_points = len(stock_data[0])

            feg = go.Figure(data=[
                go.Candlestick(x=stock_data[0], open=stock_data[1], high=stock_data[2], low=stock_data[3],
                               close=stock_data[4])])
            feg.update_layout(title_text="Full Data")
            feg.update_layout(xaxis_rangeslider_visible=False)
            plot_placeholder.plotly_chart(feg, use_container_width=True)

            data_normalized, scaler = await normalized_data(stock_data)
            unseen_data, dataset_train, dataset_val, split_index = await data_on_percent(data_normalized,
                                                                                         percent=0.9
                                                                                         )
            net_in_out = dataset_val.y.shape[1]

            model = Model(input_size=net_in_out, hidden_layer_size=32, num_layers=2, output_size=net_in_out,
                          dropout=0.2)

            if agree:
                with st.sidebar:
                    with st.expander(symbol):
                        model_graph = draw_graph(model, input_size=dataset_train.x.shape, expand_nested=True)
                        gp = model_graph.visual_graph.render(format='svg')
                        st.image(gp)
                        model_graph.resize_graph(scale=5.0)
                        model_graph.visual_graph.render(format='svg')

            trained_model = await train_model(model, dataset_train, dataset_val)

            predicted_train, predicted_val, predicted_day = await launch_model(trained_model, dataset_train,
                                                                               dataset_val, unseen_data,
                                                                               pred_days=5)
            fag, fog, FinalPred = await graph_preds(stock_data[0],
                                                    num_data_points,
                                                    predicted_train,
                                                    predicted_val,
                                                    predicted_day,
                                                    stock_data[4],
                                                    scaler,
                                                    split_index,
                                                    pred_days=5)

            table_placeholder_daily.plotly_chart(fag, use_container_width=True)
            plot_placeholder_daily.plotly_chart(fog, use_container_width=True)
            price_pred_placeholder.dataframe(FinalPred)


if __name__ == "__main__":
    asyncio.run(run())
