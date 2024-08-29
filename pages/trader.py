import streamlit as st
from utils.qlearning import update_data


st.set_page_config(
    page_title="Trader",
    page_icon='🦈',
    layout='wide',
    initial_sidebar_state='expanded'
)


st.title("QLearning Trader")

left_column, right_column = st.columns(2)


with left_column:
    st.subheader("Stock Price")
    price_placeholder = st.empty()
    time_placeholder = st.empty()

fig = update_data("TSLA", price_placeholder, 100, 10, 180, 0.1, 0.9, 0.2, time_placeholder)
st.plotly_chart(fig)
