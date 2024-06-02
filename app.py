import time

import streamlit as st
from datetime import datetime
from langchain_core.messages import ChatMessage
import asyncio
import unstructured_client
from unstructured_client.models import shared
from utils.custom_callback import StreamHandler

from utils.tools import get_stock_symbols, get_gnews_api_spec, get_fin_statements, get_latest_price, get_stock_history
from utils.chat import get_response

client = unstructured_client.UnstructuredClient(
    api_key_auth=st.secrets['UNSTRUCTURED_API_KEY'],
    # you may need to provide your unique API URL
    # server_url="YOUR_API_URL",
)

st.set_page_config(
    page_title="FinGPT",
    page_icon='🦈',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title("finChat")


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
        articles_string = ''
        financial_statements = ''
        symbols = await get_stock_symbols(company_name)

        for symbol in symbols:
            with st.spinner(f"Fetching data for {company_name}"):
                gnews = await get_gnews_api_spec(symbol)
                try:
                    income_stmt, balance_sheet, cash_flow = await get_fin_statements(symbol)
                    # Save income statement as CSV
                    income_stmt.to_csv('income_stmt.csv')

                    filename = "income_stmt.csv"
                    with open(filename, "rb") as file:
                        file_content = shared.Files(
                            content=file.read(),
                            file_name=filename,
                        )

                    income_stmt_str = shared.PartitionParameters(files=file_content).files.content

                    # Save balance sheet as CSV
                    balance_sheet.to_csv('balance_sheet.csv')

                    filename = "balance_sheet.csv"
                    with open(filename, "rb") as file:
                        file_content = shared.Files(
                            content=file.read(),
                            file_name=filename,
                        )

                    balance_sheet_str = shared.PartitionParameters(files=file_content).files.content

                    # Save cash flow as CSV
                    cash_flow.to_csv('cash_flow.csv')

                    filename = "cash_flow.csv"
                    with open(filename, "rb") as file:
                        file_content = shared.Files(
                            content=file.read(),
                            file_name=filename,
                        )

                    cash_flow_str = shared.PartitionParameters(files=file_content).files.content

                    financial_statements += f"**{symbol}**\n\n"
                    stock_history = await get_stock_history(symbol)
                    latest_price = await get_latest_price(stock_history)
                    financial_statements += f"**Latest Price**: ${latest_price}\n\n"
                    financial_statements += f"**Income Statement**\n\n"
                    financial_statements += f"{income_stmt_str}\n\n"
                    financial_statements += f"**Balance Sheet**\n\n"
                    financial_statements += f"{balance_sheet_str}\n\n"
                    financial_statements += f"**Cash Flow**\n\n"
                    financial_statements += f"{cash_flow_str}\n\n"
                except:
                    st.subheader(f":red[Financial Statement for _{symbol}_ couldn't be found 😥️]")

                with st.sidebar:
                    with st.expander(symbol):
                        st.subheader("News from GNews API", divider='rainbow')
                        for article in gnews['articles']:
                            st.write(f"**Title:** {article['title']}")
                            st.write(f"**Description:** {article['description']}")
                            st.write(f"**URL:** {article['url']}")
                            st.markdown("""---""")
                            article_string = f"**Title:** {article['title']}, **Description:** {article['description']} \n"
                            articles_string += article_string + "\n"

            time.sleep(1)

            stream_handler = StreamHandler(st.empty(), initial_text="Analysing....")
            with st.chat_message("assistant"):
                user_input = """
                Please, give me an exhaustive fundamental analysis about the companies that you find in the
                documented knowledge. I want to know the pros and cons of a large-term investment.
                Please, base your answer on what you know about the company,
                but also on wht you find useful about the documented knowledge.
                I want you to also give me your opinion in, if it is worthy to invest on that company
                given the fundamental analysis you make. \n
                If you conclude that is actually wise to invest on a given company,
                or in multiple companies
                (focus only on the ones in the documented knowledge
                 and use the financial statement data provided to enhance your fundamental analysis even further,
                 use the analysis report context to finalise and reason your decision)
                 then come up also with some strategies that I could follow to make the best out of my investments.
                 Make sure to show all the operations you make, given the data in the financial statements.
                 Show the results of the Revenue Growth, ROE, Dividends, Price to Earnings,
                 Market Trends and Industry Performance, company's leadership practices,
                 economic indicators, competitive position, regulatory environments, book value, etc.
                 I want to see all the indicators you can get from this information."""
                output = get_response(user_input, articles_string, financial_statements, symbol, stream_handler)
                if isinstance(output, str):
                    st.session_state.messages.append(ChatMessage(role="assistant", content=output, name="FinGPT"))


if __name__ == "__main__":
    asyncio.run(run())
