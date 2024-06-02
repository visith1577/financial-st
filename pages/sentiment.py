import streamlit as st

from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.messages import ChatMessage
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.custom_callback import StreamHandler

from utils.chat import get_sentiment
from utils.tools import get_article_jina


st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon='🦈',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title("Sentiment Analytics")

if 'url' not in st.session_state:
    st.session_state.url = {}
if 'url_name' not in st.session_state:
    st.session_state.url_name = {}


def run():
    articles_string = ''
    urls = []
    url_cont = 0

    if "messages" not in st.session_state:
        st.session_state["messages"] = [ChatMessage(role="assistant", content="")]

    with st.sidebar:
        with st.form(key='URL'):
            url_source = st.text_input("Enter a source URL:")
            url_names = st.text_input("Enter a name for your URL:")
            sentiment_button = st.form_submit_button("Search for Sentiments", type="primary")

    if url_source and url_names:
        if url_source not in st.session_state.url:
            st.session_state.url[url_source] = [url_source]
            st.session_state.url_name[url_source] = [url_names]
        elif url_names not in st.session_state.url_name[url_source]:
            st.session_state.url[url_source].append(url_source)
            st.session_state.url_name[url_source].append(url_names)

        url_name_options = [name for names in st.session_state.url_name.values() for name in names]

        with st.sidebar:
            options = st.multiselect('Sources', url_name_options)
            urls = [url for name in options for url, names in st.session_state.url_name.items() if name in names]

    if len(urls) > 0:
        with st.sidebar:
            with st.sidebar:
                for url in urls:
                    try:
                        article_json = get_article_jina(url)

                        with st.expander(f"Source {url_cont + 1}"):
                            st.subheader(f"{article_json['data']['title']}", divider='rainbow')
                            st.markdown(f"Published on: {article_json['data']['publishedTime']}")
                            st.markdown(f"{article_json['data']['content']}")
                            articles_string += f"Article from Source {url_cont}: \n"
                            articles_string += article_json['data']['content'] + "\n\n"
                        url_cont += 1

                    except:
                        loader = AsyncChromiumLoader([url])
                        docs = loader.load()
                        bs_transformer = BeautifulSoupTransformer()
                        docs_transformed = bs_transformer.transform_documents(docs,
                                                                              tags_to_extract=["div", "span", "h2",
                                                                                               "a"])
                        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000,
                                                                                        chunk_overlap=0)
                        splits = splitter.split_documents(docs_transformed)

                        articles = len(splits)

                        with st.expander(f"Source {url_cont + 1}"):

                            for article in range(articles):
                                pos = article + 1
                                article_content = splits[article].page_content
                                st.subheader(f"Article Number {pos}", divider='rainbow')
                                st.write(article_content)
                                articles_string += f"Article Number {pos} from Source {url_cont}: \n"
                                articles_string += article_content + "\n\n"
                        url_cont += 1
                    finally:
                        senti_button = st.button('Sentiment Task', type="primary")

        if senti_button:
            stream_handler = StreamHandler(st.empty())
            with st.chat_message("assistant"):
                user_input = """
                        Please, give me a quick sentiment analysis about the companies that you find in the documented knowledge. \n
                        I want to know the pros and cons of a large-term investment. \n
                        Please, base your answer on what you know about the company, 
                        but also on what you find useful about the documented knowledge. 
                        I want you to also give me your opinion in, if it
		                is worthy to invest on that company given the sentiment analysis you make. 
		                If you conclude that is actually wise to invest on a given company, or in multiple companies (focus only on the
		                ones in the documented knowledge) then come up also with some strategies that I could follow to make the best out of my investments.
                """
                if articles_string:
                    output = get_sentiment(user_input, articles_string, stream_handler)
                    if isinstance(output, str):
                        st.session_state.messages.append(ChatMessage(role="assistant", content=output))


if __name__ == "__main__":
    run()
