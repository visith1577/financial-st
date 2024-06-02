from utils.tools import get_news_search, insider_sentiment, insider_transactions
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate
from utils.custom_callback import StreamHandler
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings.jina import JinaEmbeddings
import streamlit as st
import vertexai

from langchain_cohere import ChatCohere

vertexai.init(
    project="artful-doodad-423906-a6",
    location="asia-east1",
    staging_bucket="gs://vck-genai-bkt",
)

embedding = JinaEmbeddings(jina_api_key=st.secrets['JINA_KEY'], model_name="jina-embeddings-v2-base-en")

prompt = PromptTemplate.from_template(
    """The following is a friendly conversation between a human and an AI. The
		AI is an AI-powered fundamental analyst. It uses Documented Information to give fundamental insights on an 
		asset determined by the user. It is specific, and gives full relevant
		perspective to let the user know, if it is worthy to make investment on a certain asset, 
		and can also build intelligent strategies with the given information, as well as from intel that it
		already knows, or will generate. Take into account, that the documented knowledge comes in the next structure.
		**Title:** (Message of the Title), **Description:** (Message of the Description)\n\n, and so on. 
		All articles from the documented knowledge have a title and a description (both of which are separated by comas), 
		and all articles are separated with the \n\n command between one another. 
		Additionally you are provided with the relevant context from analysis reports of insider sentiment, insider transactions
		and public sentiment.
		The AI will also use the Financial Statements of they are provided. 
		NOTE: As you know, Financial statements are VERY important for the fundamental analysis, so use them wisely in order to give the best financial statement there could be in the world. 
		Show the operations you make for the fundamental analysis as you make it.
		Each financial statement will be separated in function of the company/asset the belong to. So be sure to make that fundamental analysis using the financial statement belonging to the one
		company you are analyzing.

		Documented Information:
		{docu_knowledge},

		Financial Statements:
		{financial_statement}
		
		Analysis report context:
		{context}

		(You do not need to use these pieces of information if not relevant)

		Current conversation:
		Human: {input}
		AI-bot:"""
)

sentiment_prompt = PromptTemplate.from_template(
    """
    The following is a friendly conversation between a human and an AI. The
	AI is an AI-powered sentiment analyst. It uses Documented Information to give fundamental insights on an 
	assets determined by the user. 
	It is specific, and gives full relevant perspective to let the user know, if it is worthy to make investment on a certain asset, 
	and can also build intelligent strategies with the given information, as well as from intel that it already knows, or will generate.

	Documented Information:
	{docu_knowledge},

	(Identify the information that is relevant, You do not need to use these pieces of information if not relevant)

	Current conversation:
	Human: {input}
	AI-bot:
    """
)


def retriever(symbol):
    get_news = get_news_search(symbol)
    insider_tr = insider_transactions(symbol)
    insider_sent = insider_sentiment(symbol)

    documents = [
        Document(
            page_content=get_news,
            metadata={"topic": "analysis report of news sentiments", "company": symbol}
        ),
        Document(
            page_content=insider_tr,
            metadata={"topic": "analysis report of insider transaction", "company": symbol}
        ),
        Document(
            page_content=insider_sent,
            metadata={"topic": "analysis report of insider sentiments", "company": symbol}
        ),
    ]

    vector_store = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory='./vector_store')
    retriever_ = vector_store.as_retriever(
        search_kwargs={
            "filter": {"company": symbol},
        }
    )
    return retriever_


def get_response(user_message, docu_knowledge, financial_statement, symbol, callback):
    model = ChatVertexAI(model_name='gemini-1.5-flash-001', streaming=True, temperature=0,
                         callbacks=[callback])
    # model = ChatCohere(model='command-r-plus', streaming=True, temperature=0, callbacks=[stream_handler])
    retriever_model = retriever(symbol)

    conversation_with_summary = prompt | model
    output = conversation_with_summary.invoke({
        "input": user_message,
        "docu_knowledge": docu_knowledge,
        "financial_statement": financial_statement,
        "context": retriever_model.invoke(user_message),
    })

    return output


def get_sentiment(user_message, docu_knowledge, callback):
    model = ChatVertexAI(model_name='gemini-1.5-flash-001', streaming=True, temperature=0.3,
                         callbacks=[callback])

    conversation_with_summary = sentiment_prompt | model
    output = conversation_with_summary.invoke({
        "input": user_message,
        "docu_knowledge": docu_knowledge,
    })

    return output
