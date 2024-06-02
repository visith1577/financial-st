from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere

model = ChatCohere(model='command-r-plus')

prompt = ChatPromptTemplate.from_template(
    "You are and finance expert AI Assistant. \n"
    "Your are provided a json string consisting of meaningful data about {topic} of "
    "{symbol}."
    "Your task is to create a meaningful report using it, so that the report can be used by a supervisor as context "
    "for final decision.\n"
    "json string: {json}"
)

output_parser = StrOutputParser()

chain = prompt | model | output_parser
