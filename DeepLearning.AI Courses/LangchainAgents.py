## Open AI Function Calling

import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

def add_two_numbers(a,b):
    return a+b
def prime_cut_number(a):
    return a%2

# define a function
functions = [
    {
        "name": "add_two_numbers",
        "description": "adds two numbers and return the result",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                    "description": "the first number e.g, 20",
                },
                "b": {
                    "type": "integer",
                      "description": "The second number e.g, 30"},
            },
            "required": ["a","b"],
        },
    },
        {
        "name": "prime_cut_number",
        "description": "returns whether a number is primecut or not",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                    "description": "the first number e.g, 20",
                },
            },
            "required": ["a"],
        },
    }
]


messages = [
    {
        "role": "user",
        "content": "I am tasked to add the return values of prime cut numbers for 10 and 15."
    }
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions
)

print(response)



## Langchain Expression Language (LCEL)

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

simple_chain = ChatPromptTemplate.from_template("Hello Act as a {cricketer_name}")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = simple_chain | model | output_parser
print(simple_chain.invoke({"cricketer_name":"MS Dhoni"}))
print(chain.invoke({"cricketer_name":"MS Dhoni"}))

from langchain.vectorstores import DocArrayInMemorySearch
from langchain.Embeddings import OpenAIEmbeddings

vectorstore = DocArrayInMemorySearch.from_texts( 
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings()
)
print(vectorstore.similarity_search('ken'))
retriever = vectorstore.as_retriever()

retriever.get_relevant_documents("where did harrison work?")

template = """Answer to the following questions only using the below context:
{context}
Question: {question} 
"""
from langchain.schema.runnable import RunnableMap
runnable_map = RunnableMap(
    {
        "context": lambda x: retriever.get_relevant_documents(x['question']),
        "question": lambda x:x['question']
    }
) | prompt | model | output_parser
