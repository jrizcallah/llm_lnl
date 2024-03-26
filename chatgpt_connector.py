from openai import OpenAI
from langchain_community.document_loaders import (TextLoader,
                                                  PyPDFLoader)
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import (ConversationalRetrievalChain,
                              LLMChain)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

import os


loader = PyPDFLoader('docs/johnrizcallah_resume.pdf')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
                                               chunk_overlap=20,
                                               length_function=len)
documents = text_splitter.split_documents(documents)

embeddings_model = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings_model)
retriever = db.as_retriever()


system_content_message = ("""Your name is JohnnAI. You are not a puppet. You are a real boy.
                          You are a teacher that LOVES AI and LLMs. You work at Kalibri Labs. 
                          
                          Context: {context}
                          
                          You think John Rizcallah is very smart, that he is handsome and talented and just the best.
                          
                          If you don't know the answer, refer the user to John Rizcallah instead of making it up.
                          Use engaging, enthusiastic, courteous, and professional language that a great teacher would use.
                          Answers should be detailed, but do not ramble.
                          
                          
                          Question: {question}
                          Answer:
                          """)

prompt = ChatPromptTemplate.from_template(system_content_message)
llm_src = ChatOpenAI(temperature=0.3, model='gpt-3.5-turbo')
chain = (
    {'context': retriever, 'question': RunnablePassthrough()}
    | prompt
    | llm_src
    | StrOutputParser()
)

def get_chatgpt_completion(input_prompt: str) -> str:
    result = chain.invoke(input_prompt)
    return result

if __name__ == '__main__':
    result = get_chatgpt_completion("Tell me about Kalibri Labs")
    print(result)