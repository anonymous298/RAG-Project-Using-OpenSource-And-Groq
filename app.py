import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
from streamlit.runtime.state import session_state

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

if 'vectordb' not in st.session_state:
    # Creating Document Loader Session
    st.session_state.loader = WebBaseLoader('https://docs.smith.langchain.com/')
    st.session_state.documents = st.session_state.loader.load()

    # Creating Splitter Session
    st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.splitter.split_documents(st.session_state.documents)

    # Creating Vector Store Session
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.vectordb = Chroma.from_documents(st.session_state.final_documents, st.session_state.embeddings)

if 'retrieval_chain' not in session_state:
    # Creating Model LLM Session
    st.session_state.llm = ChatGroq()

    # Creating Prompt Template Session State
    st.session_state.template = '''
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions:{input}
    '''
    st.session_state.prompt = ChatPromptTemplate(
        [
            ('system', 'You are an helpful assistant'),
            ('user', st.session_state.template)
        ]
    )

    # Creating Document Chain
    st.session_state.document_chain = create_stuff_documents_chain(st.session_state.llm, st.session_state.prompt)

    # Creating retriever session
    st.session_state.retriever = st.session_state.vectordb.as_retriever()

    st.session_state.retrival_chain = create_retrieval_chain(st.session_state.retriever, st.session_state.document_chain)


user_input = st.text_input('Enter what you want to ask...')

if user_input:
    response = st.session_state.retrieval_chain.invoke({'input' : user_input})
    st.write(response['answer'])