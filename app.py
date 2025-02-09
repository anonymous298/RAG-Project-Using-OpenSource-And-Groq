import streamlit as st
import os
import time

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Load environment variables
# load_dotenv()
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# groq_api_key = os.environ['GROQ_API_KEY']

# Initialize Vector DB and Processing Steps

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'vectordb' not in st.session_state:
    st.session_state.loader = WebBaseLoader('https://docs.smith.langchain.com/')
    st.session_state.documents = st.session_state.loader.load()

    st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.splitter.split_documents(st.session_state.documents)

    st.session_state.embeddings = OllamaEmbeddings(model='nomic-embed-text')
    st.session_state.vectordb = Chroma.from_documents(
        documents=st.session_state.final_documents,
        embedding=st.session_state.embeddings,
        persist_directory="./chroma_db"  # This ensures persistence
        )

# Initialize Retrieval Chain
if 'retrieval_chain' not in st.session_state:
    # st.session_state.llm = ChatGroq(
    #     groq_api_key=groq_api_key,
    #     model='mixtral-8x7b-32768'
    # )

    st.session_state.llm = ChatOllama(  # Comment this model and uncommet Groq model to use it
        model='deepseek-r1:1.5b'
    )

    # Prompt Template
    st.session_state.template = '''
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Question: {input}
    '''
    
    st.session_state.prompt = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful assistant'),
        ('user', st.session_state.template)
    ])

    st.session_state.document_chain = create_stuff_documents_chain(st.session_state.llm, st.session_state.prompt)
    st.session_state.retriever = st.session_state.vectordb.as_retriever()
    st.session_state.retrieval_chain = create_retrieval_chain(st.session_state.retriever, st.session_state.document_chain)

st.title('RAG-Project-Using-OpenSource-And-Groq')

# Display previous messages in chat format
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_input = st.chat_input('Enter what you want to ask...')
 
if user_input:

    # Add user input to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    start = time.process_time()
    
    response = st.session_state.retrieval_chain.invoke({'input': user_input})
    response_time = time.process_time() - start
    
    st.write(f"Response time: {response_time:.2f} seconds")
    # st.write(response['answer'])

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Display AI response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response['answer']:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)

    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Show document similarity search results
    with st.expander("Document Similarity Search"):
        if "context" in response:
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
        else:
            st.write("No similar documents found.")
