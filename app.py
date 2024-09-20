import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

groq_api = "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"

llm = ChatGroq(groq_api_key=groq_api, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
    <context>
    {context}
    <context>
    Question:{input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="llama3.1")
        st.session_state.loader = PyPDFDirectoryLoader("documents/")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    
user_prompt = st.text_input("Enter your query")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")
    
import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.perf_counter()
    response = retrieval_chain.invoke({"input": user_prompt})
    st.write(f"Time Taken: {time.perf_counter() - start} seconds")
    
    st.write(response['answer'])
    
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("--------------------------------------------------------")
        
    