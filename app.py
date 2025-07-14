import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
## create stuff doc chain and create retrieval chain are very very important
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai

from dotenv import load_dotenv
load_dotenv()
## Loading the GROQ api
## os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma-7b-It")
##Llama-8b-8192

## Lets formulate chat prompy template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on provided context only.
    Please provide the most accuarte response based on question
    <context>
    {context}
    <context>
    Question:{input}
    """
)

## Create vector embeddings 
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings()
        ## Setting up the address for Directory
        ## data ingestion step
        st.session_state.loader=PyPDFDirectoryLoader("research_papers")
        ## Now load the docs
        st.session_state.docs=st.session_state.loader.load()
        ## after loading docs will split the text
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        ## after text splitting have to save somewhere
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        ## After converting store in VectorStore
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
st.title("RAG Document Q&A with Groq and Lama3")   
user_prompt=st.text_input("Enter your query from the research papers")
        
## This button will help create vector store
if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")
    
## WIll check the power of Grok API
import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm, prompt)
    ## will store in form of retriever variable
    retriever=st.session_state.vectors.as_retriever()
    ## will create a retriever chain
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    
    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time :{time.process_time()-start}")
    
    st.write(response['answer'])
    
    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('---------------------')
            
            
            