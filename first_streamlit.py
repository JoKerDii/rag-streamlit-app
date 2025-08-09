import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from streamlit_option_menu import option_menu
from langchain_groq import ChatGroq

#load API key
load_dotenv(find_dotenv())
os.environ["GROQ_API_KEY"]=str(os.getenv("GROQ_API_KEY"))

llm = ChatGroq(temperature = 0.5,groq_api_key=os.environ["GROQ_API_KEY"],model_name="llama3-70b-8192")

header = st.container()

def streamlit_ui():
    with st.sidebar:
        choice = option_menu('Navigation',['Simple_Chat'])

    if choice == 'Simple_Chat':
        if prompt:= st.chat_input("Ask a question"):
            result = llm.invoke(prompt)
            st.markdown(result)

streamlit_ui()

