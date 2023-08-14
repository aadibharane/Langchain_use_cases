# Contents of ~/my_app/streamlit_app.py
import streamlit as st
from pages.Pandas_agent import pandas_agent
from pages.Pdf_query import pdf_query
from pages.BabyAGI import babyagi
from pages.Csv_agent import csv_agent
from pages.chatgpt_clone import chatgpt_clone
from pages.Two_agents_camel import camel
from pages.AutoGPT import autogpt
from pages.conversation_memory import conversation_memory
from pages.img_generation import img_genration
# from pandas_agent import pandas_agent
# from pdf import pdf_query
# from csv_agent import csv_agent
# from babyagi import babyagi

def main_page():
    st.markdown("# Langchain🎈")
    st.sidebar.markdown("# Langchain🎈")
    return ('Welcome to langchain uses_cases')

def Pandas_agent():
    st.markdown("# pandas Agent❄️")
    st.sidebar.markdown("# pandas Agent❄️")
    res=pandas_agent()
    return res

def Csv_Agent():
    st.markdown("# Csv Agent❄️")
    st.sidebar.markdown("# Csv Agent❄️")
    res=csv_agent()
    return res

def Babyagi():
    st.markdown("#Baby_AGI ❄️")
    st.sidebar.markdown("# Baby_AGI_Agent❄️")
    res=babyagi()
    return res

    

def Pdf_Query():
    st.markdown("# Q&S over pdf File 🎉")
    st.sidebar.markdown("# Pdf_que🎉")
    res=pdf_query()
    return res

def Chatgpt_Clone():
    st.markdown("# Q&S over pdf File 🎉")
    st.sidebar.markdown("# Pdf_que🎉")
    res= chatgpt_clone()
    return res

def Camel():
    st.markdown("# Role-Playing Autonomous 🎉")
    st.sidebar.markdown("# Camel🎉")
    res= camel()
    return res


def AutoGPT():
    st.markdown("# AutoGPT Creater  🎉")
    st.sidebar.markdown("# autogpt🎉")
    res= autogpt()
    return res

def Conversation_Memory():
    st.markdown("Conversation Memory🎉")
    st.sidebar.markdown("# Conversation Memory🎉")
    res= conversation_memory()
    return res

def Image_generation():
    st.markdown("Image genertion🎉")
    st.sidebar.markdown("# AI Image Generation using DALL-E🎉")
    res= img_genration()
    return res



page_names_to_funcs = {
    "Main Page": main_page,
    "Pandas Agent": Pandas_agent,
    "Pdf_query": Pdf_Query,
    "Csv Agent":Csv_Agent,
    "BabyAgi":Babyagi,
    "Chatgpt_Clone":Chatgpt_Clone,
    "Simulation with Two agent CAMEL":Camel,
    "AutoGPT Creator":AutoGPT,
    "Conversation_Memory":Conversation_Memory,
    "Image Genration":Image_generation,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
