import streamlit as st
from PIL import Image
from langchain_helper import get_qa_chain, create_vector_db

# Set up a two-column layout
col1, col2, col3, _ = st.columns([50, 1, 3, 1])

# About Amir in the first column
with col1:
    st.title("About Amir 🌱")

# Logo in the second column
with col2:
    st.write("")  # Empty space for horizontal gap

# Add some horizontal spacing
with col3:
    st.image(Image.open(r"C:\Users\amirh\OneDrive\Desktop\personal\others/aslogo.jpg"), width=40)

btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])
