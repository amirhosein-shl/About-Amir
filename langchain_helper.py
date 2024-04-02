# Import necessary libraries
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create Google Palm LLM model instance
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

# Initialize instructor embeddings using the Hugging Face model
my_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

# File path for the FAISS vector database
vectordb_file_path = "faiss_index"

def create_vector_db():
    """
    Function to create and save a FAISS vector database from CSV data.
    """
    # Load data from CSV file
    loader = CSVLoader(file_path='About_me.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from the loaded data
    vectordb = FAISS.from_documents(documents=data, embedding=my_embeddings)

    # Save the vector database locally
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    """
    Function to create and return a QA chain for retrieval-based question answering.
    """
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, my_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # Define a prompt template for generating questions and answers
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    # Create a PromptTemplate object
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create a RetrievalQA chain
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    # Create and save the vector database
    create_vector_db()

    # Get the QA chain
    chain = get_qa_chain()

    # Test the QA chain with a sample question
    print(chain("What are your career goals?"))
