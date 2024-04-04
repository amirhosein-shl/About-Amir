# Personalized Q&A App

This project implements a personalized question-answering (Q&A) application utilizing language models and vector retrieval for querying responses. The app is designed to provide detailed answers to questions about the developer's personal information and interests.

## Overview

The Q&A app leverages the LangChain library to create an interactive interface for users to ask questions about the developer, Amir Shahlaee. It uses advanced natural language processing (NLP) techniques to retrieve relevant answers from a pre-defined database of information.

## Features

- **Customized Responses**: The app provides personalized responses to user questions based on the developer's input and preferences.
- **Language Models**: It utilizes the Google Palm language model for generating accurate and context-aware responses.
- **Vector Retrieval**: The application uses vector stores and retrieval mechanisms to efficiently match user queries with the most relevant responses.

## Usage

To use the Q&A app, follow these steps:

1. **Setup Environment**: Make sure you have Python installed on your system.
2. **Install Dependencies**: Install the required Python packages by running `pip install -r requirements.txt`.
3. **Configure Environment Variables**: Set up environment variables, especially the Google API key, by creating a `.env` file and adding the necessary keys.
4. **Create Vector Database**: Run the `create_vector_db()` function to generate a vector database from the provided dataset.
5. **Query the Application**: Use the `get_qa_chain()` function to initialize the Q&A chain and start querying the application with your questions.

## Sample Usage

```python
from qa_app import get_qa_chain

if __name__ == "__main__":
    chain = get_qa_chain()
    question = "What are your career goals?"
    print(chain(question))

