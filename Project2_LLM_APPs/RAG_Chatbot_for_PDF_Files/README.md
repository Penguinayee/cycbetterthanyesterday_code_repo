
# 📚 RAG Chatbot for PDF Files

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** chatbot using Gradio and LangChain. The chatbot can:

1. Upload and process PDF files.
2. Embed PDF contents as vector representations.
3. Answer questions based on the embedded PDF contents.

## Features
- **PDF Upload and Embedding:** Uses OpenAI embeddings with FAISS to store vector representations.
- **Conversational Interface:** Uses Gradio's Chatbot component to interact with users.
- **Persistent Memory:** Remembers chat history during the session.
- **API Integration:** Utilizes OpenAI's GPT-4o-mini model for answering questions.
- **Hosted on Hugging Face:** Deployed as a Gradio app on Hugging Face Spaces.

## Installation
### Prerequisites
- Python 3.9+
- pip

## How to Run
1. Download the files:
- requirements.txt
- app.py
- (Optiional) Example_pdf_documents

2. Install required packages:
You can install the required packages directly:
```
pip install -qU gradio langchain langchain_openai langchain-community faiss-cpu pypdf 
```

Or use a requirements.txt file:
```
pip install -r requirements.txt
```

3. Run the application locally:
```
python app.py
```

### Run on Hugging Face Spaces
- Visit the app's page on Hugging Face Spaces: [Hugging Face RAG Chatbot](https://huggingface.co/spaces/cycbetterthanyesterday/RAG-Based_PDF_summarizer)

## Usage
1. **Paste Your OPENAI API-KEY:** Ususlly starts with 'sk-xxxxxx'
2. **Upload PDF:** Click the 'Upload & Process PDF' button after selecting a file.
3. **Ask Questions:** Type your question and hit enter. The chatbot will respond based on the uploaded PDF content.

## Troubleshooting
- **No PDF Uploaded Warning:** Make sure to upload a PDF before asking questions.
- **Missing API Key:** Provide a valid OpenAI API key for the chatbot to function.

## License
MIT License
