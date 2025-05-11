import gradio as gr
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

MODEL_NAME = "gpt-4o-mini"
vectorstore = None
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

def embed_pdf(pdf_path, api_key):
    global vectorstore
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    num, dim = vectorstore.index.ntotal, vectorstore.index.d
    return f"✅ Processed PDF: {num} vectors × {dim} dimensions."

def chat_fn(question, api_key, history):
    # make sure we have a list to append to
    if history is None:
        history = []

    # first, add the user’s question as a message
    history.append({"role": "user", "content": question})

    if vectorstore is None:
        # now add the assistant’s warning as a proper message
        history.append({
            "role": "assistant",
            "content": "⚠️ Please upload & process a PDF first."
        })
        return history, history, ""
        
    llm = ChatOpenAI(model_name=MODEL_NAME, openai_api_key=api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )
    resp = chain.invoke({"question": question})
    answer = resp["answer"]
    # append the assistant’s reply as a message
    history.append({"role": "assistant", "content": answer})
    return history, history, ""       # <-- clear input

def handle_upload(pdf_file, api_key):
    if pdf_file is None:
        return "❌ No PDF uploaded."
    # pdf_file.name is already the path to the temp file on disk
    return embed_pdf(pdf_file.name, api_key)


with gr.Blocks() as app:
    gr.Markdown("# 📚 RAG Chatbot for PDF files")

    api_key_input = gr.Textbox(
        label="Step 1. Input Your OpenAI API Key",
        type="password",
        placeholder="sk-…"
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input     = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn    = gr.Button("Step 2. Upload & Process PDF")
            upload_status = gr.Textbox(label="Status", interactive=False)
        with gr.Column(scale=2):
            chatbot       = gr.Chatbot(type="messages")
            question_in   = gr.Textbox(
                label="Step 3. Ask Questions",
                placeholder="Type and hit Enter…"
            )

    upload_btn.click(
        fn=handle_upload,
        inputs=[pdf_input, api_key_input],
        outputs=upload_status
    )

    question_in.submit(
        fn=chat_fn,
        inputs=[question_in, api_key_input, chatbot],
        outputs=[chatbot, chatbot, question_in]  # <-- hook the clear
    )

if __name__ == "__main__":
    # disable the auto-API schema endpoint (which is hitting a bug)
    app.launch(show_api=False, share=True, inbrowser = True)