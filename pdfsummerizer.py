import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

qa_chain = None


def process_pdf(file):
    global qa_chain

    tmp_path = file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory
    )

    return "PDF processed! You can now ask questions."

def ask_question(message, history):
    if not qa_chain:
        return "Please upload and process a PDF first."
    response = qa_chain.run(message)
    return response


with gr.Blocks(title="PDF Chat with Gemini") as demo:
    gr.Markdown("Ask your questions regarding PDF")

    with gr.Row():
        file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_button = gr.Button("Process PDF")
        status = gr.Textbox(label="Status")

    chatbot = gr.ChatInterface(fn=ask_question, title="PDF QA Chat")

    upload_button.click(fn=process_pdf, inputs=file_input, outputs=status)

demo.launch()
