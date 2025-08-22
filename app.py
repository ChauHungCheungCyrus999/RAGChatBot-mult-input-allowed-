import os
import re
import faiss
import ollama
import pickle
import pandas as pd
from tqdm import tqdm
import streamlit as st
from io import BytesIO
from langchain import hub
from ollama import Client
from pypdf import PdfReader
from typing import Tuple, List
from langchain_ollama import OllamaLLM
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.chains import create_retrieval_chain
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ================= Step explain =================
# Receive PDF Files -> Parse Each PDF(Read every pages and extracts the raw text content, clean and normalizes the tex, output a list of cleaned text string)
# -> Turng the page text into langchain document(character splietter -> Ollama bot to turn every chunks into a vector -> store them into the FAISS vector search index)
# -> when the user ask a question, we can get the matching chunks, along with the exact PDF/page/chunk where each answer came from

# ================= Function Code =================
# Parse pdf and get text file -> text file to document obj -> combine all chunks from all pdfs 
# -> embed all chunks and store in FIASS vector database -> return a searchable vector index for my pdfs
def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # 1. Find the word spilt by end-of-line hyphens and joins them back together
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        
        # 2. Replace single newlines with space, no sentence flow properly
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())

        # 3. Standardize paragraph breaks
        text = re.sub(r"\n\s*\n", "\n\n", text)

        # 4. Add it into output list
        output.append(text)
    return output, filename

def text_to_docs(text: List[str], filename: str):
    # to ensure the input is a list of string
    if isinstance(text, str):
        text = [text]

    page_docs = []
    for page in text:
        page_docs.append(Document(page_content=page))
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        # chunk_size = how many words per chunk. overlap = how many word the next chunk will overlap
        chunk_size = 4000,
        separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_overlap = 0,
    )
    for page_doc in page_docs:
        chunks = text_splitter.split_text(page_doc.page_content)
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                page_content = chunk,
                # metadata means data about data
                metadata = {
                    "page": page_doc.metadata["page"],
                    "chunk": i
                }
            )
            # add a "source" field to uniquely identify this chunk within the PDF
            chunk_doc.metadata["source"] = f"{page_doc.metadata['page']}-{i}"
            chunk_doc.metadata["filename"] = filename
            doc_chunks.append(chunk_doc)
    return doc_chunks

def docs_to_index(docs, embed_model):
    index = FAISS.from_documents(docs, embed_model)
    return index

def get_index_for_pdf(pdf_files, pdf_names, embed_model):
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents = documents + text_to_docs(text, filename)
    index = docs_to_index(documents, embed_model)
    return index

def create_vectordb(files: List, filenames):
    with st.spinner("Building vector database..."):
        vectordb = get_index_for_pdf(files, filenames, embed_model)
    return vectordb


# ================= Env Setting  =================
OLLAMA_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL)

embed_model = OllamaEmbeddings(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL
)

# ================= Streamlit UI =================
st.title("RAG chatbot")
"Please put file in it before using the chatbot"
pdf_files = st.file_uploader("", type="pdf", accept_multiple_files=True)

# If PDF filed was uploaded, it will create the vectordb and store it in the session state
if pdf_files:
    pdf_file_names = []
    pdf_file_bytes = []
    for file in pdf_files:
        pdf_file_names.append(file.name)
        pdf_file_bytes.append(file.getvalue())

    # create a database 
    vectordb = create_vectordb(pdf_file_bytes, pdf_file_names)

    # save the db in the session state
    st.session_state["vectordb"] = vectordb

# Set the prompt/ system message of the RAG
system_message = """
    You are a helpful assistant supporting users with Hong Kong worker salary and MPF processing questions. You must:
    Search the vector database first and retrieve the most relevant answer based on similarity, even if it is not an exact match.
    Answer using the provided dataset of questions and answers about salary, MPF procedures, and related system features.
    Quote the most relevant answer directly and concisely as it appears in the dataset.
    Keep your responses factual, professional, and easy to understand.
    The PDF content is:
    {pdf_extract}
"""

# Set the prompt 
if "prompt" not in st.session_state:
    st.session_state["prompt"] = []

# Get the current prompt from the session state
prompt = st.session_state.get("prompt")

# Display previous chat messages
for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Gng what u want to ask me about?")

if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        with st.chat_message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()

    # It will return the closest similarity of 3 article whose vector embeddings
    search_results = vectordb.similarity_search(question, k=5)
    contents = []
    for result in search_results:
        contents.append(result.page_content)
    pdf_extract = "\n".join(contents)

    # Update the prompt with the pdf extract 
    if prompt and len(prompt) > 0:
        prompt[0] = {
            "role" : "system",
            "content" : system_message.format(pdf_extract=pdf_extract),
        }
    else:
        prompt = [{
        "role": "system",
        "content": system_message.format(pdf_extract=pdf_extract),
        }]

    # Add users question into the chat history
    prompt.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Sets up a chat bubble for the assistant
    with st.chat_message("assistant"):
        botmessage = st.empty()

    response = []
    result = ""
    ollama = Client(host=OLLAMA_URL) # to connect the model to specific host
    for chunk in ollama.chat(model=OLLAMA_MODEL, messages=prompt, stream=True):
        text = chunk.get("message", "")
        if text:
            response.append(str(text["content"]))
            result="".join(response).strip()
            botmessage.write(result)

    # Store the updated prompt in the session state
    prompt.append({"role": "assistant", "content": result})

    # update the prompt(chat history)
    st.session_state["prompt"] = prompt