# PDF RAG Chatbot for Salary & MPF Q&A

An interactive chatbot powered by Streamlit and Retrieval-Augmented Generation (RAG) that enables users to upload PDF documents, search for answers, and receive accurate, context-based responses about Hong Kong worker salary and MPF processing.

---

## Features

- **PDF Upload & Parsing:** Easily upload single or multiple PDFs. The app auto-extracts, cleans, and normalizes their text.
- **Document Chunking & Embedding:** Documents are split into chunks, converted to embeddings using Ollama, and stored for rapid search.
- **FAISS Vector Search:** Quickly retrieves relevant document sections by similarity using a FAISS vector index.
- **Context-Rich Q&A:** Input questions into the chat; the bot finds and quotes the most relevant information from the documents.
- **Streamlit Web UI:** A user-friendly interface for document upload, chat history, and interactive conversation.
- **Multi-Turn Chat:** Maintains session state so users can have multi-turn conversations referencing previous context.
- **Custom Prompting:** Answers are tailored for Hong Kong salary and MPF subject matter using professional, easy-to-understand language.

---

## Prerequisites

- Python 3.9+
- Ollama server with supported models (such as llama3.2) running and accessible
- PDF documents relevant to your use case

---

## Technologies

- Streamlit
- LangChain
- Ollama
- FAISS
- SentenceTransformers
- PyPDF
- Python (pandas, tqdm, etc.)

---

## Example Screenshot

*(Insert screenshot of UI here)*
(ragchatbot.png)
---

## Configuration Notes

- **Environment Variables:**  
The project uses environment variables for the Ollama server URL (`OLLAMA_URL`) and model version (`OLLAMA_MODEL`). You can set these in your shell or use a `.env` file with tools like `python-dotenv`.
- **Extending Document Sources:**  
To support other file types, modify the PDF parsing and chunking functions.

---

## Contribution

Pull requests, issues, and suggestions are welcome!  
Please follow conventional commit formats and code style rules.

---

## License

MIT License

---

## Author
[Chau Hung Cheung, cyrus]
[c60413094@gmail.com]
