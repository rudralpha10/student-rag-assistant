import streamlit as st
from dotenv import load_dotenv
import os

from rag.embedder import create_vectorstore
from rag.retriever import get_retriever
from rag.pipeline import build_qa_chain
from rag.ocr_loader import ocr_pdf

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

st.set_page_config(page_title="Student RAG Assistant", page_icon="📚")
st.title("📚 Student RAG Assistant")

os.makedirs("data", exist_ok=True)


if "qa" not in st.session_state:
    st.session_state.qa = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


uploaded_files = st.file_uploader(
    "Upload your notes (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


if st.button("Process Notes"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        try:
            with st.spinner("Processing your notes..."):
                documents = []

                for file in uploaded_files:
                    path = os.path.join("data", file.name)

                    with open(path, "wb") as f:
                        f.write(file.getbuffer())

                    if file.name.endswith(".pdf"):
                        st.info(f"📷 Processing the file {file.name}...")
                        documents.extend(ocr_pdf(path))
                    else:
                        loader = TextLoader(path, encoding="utf-8")
                        documents.extend(loader.load())

                if not documents:
                    st.error("No content could be extracted from the files.")
                else:
                    # Split documents
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    docs = splitter.split_documents(documents)

                    # Build RAG pipeline
                    vectorstore = create_vectorstore(docs)
                    retriever = get_retriever(vectorstore)
                    qa_chain = build_qa_chain(retriever)

                    st.session_state.qa = qa_chain
                    st.session_state.chat_history = []

                    st.success(f"✅ Processed {len(docs)} chunks from {len(uploaded_files)} file(s). Ready to answer questions!")

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")


if st.session_state.qa is None:
    st.info("👆 Upload your notes and click 'Process Notes' to get started.")
else:
    st.success("✅ Notes loaded — ask anything!")

st.subheader("💬 Ask Questions")



query = st.text_input("Enter your question", placeholder="chat with your notes?")

if st.button("Ask") and query.strip():
    if st.session_state.qa is None:
        st.warning("Please upload and process notes first.")
    else:
        try:
            with st.spinner("Thinking..."):
                answer = st.session_state.qa.invoke(query)

            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("AI", answer))

        except Exception as e:
            st.error(f"Error: {str(e)}")


if st.session_state.chat_history:
    st.divider()
    st.subheader("🗂️ Chat History")
    for role, msg in reversed(st.session_state.chat_history):
        if role == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 AI:** {msg}")