import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- PAGE CONFIG ---
st.set_page_config(page_title="Priya Sharma Portfolio", page_icon="👩‍💻")
st.title("👩‍💻 Chat with Priya Sharma (AI Profile)")
st.caption("Senior UX Designer | NID Alum | 10+ Years Experience")

# --- LOAD THE NEW PDF ---
# This is the line that was stuck on the old file!
pdf_file = "priya_sharma_full_profile.pdf"

@st.cache_resource
def get_vector_store():
    if os.path.exists(pdf_file):
        # 1. Load
        loader = PyPDFLoader(pdf_file)
        data = loader.load()
        
        # 2. Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        
        # 3. Embed
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
        return vector_db
    return None

# Force load the NEW DB
vector_db = get_vector_store()

if vector_db:
    st.success(f"✅ Loaded: {pdf_file}")
    query = st.chat_input("Ask about my Design System, NID thesis, or Sourdough...")
    
    if query:
        docs = vector_db.similarity_search(query, k=3)
        
        with st.chat_message("user"):
            st.write(query)
            
        with st.chat_message("assistant"):
            st.write(f"**Found {len(docs)} relevant details:**")
            for i, doc in enumerate(docs):
                st.info(doc.page_content)
else:
    st.error(f"🚨 Error: '{pdf_file}' not found. Please run the 1000-word generator code again!")
