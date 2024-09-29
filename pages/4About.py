import uuid
import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

# Tentukan direktori untuk penyimpanan vektor dan dokumen
vectorstore_directory = "./vectorDB"
store_directory = "./docstore"

# Muat docstore dari direktori
store = LocalFileStore(store_directory)
docstore = create_kv_docstore(store)

# Buat vectorstore untuk menyimpan embedding
vectorstore = Chroma(
    collection_name="database",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory=vectorstore_directory,
)

# Inisialisasi retriever
id_key = "doc_id"
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=id_key,
)

# Debug: Lihat jumlah dokumen yang ada di vectorstore
st.write("Dokumen yang tersimpan di vectorstore:", len(vectorstore.get()))

# Template Prompt
template = """Answer the question based only on the following context, which can include text and tables, and answer with Indonesia language:
{context}
Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# Model LLM (gunakan ChatGroq)
model = ChatGroq(
    temperature=0,
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API")
)

# RAG pipeline (retrieval dan generation)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Fungsi untuk melakukan query dan menampilkan hasil
def ask_question(question):
    # Mengambil konteks dari pertanyaan
    context = retriever.get_relevant_documents(question)
    st.write(f"Konteks yang ditemukan untuk pertanyaan '{question}':", context)

    # Jika konteks ditemukan, lakukan inferensi
    if context:
        response = chain.invoke(question)
        st.write(f"Jawaban untuk pertanyaan '{question}': {response}")
    else:
        st.write(f"Tidak ada konteks yang ditemukan untuk pertanyaan '{question}'.")

# Query pertama
question_1 = "Ketua jurusan ?"
ask_question(question_1)

# Query kedua
question_2 = "Nama lengkap pak misbah ?"
ask_question(question_2)

# Query ketiga
question_3 = "Sekjur ?"
ask_question(question_3)

# Query keempat
question_4 = "Nama lengkap pak syafar ?"
ask_question(question_4)
