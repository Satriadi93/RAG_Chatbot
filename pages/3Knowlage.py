import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from io import StringIO
from lxml import etree
from typing import Any
from pydantic import BaseModel
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import uuid
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from unstructured_client import UnstructuredClient
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from streamlit_pdf_viewer import pdf_viewer


st.set_page_config(
    page_title="Knowledge Chatbot",
    page_icon="🧠"
)

load_dotenv(find_dotenv())

if not os.path.exists('pdfFiles'):
    os.makedirs('pdfFiles')

if not os.path.exists('vectorDB'):
    os.makedirs('vectorDB')

unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY2")
unstructured_api_url = os.getenv("UNSTRUCTURED_API_URL")

client = UnstructuredClient(
    api_key_auth=unstructured_api_key,
    server_url=unstructured_api_url,
)

uploaded_file = st.file_uploader("", type="pdf")

if uploaded_file is not None:
    if not os.path.exists('pdfFiles/' + uploaded_file.name):
        with st.status("Embedding file..."):
            bytes_data = uploaded_file.read()
            with open('pdfFiles/' + uploaded_file.name, 'wb') as f:
                f.write(bytes_data)

            pdf_path = 'pdfFiles/' + uploaded_file.name
            elements = partition_pdf(filename=pdf_path,
                            infer_table_structure=True,
                            strategy='hi_res',
                            max_characters=2000,
                            new_after_n_chars=1800,
                            # chunking_strategy="by_title",
                            combine_text_under_n_chars=1000
            )

            class Element(BaseModel):
                type: str
                page_content: Any

            # Categorize by type
            categorized_elements = []

            for element in elements:
                if "unstructured.documents.elements.Table" in str(type(element)):
                    categorized_elements.append(Element(type="table", page_content=str(element.metadata.text_as_html)))
                elif "unstructured.documents.elements.NarrativeText" in str(type(element)):
                    categorized_elements.append(Element(type="text", page_content=str(element)))
                elif "unstructured.documents.elements.Address" in str(type(element)):
                    categorized_elements.append(Element(type="text", page_content=str(element)))
                elif "unstructured.documents.elements.EmailAddress" in str(type(element)):
                    categorized_elements.append(Element(type="text", page_content=str(element)))
                elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                    categorized_elements.append(Element(type="text", page_content=str(element)))

            # Pisahkan dokumen berdasarkan tipe (Table dan Text)
            table_elements = [element for element in categorized_elements if element.type == "table"]
            text_elements = [element for element in categorized_elements if element.type == "text"]

            # Summary Chain untuk teks dan tabel
            summary_chain = (
                {"doc": lambda x: x}
                | PromptTemplate.from_template(
                    """" Summarize the content of the following document into a single sentence in Bahasa Indonesia, directly without any introductory phrases or opening sentences : {doc} """
                )
                | ChatGroq(
                    model_name="llama3-8b-8192",
                    api_key=os.getenv("GROQ_API"))
                | StrOutputParser()            )

            tables_content = [i.page_content for i in table_elements]
            tables_summaries = summary_chain.batch(tables_content, {"max_concurrency": 5})

            text_content = [i.page_content for i in text_elements]
            text_summaries = summary_chain.batch(text_content, {"max_concurrency": 5})

            # Simpan dokumen asli dan ringkasannya

            # Ganti InMemoryStore dengan LocalFileStore
            store = LocalFileStore("./docstore")
            id_key = "doc_id"

            # Buat docstore dari LocalFileStore
            docstore = create_kv_docstore(store)

            # Vectorstore untuk indeks chunk data
            vectorstore = Chroma(
                collection_name="database",
                embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
                persist_directory="./vectorDB",
            )

            # Retriever dengan penyimpanan dokumen yang diperbarui
            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                docstore=docstore,
                id_key=id_key,
            )

            # Simpan dokumen asli dan ringkasannya untuk Tabel
            doc_ids = [str(uuid.uuid4()) for _ in table_elements]
            table_documents_asli = [Document(page_content=e.page_content, metadata={id_key: doc_ids[i]}) for i, e in enumerate(table_elements)]
            table_documents_summary = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(tables_summaries)]

            # Simpan dokumen asli di docstore
            retriever.docstore.mset(list(zip(doc_ids, table_documents_asli)))
            # Simpan ringkasan di vectorstore
            retriever.vectorstore.add_documents(table_documents_summary)

            # Simpan dokumen asli dan ringkasannya untuk Teks
            doc_ids = [str(uuid.uuid4()) for _ in text_elements]
            text_documents_asli = [Document(page_content=e.page_content, metadata={id_key: doc_ids[i]}) for i, e in enumerate(text_elements)]
            text_documents_summary = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(text_summaries)]

            # Simpan dokumen asli di docstore
            retriever.docstore.mset(list(zip(doc_ids, text_documents_asli)))
            # Simpan ringkasan di vectorstore
            retriever.vectorstore.add_documents(text_documents_asli)

chroma = Chroma(persist_directory='./vectorDB', collection_name="database")

# Ambil semua metadata dari Chroma
data = chroma.get()
st.sidebar.subheader("Chroma Database")
st.sidebar.write(data)
