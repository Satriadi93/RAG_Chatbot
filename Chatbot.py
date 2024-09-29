import time
import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_vector import SearchType

# Tentukan direktori untuk penyimpanan vektor dan dokumen
vectorstore_directory = "./vectorDB"
store_directory = "./docstore"

# Muat docstore dari direktori
store = LocalFileStore(store_directory)
docstore = create_kv_docstore(store)

# Buat vectorstore untuk menyimpan embedding
vectorstore = Chroma(
    collection_name="database",
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    persist_directory=vectorstore_directory,
)


# Inisialisasi retriever
id_key = "doc_id"
history_aware_retriever = MultiVectorRetriever(
    search_type=SearchType.similarity_score_threshold ,
    vectorstore=vectorstore,
    # search_kwargs={'k': 5},
    docstore=docstore,
    id_key=id_key,
)

# st.write("data : ", history_aware_retriever)

# Model LLM (gunakan ChatGroq)
llm = ChatGroq(
    temperature=0,
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API2")
)

# Template Prompt untuk question answering
qa_system_prompt = """You are an indonesian assistant for question-answering about "Information of Electrical Engineering at the University of Mataram". \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say Informasi belum tersedia. \
if you want make table use |. \
and answer only from the context. \

{context}"""

# Buat ChatPromptTemplate yang benar
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"), 
        ("human", "{input}"),
    ]
)

# Membuat chain untuk question_answering dengan dokumen yang di-retrieve
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Membuat RAG (Retrieval-Augmented Generation) chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Inisialisasi chat history di session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Membuat antarmuka chat menggunakan Streamlit
st.subheader("🤖 Chatbot Teknik Elektro Universitas Mataram")

# Debugging: Tampilkan chat_history saja di sidebar
# st.sidebar.write("Chat History :", st.session_state.get("chat_history", []))

# # Handle input chat dan responsnya
for chat_history in st.session_state.chat_history:
    with st.chat_message(chat_history["role"]):
        st.markdown(chat_history["content"])

if prompt := st.chat_input("Tanyakan informasi seputar jurusan teknik elektro?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sedang menulis..."):
            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history  
            })
        print("jawaban : ", response["answer"])
        st.sidebar.write("Context : ",response["context"])
        typing_placeholder = st.empty()
        typed_text = ""

        # Asumsi `response['answer']` berisi teks yang ingin ditampilkan
        for char in response['answer']:
            typed_text += char
            typing_placeholder.markdown(typed_text)
            time.sleep(0.01)

    st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

# Hindari menampilkan seluruh session_state
# st.sidebar.write(st.session_state)
