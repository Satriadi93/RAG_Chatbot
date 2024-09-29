import time
import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


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
history_aware_retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=id_key,
)

# Model LLM (gunakan Ollama)
llm = Ollama(
    base_url="http://localhost:11434",
    model="llama3",
    verbose=True,
    callback_manager=CallbackManager(
        [StreamingStdOutCallbackHandler()]
    )
)

# Template Prompt untuk question answering
qa_system_prompt = """You are an assistant for question-answering tasks using indonesian language. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say Informasi belum tersedia. \
and answer only to the point.

{context}"""

# Buat ChatPromptTemplate yang benar
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),  # Menggunakan MessagesPlaceholder
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

# Debugging: Tampilkan struktur chat_history untuk memastikan formatnya benar
st.sidebar.write("Chat History Structure:", st.session_state.chat_history)

# Tampilkan chat history
for message in st.session_state.chat_history:
    if "role" in message and "content" in message:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])  # Gunakan 'content' alih-alih 'message'
    else:
        st.write(f"Invalid message format: {message}")

# Handle input chat dan responsnya
if user_input := st.chat_input("Tanyakan informasi seputar teknik elektro Universitas Mataram...", key="user_input"):
    # Tambahkan pesan pengguna ke chat history
    user_message = {"role": "user", "content": user_input}  # Gunakan 'content' bukan 'message'
    st.session_state.chat_history.append(user_message)
    
    # Tampilkan pesan pengguna
    with st.chat_message("user"):
        st.markdown(user_input)

    # Tampilkan respons asisten dengan spinner
    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            # Jalankan RAG chain dengan riwayat percakapan dan input pengguna
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history  # Pastikan chat_history dimasukkan
            })

        # Placeholder untuk efek mengetik
        message_placeholder = st.empty()

        # Animasi pengetikan (typing effect)
        full_response = ""
        for chunk in response["answer"].split():
            full_response += chunk + " "
            time.sleep(0.05)  # Simulasi typing delay
            message_placeholder.markdown(full_response + "▌")  # Indikator typing
        message_placeholder.markdown(full_response)  # Tampilkan hasil akhir

    # Tambahkan respons asisten ke chat history
    chatbot_message = {"role": "assistant", "content": full_response}  # Gunakan 'content' bukan 'message'
    st.session_state.chat_history.append(chatbot_message)
