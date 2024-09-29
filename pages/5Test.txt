from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.schema.document import Document

vectorstore_directory = "./vectorDB"
store_directory = "./docstore"

# # Muat docstore dari direktori
store = LocalFileStore(store_directory)
docstore = create_kv_docstore(store)

# Buat vectorstore untuk menyimpan embedding
vectorstore = Chroma(
    collection_name="database",
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    persist_directory=vectorstore_directory,
)

# Misalkan kita ingin menghapus dokumen berdasarkan doc_id yang diberikan
# doc_id_to_delete = "523f923f-6663-46da-9baf-5a89e02ab3ba"
# doc_id_to_delete2 = "506a6978-f399-47da-9232-b6ec001a8950"

# 1. Hapus dari Chroma Vectorstore
# vectorstore.delete(ids=doc_id_to_delete2)


id_update = "83dc8289-a266-4e9a-8921-3d75090f007f"
doc_id = "d8d4ac08-36d6-422d-9b7f-1bf6cd2095f0"

new_document = Document(page_content="Nama lengkap, NIP, email 1, email 2, dan alamat dosen. Adapun nama-nama dosen yang tertulis dalam tabel ini antara lain A. Sjamsjiar Rachman, ST., MT, pak Abdul Natsir, ST., MT,pak Abdullah Zainuddin, ST., MT,pak Agung Budi Muljono, ST., MT, Budi Darmawan, ST., M.Eng., Bulkis Kanata, ST., MT.,pak Cahyo Mustiko O. M., ST., M.Sc., Ph.D., pak Cipta Ramadhani, ST., M.Eng., Djul Fikry Budiman, ST., MT., Dr. I Made Ginarsa, ST., MT., Dr. Ida Ayu Sri Adnyani, ST., M.Erg.,pak Dr.Ir. Misbahuddin, ST., MT.IPu, Dr. rer. nat. Teti Zubaidah, ST., MT., Dr. Rismon H. Sianipar, ST., MT., M. Eng., pak Dr. Warindi, ST., M.Eng., Dwi Ratnasari, S.Kom., MT., Giri Wahyu Wiriasto, ST., MT., I Ketut Perdana Putra, ST., MT., I Ketut Wiryajati, ST., MT., I Made Ari Nrartha, ST., MT., I Made Budi Suksmadana, ST., MT., I Nyoman Wahyu S., ST., MSc., Ph.D., Ida Bagus Fery Citarsa, ST., MT., Lalu A. Syamsul Irfan Akbar, ST., M.Eng., Made Sutha Yadnya, ST., MT., Muhamad Irwan, ST., MT., Muhamad Syamsu Iqbal, ST., MT., Ph.D., Ni Made Seniari, ST., MT., Paniran, ST., MT., Rosmaliati, ST., MT., Sabar Nababan, ST., MT., Sayidiman, ST. M.Eng., Sudi M. Al Sasongko, ST., MT., Sultan, ST., MT., Supriono, ST., MT., Supriyatna, ST., MT., Syafarudin Ch, ST., MT., dan Suthami Ariessaputra, ST., M.Eng.", metadata={"doc_id": doc_id})


vectorstore.update_document(id_update,new_document)
