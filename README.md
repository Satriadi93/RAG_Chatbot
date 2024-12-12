Retrieval Augmented Generation (RAG) adalah teknik yang dapat memberikan pengetahuan khusus yang tidak dimiliki secara publik kepada sebuah model yang telah di latih,
Alih-alih melakukan pelatihan model sendiri atau melakukan fine tuning terhadap model membutuhkan keahlian, waktu dan environment yang memadai.
Dengan adanya teknik RAG ini dapat mengurangi banyak keterbatasan yang ada, kita dapat memberikan data yang kita miliki tanpa harus melatih model yang memakan banyak waktu dan sumber daya.

Projek ini dirancang untuk membuat chatbot yang dapat memberikan response hanya terkait informasi jurusan Teknik Elektro Universitas Mataram dalam bahasa indonesia, dan data yang di berikan kepada chatbot dalam bentuk PDF yang hanya berisi teks dan tabel

Pada projek ini, menggunakan beberapa library yaitu:
<li>Unstructured
<li>Langchain
<li>Chroma DB
<li>Groq
<li>Ollama

untuk menjalankan projek menggunaka framework steamlit sebagai antarmuka
<code> Streamlit run Chatbot.py </code>

untuk mengganti pengetahuan chatbot dapat di lakukan di halaman knowlage setelah projek di jalankan
