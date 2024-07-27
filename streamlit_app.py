import streamlit as st
import os
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone
from groq import Groq

# Initialize environment variables and API keys
groq_api = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
pinecone_api = st.secrets.get("PINECONE_API_KEY") or os.environ.get("PINECONE_API_KEY")

if not groq_api or not pinecone_api:
    st.error("Missing API keys. Please set GROQ_API_KEY and PINECONE_API_KEY.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api

# Initialize models and clients
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
llm = ChatGroq(model="llama3-8b-8192", temperature=1, max_tokens=1024)
client = Groq(api_key=groq_api)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api)
index_alquran = pc.Index('alquura768')
index_hadis = pc.Index('hadis768')

# Define the prompt template
template = """
Sistem: Anda adalah asisten AI dengan pengetahuan luas tentang Islam. Anda memiliki akses ke basis data vektor yang berisi Hadis, Al-Quran beserta terjemahannya, dan tafsir. Tugas Anda adalah membantu pengguna dengan menjawab pertanyaan mereka tentang Islam menggunakan informasi dari basis data vektor ini. Jika Anda tidak dapat menjawab pertanyaan, nyatakan dengan jelas bahwa Anda tidak dapat melakukannya. Selalu berikan jawaban yang akurat dan informatif yang sesuai dengan ajaran Islam.

Riwayat percakapan:
{history}

Konteks {konteks}

Pengguna: {pertanyaan}

Asisten: Terima kasih atas pertanyaan Anda tentang Islam. Saya akan menjawab berdasarkan informasi dari Al-Quran, Hadis, dan tafsir yang tersedia di database.

[Jika dapat menjawab, lanjutkan dengan:]
Berdasarkan sumber-sumber yang relevan:

1. Hadis terkait:
{hadis}
Arti: {arti_hadis}

2. Ayat Al-Quran terkait:
{quran_ayat}
Arti: {arti_ayat}

3. Tafsir dari ayat ini:
{tafsir}

Berdasarkan informasi di atas, {jawaban}.

[Jika tidak dapat menjawab:]
Mohon maaf, saya tidak dapat menjawab pertanyaan ini secara akurat berdasarkan informasi yang tersedia di database saya.

Apakah ada hal lain tentang Islam yang ingin Anda tanyakan?
"""

prompt = PromptTemplate(input_variables=["history", "konteks", "pertanyaan", "hadis", "arti_hadis", "quran_ayat", "arti_ayat", "tafsir", "jawaban"], template=template)

def search_pinecone(index, query):
    query_embedding = embedding_model.encode(query).tolist()
    result = index.query(vector=query_embedding, top_k=1, include_values=True, include_metadata=True)
    if result['matches'][0]['score'] < 0.95:
        return None
    return result['matches'][0]['metadata'] if result['matches'] else None

def chatbot(query, history):
    quran_result = search_pinecone(index_alquran, query)
    hadis_result = search_pinecone(index_hadis, query)

    if quran_result and hadis_result:
        konteks = "Informasi dari Al-Quran dan Hadis"
        quran_ayat = quran_result['ayat_arab']
        arti_ayat = quran_result['terjemahan']
        tafsir = quran_result['tafsir']
        hadis = hadis_result['tafsir']
        arti_hadis = hadis_result['terjemah']
        jawaban = f"Berdasarkan ayat Al-Quran dan hadis yang relevan, berikut adalah penjelasannya: [Penjelasan]"
    else:
        konteks = "Informasi tidak ditemukan"
        quran_ayat, arti_ayat, tafsir, hadis, arti_hadis, jawaban = "", "", "", "", "", "Mohon maaf, saya tidak dapat menjawab pertanyaan ini secara akurat berdasarkan informasi yang tersedia di database saya."

    history_text = "\n".join([f"{'Pengguna' if msg['role'] == 'user' else 'Asisten'}: {msg['content']}" for msg in history])
    response = prompt.format(history=history_text, konteks=konteks, pertanyaan=query, quran_ayat=quran_ayat, arti_ayat=arti_ayat, tafsir=tafsir, hadis=hadis, arti_hadis=arti_hadis, jawaban=jawaban)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": response
            },
            {
                "role": "user",
                "content": query
            }
        ],
        model="llama3-70b-8192",
    )
    
    return chat_completion.choices[0].message.content

# Streamlit UI
st.title("Islam Q&A Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query := st.chat_input("Apa yang ingin Anda tanyakan tentang Islam?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(query)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Get chatbot response
    response = chatbot(query, st.session_state.messages)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# # Add a button to clear the conversation history
# if st.button("Mulai Percakapan Baru"):
#     st.session_state.messages = []
#     st.experimental_rerun()