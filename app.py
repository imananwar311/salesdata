import os
import requests
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import gradio as gr
import pdfplumber

# Load API
from google.colab import userdata
os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Your drive doc URLs → must be direct download links
DRIVE_DOCS = [
    "https://drive.google.com/uc?export=download&id=1uozI5qhA9G_YiYGPviC_32q4r2RNVLB7",
    "https://drive.google.com/uc?export=download&id=1gl_6EAvN5uzTUbir_ytOBUaSmr9pWKNF"
]

# FAISS global
kb_chunks = []
faiss_index = None

def load_and_index_docs():
    global kb_chunks, faiss_index

    texts = []

    for url in DRIVE_DOCS:
        r = requests.get(url)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(r.content)
        tmp.close()

        with pdfplumber.open(tmp.name) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    texts.append(t)

    # chunk by paragraphs
    chunk_size = 500
    kb_chunks = []
    for t in texts:
        for i in range(0, len(t), chunk_size):
            kb_chunks.append(t[i:i+chunk_size])

    # embed
    emb = embed_model.encode(kb_chunks).astype("float32")
    faiss_index = faiss.IndexFlatL2(emb.shape[1])
    faiss_index.add(emb)

    return f"Loaded and indexed {len(kb_chunks)} chunks from knowledge base 💾"

def get_answer(question):
    global kb_chunks, faiss_index

    if faiss_index is None:
        return "Knowledge base not indexed yet! 🧠"

    q_emb = embed_model.encode([question]).astype("float32")
    D, I = faiss_index.search(q_emb, 4)

    retrieved = [kb_chunks[i] for i in I[0]]
    context = "\n\n".join(retrieved)

    if max(D[0]) < 0.1:
        return "I don't know. ❌"

    # Ask Groq
    chat = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Answer using the following context"},
            {"role": "user", "content": f"C:{context}\n\nQ:{question}"}
        ],
        model="llama-3.3-70b-versatile"
    )

    return chat.choices[0].message.content

# UI
with gr.Blocks(css="""
    .answer-box { border: 2px solid #4B39EF; padding: 18px; border-radius: 12px; background: #f7f4ff; font-size: 18px; }
""") as demo:

    gr.HTML(
        """
        <div style="
            text-align:center;
            margin-bottom:30px;
            font-family:Poppins, sans-serif;">
            <h1 style="font-size:3rem; color:#4B39EF;">📚 My RAG Knowledge Base</h1>
            <p style="font-size:1.2rem; color:#555;">
                Ask any question about the documents below. If it's outside the knowledge base, I'll tell you I don't know! 🧠
            </p>
        </div>
        """
    )

    load_btn = gr.Button("Index My Knowledge Base 🚀")
    q_in = gr.Textbox(label="Ask your question:", placeholder="Type here…")
    answer = gr.HTML("<div class='answer-box'>Answer will show here...</div>", label="Answer")

    load_btn.click(load_and_index_docs, outputs=answer)
    q_in.submit(get_answer, inputs=q_in, outputs=answer)

demo.launch()
