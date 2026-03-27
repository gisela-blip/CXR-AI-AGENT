"""
app.py
------
Medical X-Ray AI Chatbot — Main Application
Uses MedGemma (Gisela13154/medgemma-4b-it-sft-lora-ms-cxr-evaluation) for image analysis
+ FAISS RAG from PPK COMPILE + EN→ID translation.

Run with:
    py -3 app.py
"""

import os
import sys
import gc
import pickle
import time
import textwrap
import numpy as np
import gradio as gr
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR   = os.path.join(BASE_DIR, "faiss_index")
EXCEL_PATH  = os.path.join(BASE_DIR, "PPK COMPILE.xlsx")

# ── Model IDs ─────────────────────────────────────────────────────────────────
MEDGEMMA_MODEL_ID = "Gisela13154/medgemma-4b-it-sft-lora-ms-cxr-evaluation"
EMBED_MODEL_ID    = "sentence-transformers/all-MiniLM-L6-v2"
TRANS_MODEL_ID   = "Helsinki-NLP/opus-mt-en-id"          # EN → Bahasa Indonesia

# ── Global model handles ───────────────────────────────────────────────────────
med_model        = None
med_processor    = None
embed_model      = None
faiss_index      = None
ppk_records      = None
trans_tokenizer  = None
trans_model      = None


# ════════════════════════════════════════════════════════════════════════════
# 1. Model Loading
# ════════════════════════════════════════════════════════════════════════════

def load_medgemma():
    """
    Load MedGemma LoRA model.
    Strategy:
      1. Try loading as a full merged model (AutoModelForImageTextToText).
      2. If the repo is a LoRA-only adapter, load the base model then apply PEFT.
    """
    global med_model, med_processor
    if med_model is not None:
        return

    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch

    use_gpu   = torch.cuda.is_available()
    dtype     = torch.bfloat16 if use_gpu else torch.float32
    dev_map   = "auto" if use_gpu else "cpu"

    print(f"[App] Loading MedGemma model: {MEDGEMMA_MODEL_ID}")
    print(f"[App] Device: {'GPU (bfloat16)' if use_gpu else 'CPU (float32 — this will be slow)'}")

    # Load processor first (works for both merged and LoRA repos)
    med_processor = AutoProcessor.from_pretrained(
        MEDGEMMA_MODEL_ID,
        trust_remote_code=True,
    )

    # ── Attempt 1: load as a full (merged) model ──────────────────────────────
    try:
        med_model = AutoModelForImageTextToText.from_pretrained(
            MEDGEMMA_MODEL_ID,
            torch_dtype=dtype,
            device_map=dev_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print("[App] ✅  MedGemma loaded as merged model.")

    except Exception as e_merged:
        # ── Attempt 2: LoRA adapter → load base + apply adapter ──────────────
        print(f"[App] Merged load failed ({e_merged}). Trying PEFT LoRA approach…")
        BASE_MODEL = "google/medgemma-4b-it"
        from peft import PeftModel

        base = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            device_map=dev_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        med_model = PeftModel.from_pretrained(base, MEDGEMMA_MODEL_ID)
        med_model = med_model.merge_and_unload()   # fuse LoRA weights
        print("[App] ✅  MedGemma LoRA merged and loaded.")

    med_model.eval()


def load_rag():
    """Load FAISS index and PPK records. Build index if not found."""
    global faiss_index, ppk_records, embed_model
    import faiss
    from sentence_transformers import SentenceTransformer

    index_path   = os.path.join(INDEX_DIR, "ppk.index")
    records_path = os.path.join(INDEX_DIR, "ppk_records.pkl")

    if not os.path.exists(index_path):
        print("[App] FAISS index not found — building now…")
        from rag_builder import load_ppk_data, build_index
        records = load_ppk_data(EXCEL_PATH)
        build_index(records, EMBED_MODEL_ID, INDEX_DIR)

    print("[App] Loading FAISS index…")
    faiss_index = faiss.read_index(index_path)
    with open(records_path, "rb") as f:
        ppk_records = pickle.load(f)

    print(f"[App] Loading embedding model: {EMBED_MODEL_ID}")
    embed_model = SentenceTransformer(EMBED_MODEL_ID)
    print(f"[App] ✅  RAG ready ({faiss_index.ntotal} vectors).")


def load_translator():
    """Load Helsinki-NLP EN→ID translator."""
    global trans_tokenizer, trans_model
    if trans_model is not None:
        return

    from transformers import MarianMTModel, MarianTokenizer
    print(f"[App] Loading translator: {TRANS_MODEL_ID}")
    trans_tokenizer = MarianTokenizer.from_pretrained(TRANS_MODEL_ID)
    trans_model     = MarianMTModel.from_pretrained(TRANS_MODEL_ID)
    trans_model.eval()
    print("[App] ✅  Translator loaded.")


# ════════════════════════════════════════════════════════════════════════════
# 2. Inference Helpers
# ════════════════════════════════════════════════════════════════════════════

ANALYSIS_PROMPT = (
    "Provide a detailed radiology report for the given X-ray image and list the finding"
)


def run_medgemma(image: Image.Image) -> str:
    """Run MedGemma on an image and return the English analysis report."""
    import torch

    load_medgemma()

    # MedGemma / Gemma chat template
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": ANALYSIS_PROMPT},
            ],
        }
    ]

    # Apply Gemma chat template
    try:
        prompt_text = med_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = med_processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
        )
    except Exception:
        # Fallback: plain text + image
        inputs = med_processor(
            text=ANALYSIS_PROMPT,
            images=image,
            return_tensors="pt",
        )

    device = next(med_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = med_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.1,
        )

    # Strip the prompt tokens
    input_len = inputs["input_ids"].shape[1]
    gen_ids   = output_ids[0][input_len:]
    result    = med_processor.decode(gen_ids, skip_special_tokens=True).strip()
    return result


def translate_to_id(text: str) -> str:
    """Translate English text to Bahasa Indonesia in sentence-sized chunks."""
    load_translator()

    # Split by newlines preserving structure
    lines      = text.split("\n")
    translated = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            translated.append("")
            continue
        # Chunk long lines to avoid token limit
        chunks = textwrap.wrap(stripped, width=400) or [stripped]
        trans_chunks = []
        for chunk in chunks:
            tokens = trans_tokenizer([chunk], return_tensors="pt",
                                     padding=True, truncation=True, max_length=512)
            out    = trans_model.generate(**tokens, max_length=512)
            trans_chunks.append(
                trans_tokenizer.decode(out[0], skip_special_tokens=True)
            )
        translated.append(" ".join(trans_chunks))

    return "\n".join(translated)


def rag_retrieve(query: str, threshold: float = 0.20) -> list[dict]:
    """Search for the exact disease keywords first to avoid semantic blending (like pneumonia + effusion = pleuropneumonia)."""
    if embed_model is None or faiss_index is None:
        return []

    lower_q = query.lower()
    
    # -- 1. Explicit Keyword Mapping --
    # Map English clinical findings directly to their Indonesian PPK Disease counterparts
    keyword_map = {
        "pneumothorax": "Pneumotoraks",
        "atelectasis": "Atelektasis",
        "pleuropneumonia": "Pleuropneumonia",
        "pleural effusion": "Efusi Pleura",
        "pneumonia": "Pneumonia",
        "consolidation": "Konsolidasi",
    }
    
    found_diseases = set()
    
    # Carefully check for exact keywords (prioritize complex phrases)
    if "pleuropneumonia" in lower_q:
        found_diseases.add("Pleuropneumonia")
    else:
        if "pneumonia" in lower_q:
            found_diseases.add("Pneumonia")
        if "pleural effusion" in lower_q:
            found_diseases.add("Efusi Pleura")
            
    if "pneumothorax" in lower_q: found_diseases.add("Pneumotoraks")
    if "atelectasis" in lower_q: found_diseases.add("Atelektasis")
    if "consolidation" in lower_q: found_diseases.add("Konsolidasi")

    results = []
    
    # -- 2. If we confidently found keyword matches, lock onto those specifically --
    if found_diseases:
        for r in ppk_records:
            if r["disease"] in found_diseases:
                results.append({**r, "score": 1.0})
        return results

    # -- 3. Fallback: FAISS Vector Semantic Search --
    # Focus the query on the diagnosis part of the report if possible
    search_query = query
    if "differential diagnosis" in lower_q:
        search_query = query[lower_q.find("differential diagnosis"):]
    elif "conclusion" in lower_q:
        search_query = query[lower_q.find("conclusion"):]

    q_emb  = embed_model.encode([search_query], normalize_embeddings=True)
    q_emb  = np.array(q_emb, dtype=np.float32)
    scores, idxs = faiss_index.search(q_emb, 3)

    best_disease = None
    best_score = 0
    # Find the top single disease keyword that passes the similarity threshold
    for score, idx in zip(scores[0], idxs[0]):
        if idx != -1 and score >= threshold:
            best_disease = ppk_records[idx]["disease"]
            best_score = float(score)
            break  # Stop at the highest matching disease
            
    if not best_disease:
        return []
        
    # Now that we found the specific disease keyword, retrieve ALL its clinical sections
    results = []
    for r in ppk_records:
        if r["disease"] == best_disease:
            results.append({**r, "score": best_score})
            
    return results


def format_rag_english(records: list[dict]) -> str:
    """Returns empty to avoid duplicating the RAG blocks in the English section."""
    return ""


def format_rag_indonesian(records: list[dict]) -> str:
    """Format retrieved PPK records exactly to the user's requested template."""
    if not records:
        return ""

    by_disease: dict[str, dict] = {}
    for r in records:
        d = r["disease"]
        if d not in by_disease:
            by_disease[d] = {}
        section = r["section"].lower()
        if "anamnesis" in section:
            by_disease[d]["Anamnesis"] = r["content"]
        elif "tata laksana" in section or "tatalaksana" in section:
            by_disease[d]["Tata Laksana"] = r["content"]
        elif "daftar pustaka" in section or "referensi" in section:
            by_disease[d]["Referensi"] = r["content"]
        else:
            by_disease[d][r["section"]] = r["content"]

    lines = [""]
    for disease, sections in by_disease.items():
        lines.append("Retrieved PPK:\nPenyakit\n" + disease)
        
        if "Anamnesis" in sections:
            lines.append("Retrieved PPK: Anamnesis\n" + sections["Anamnesis"])
            
        if "Tata Laksana" in sections:
            lines.append("Retrieved PPK: Tata\nLaksana (Management)\n" + sections["Tata Laksana"])
            
        if "Referensi" in sections:
            lines.append("Retrieved PPK: Referensi\n" + sections["Referensi"])
            
    return "\n\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 3. Main Chatbot Logic
# ════════════════════════════════════════════════════════════════════════════

def analyze_xray(image, history):
    """
    Main handler called by Gradio.
    Returns a generator that streams status messages, then the final report.
    """
    if image is None:
        yield history + [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "⚠️ Please upload an X-ray image first."}
        ]
        return

    pil_image = Image.fromarray(image).convert("RGB")

    # Status updates
    yield history + [
        {"role": "user", "content": "[Analysing X-ray image…]"},
        {"role": "assistant", "content": "⏳ Analysing your X-ray image with MedGemma… (this may take 1–3 mins on first run)"}
    ]

    # ── Step 1: LLaVA Analysis (English) ─────────────────────────────────────
    try:
        english_report = run_medgemma(pil_image)
    except Exception as e:
        yield history + [
            {"role": "user", "content": "[Analysis]"},
            {"role": "assistant", "content": f"❌ Error during image analysis:\n```\n{e}\n```"}
        ]
        return

    # ── Step 2: RAG retrieval ─────────────────────────────────────────────────
    yield history + [
        {"role": "user", "content": "[Analysis]"},
        {"role": "assistant", "content": "⏳ Searching clinical guidelines (PPK)…"}
    ]

    rag_records = rag_retrieve(english_report, threshold=0.20)
    rag_en_block = format_rag_english(rag_records)
    rag_id_block = format_rag_indonesian(rag_records)

    # ── Step 3: Translate to Bahasa Indonesia ─────────────────────────────────
    yield history + [
        {"role": "user", "content": "[Analysis]"},
        {"role": "assistant", "content": "⏳ Translating to Bahasa Indonesia…"}
    ]

    indonesian_report = translate_to_id(english_report)

    # ── Step 4: Compose final output ─────────────────────────────────────────
    separator = "\n\n" + "═" * 60 + "\n"

    final_output = (
        "## 🇬🇧 RADIOLOGY REPORT (English)\n\n"
        + english_report
        + rag_en_block
        + separator
        + "## 🇮🇩 LAPORAN RADIOLOGI (Bahasa Indonesia)\n\n"
        + indonesian_report
        + rag_id_block
    )

    new_history = history + [
        {"role": "user", "content": "[X-Ray Analysis Complete]"},
        {"role": "assistant", "content": final_output}
    ]
    yield new_history


def reset_chat():
    return [], None


# ════════════════════════════════════════════════════════════════════════════
# 4. Gradio UI
# ════════════════════════════════════════════════════════════════════════════

CSS = """
/* ── Global Reset ──────────────────────────────────── */
* { box-sizing: border-box; }

body, .gradio-container {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #091a2a 100%) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    min-height: 100vh;
}

/* ── Header ─────────────────────────────────────────── */
#header-box {
    background: linear-gradient(135deg, #0a2540 0%, #0e3460 50%, #0a2540 100%);
    border: 1px solid rgba(0, 200, 255, 0.25);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(0, 150, 255, 0.12);
}

#header-box::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center,
        rgba(0, 180, 255, 0.06) 0%,
        transparent 60%);
    animation: pulse-bg 4s ease-in-out infinite;
}

@keyframes pulse-bg {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}

#header-title {
    font-size: 2.0rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    text-align: center !important;
    letter-spacing: -0.5px;
    text-shadow: 0 0 30px rgba(0, 200, 255, 0.5);
    margin: 0 !important;
}

#header-subtitle {
    text-align: center !important;
    color: rgba(150, 210, 255, 0.85) !important;
    font-size: 0.95rem !important;
    margin-top: 8px !important;
}

/* ── Image Upload ────────────────────────────────────── */
#xray-upload {
    border: 2px dashed rgba(0, 180, 255, 0.4) !important;
    border-radius: 14px !important;
    background: rgba(0, 30, 60, 0.6) !important;
    min-height: 280px !important;
    transition: border-color 0.3s, box-shadow 0.3s;
}

#xray-upload:hover {
    border-color: rgba(0, 200, 255, 0.8) !important;
    box-shadow: 0 0 20px rgba(0, 180, 255, 0.2) !important;
}

/* ── Buttons ─────────────────────────────────────────── */
#analyze-btn {
    background: linear-gradient(135deg, #005fa3 0%, #0080d0 50%, #00aaff 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 14px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 20px rgba(0, 150, 255, 0.35) !important;
}

#analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0, 180, 255, 0.5) !important;
}

#reset-btn {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    color: rgba(200, 220, 255, 0.8) !important;
    font-weight: 600 !important;
    padding: 10px 0 !important;
    width: 100% !important;
    transition: background 0.2s, border-color 0.2s;
}

#reset-btn:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: rgba(0, 180, 255, 0.4) !important;
}

/* ── Chatbot ─────────────────────────────────────────── */
#chat-window {
    border-radius: 14px !important;
    background: rgba(5, 15, 30, 0.7) !important;
    border: 1px solid rgba(0, 150, 255, 0.15) !important;
    min-height: 500px !important;
}

/* ── Info Panel ──────────────────────────────────────── */
#info-panel {
    background: rgba(0, 30, 60, 0.5) !important;
    border: 1px solid rgba(0, 150, 255, 0.2) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    color: rgba(180, 220, 255, 0.9) !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
}

/* ── Status Bar ──────────────────────────────────────── */
#status-bar {
    background: rgba(0, 50, 80, 0.4) !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
    color: rgba(100, 200, 255, 0.9) !important;
    font-size: 0.82rem !important;
    text-align: center !important;
    border: 1px solid rgba(0, 150, 255, 0.15) !important;
}

/* ── Disclaimer ──────────────────────────────────────── */
#disclaimer {
    background: rgba(180, 60, 0, 0.12) !important;
    border: 1px solid rgba(255, 120, 0, 0.25) !important;
    border-radius: 10px !important;
    padding: 12px 18px !important;
    color: rgba(255, 180, 100, 0.9) !important;
    font-size: 0.82rem !important;
    text-align: center !important;
}
"""

WELCOME_MSG = """
**Selamat datang di Medical X-Ray AI Chatbot 🏥**

Saya adalah asisten radiologi AI yang didukung oleh model **MedGemma** dan panduan klinis PPK.

---
**Cara Penggunaan / How to Use:**
1. Upload foto X-ray Anda di panel kiri / Upload your X-ray image on the left
2. Klik **"🔬 Analyse X-Ray"** untuk memulai analisis
3. Laporan akan muncul dalam bahasa **Inggris** dan **Indonesia**
4. Panduan PPK (Anamnesis, Tata Laksana, Daftar Pustaka) akan otomatis ditampilkan

---
> ⚠️ *Hasil analisis AI ini hanya sebagai alat bantu diagnosis. Keputusan klinis tetap berada di tangan dokter.*
"""


def build_ui():
    with gr.Blocks(
        title="Medical X-Ray AI — Radiology Assistant",
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        with gr.Column(elem_id="header-box"):
            gr.HTML("""
                <p id="header-title">🩻 Medical X-Ray AI Chatbot</p>
                <p id="header-subtitle">
                    Powered by MedGemma &nbsp;|&nbsp; RAG dari PPK &nbsp;|&nbsp;
                    Bilingual (EN / ID)
                </p>
            """)

        # ── Main Layout ───────────────────────────────────────────────────────
        with gr.Row():
            # Left column: image upload + controls
            with gr.Column(scale=1, min_width=320):
                image_input = gr.Image(
                    label="Upload X-Ray Image",
                    type="numpy",
                    elem_id="xray-upload",
                    height=300,
                )

                analyze_btn = gr.Button(
                    "🔬  Analyse X-Ray",
                    variant="primary",
                    elem_id="analyze-btn",
                )
                reset_btn = gr.Button(
                    "🗑️  Reset",
                    elem_id="reset-btn",
                )

                gr.HTML("""
                    <div id="info-panel">
                        <strong>📌 Supported formats:</strong><br>
                        JPEG · PNG · BMP · TIFF<br><br>
                        <strong>🔬 Model:</strong><br>
                        Gisela13154/medgemma-4b-it-sft-lora-ms-cxr-evaluation<br><br>
                        <strong>📋 RAG Source:</strong><br>
                        PPK COMPILE (Panduan Praktik Klinis)<br><br>
                        <strong>🌐 Translation:</strong><br>
                        Helsinki-NLP EN → Bahasa Indonesia
                    </div>
                """)

                gr.HTML("""
                    <div id="disclaimer">
                        ⚠️ Hanya alat bantu diagnosis — bukan pengganti dokter.<br>
                        For professional use only. Always verify with a physician.
                    </div>
                """)

                gr.HTML("""
                    <div id="status-bar">
                        Model loads automatically on first use · GPU recommended
                    </div>
                """)

            # Right column: chat output
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": WELCOME_MSG}],
                    elem_id="chat-window",
                    label="Analysis Report",
                    height=650,
                    show_label=False,
                )

        # ── Events ───────────────────────────────────────────────────────────
        analyze_btn.click(
            fn=analyze_xray,
            inputs=[image_input, chatbot],
            outputs=[chatbot],
        )

        reset_btn.click(
            fn=reset_chat,
            inputs=[],
            outputs=[chatbot, image_input],
        )

    return demo


# ════════════════════════════════════════════════════════════════════════════
# 5. Startup
# ════════════════════════════════════════════════════════════════════════════

def startup():
    """Load RAG + translator at startup (MedGemma loads lazily on first image upload)."""
    print("\n" + "═" * 60)
    print("   Medical X-Ray AI Chatbot  —  Starting Up")
    print("═" * 60)
    load_rag()
    load_translator()
    print("[App] ✅  Startup complete. MedGemma will load on first image upload.\n")


if __name__ == "__main__":
    startup()
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=CSS,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
    )
