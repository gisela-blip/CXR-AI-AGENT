# 🩻 Medical X-Ray AI Chatbot

A bilingual (English / Bahasa Indonesia) AI-powered radiology assistant that:

- **Analyses chest X-ray images** using the fine-tuned **MedGemma** model (`Gisela13154/medgemma-4b-it-sft-lora-ms-cxr-evaluation`)
- **Augments every report** with relevant clinical guidelines from **PPK COMPILE.xlsx** via a FAISS RAG pipeline
- **Translates** the full report from English → Bahasa Indonesia automatically

---

## 📁 Project Structure

```
AI AGENT/
├── app.py                  # Main chatbot application (Gradio UI)
├── rag_builder.py          # Builds FAISS index from PPK COMPILE.xlsx
├── requirements.txt        # All Python dependencies (fully documented)
├── install.bat             # One-click Windows installer
├── PPK COMPILE.xlsx        # PPK clinical guidelines source data
├── faiss_index/            # Auto-generated FAISS vector store
│   ├── ppk.index           #   └── FAISS binary index
│   └── ppk_records.pkl     #   └── Serialised PPK records
└── README.md               # This file
```

---

## ⚙️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.13 |
| RAM | 16 GB | 32 GB |
| GPU VRAM | 8 GB | 16 GB |
| GPU | Optional | NVIDIA CUDA |
| Disk | 20 GB free | 30 GB free |

> **No GPU?** The app runs in CPU mode automatically — analysis will take ~5–10 min per image.

> **⚠️ Model Access:** `medgemma-4b-it` is a gated model on HuggingFace. If the LoRA load path fails, you need to [accept the licence](https://huggingface.co/google/medgemma-4b-it) on HuggingFace and run `huggingface-cli login` in your terminal first.

---

## 🚀 Quick Start

### Step 1 — Install Dependencies

**Option A: Use the install script (Windows)**
```bat
install.bat
```

**Option B: Manual install**
```powershell
# Install PyTorch first (choose one):
# GPU (CUDA 11.8):
py -3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only:
py -3 -m pip install torch torchvision torchaudio

# Then install all other packages:
py -3 -m pip install -r requirements.txt
```

### Step 2 — Build the RAG Index

```powershell
py -3 rag_builder.py
```

This reads `PPK COMPILE.xlsx` and creates the FAISS vector index in `./faiss_index/`.  
You only need to run this **once** (or whenever PPK data changes).

### Step 3 — Run the Chatbot

```powershell
py -3 app.py
```

The app will open automatically in your browser at `http://localhost:7860`.

---

## 🔄 How It Works

```
User uploads X-ray image
        │
        ▼
┌─────────────────────────────┐
│  LLaVA-Med                  │  ← Gisela13154/retune-LlaVA-Med-1.5B-7B-it
│  Visual Language Model      │     Runs structured radiology prompt
│  Output: English report     │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  FAISS RAG Engine           │  ← PPK COMPILE.xlsx → sentence-transformers
│  Finds relevant diseases    │     Retrieves Anamnesis, Tata Laksana,
│  from PPK database          │     Daftar Pustaka
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Helsinki-NLP Translator    │  ← opus-mt-en-id (offline, no API key)
│  EN → Bahasa Indonesia      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Gradio UI                  │  Displays bilingual report:
│  (Dark Medical Theme)       │  🇬🇧 English + PPK (EN)
│                             │  🇮🇩 Indonesian + PPK (ID)
└─────────────────────────────┘
```

---

## 📋 PPK RAG — Data Format

The `PPK COMPILE.xlsx` file must have 3 columns:

| Column | Description |
|--------|-------------|
| `Disease_key` | Name of the disease (only filled in first row of each disease block) |
| `Section` | One of: `Anamnesis`, `Tata Laksana`, `Daftar Pustaka` |
| `Content` | Clinical guideline text (in Bahasa Indonesia) |

---

## 🤖 Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| `Gisela13154/medgemma-4b-it-sft-lora-ms-cxr-evaluation` | X-ray image analysis | HuggingFace |
| `sentence-transformers/all-MiniLM-L6-v2` | RAG embeddings | HuggingFace |
| `Helsinki-NLP/opus-mt-en-id` | EN → ID translation | HuggingFace |

All models are **downloaded automatically** on first run and cached locally.

---

## 🛠️ Troubleshooting

### `CUDA out of memory`
MedGemma-4b fits in ~8 GB VRAM in full precision. Lower `max_new_tokens` in `app.py` if needed.

### HuggingFace login required
If loading `google/medgemma-4b-it` (LoRA fallback path):
```powershell
py -3 -m huggingface_hub.commands.huggingface_cli login
```
Then paste your HuggingFace token.

### `ModuleNotFoundError`
Run `install.bat` or `py -3 -m pip install -r requirements.txt`.

### FAISS index missing
Run `py -3 rag_builder.py` to rebuild the index.

### Slow on first run
Models are ~15 GB total. First run involves downloading + loading all models.  
Subsequent runs are faster as models are cached.

---

## ⚠️ Disclaimer

This chatbot is an **AI-assisted diagnostic tool** only.  
All output should be reviewed by a qualified radiologist or physician.  
It does not replace professional medical judgment.

---

## 📄 License

For research and clinical support purposes only.  
Model weights governed by their respective HuggingFace licenses.
