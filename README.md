# 🐛 MoultGPT

**MoultGPT** is a modular NLP pipeline for extracting biologically relevant traits related to moulting in arthropods from scientific literature. It supports PDF parsing, sentence summarization, trait extraction using a local LLM (Mistral 7B), and can be fine-tuned using LoRA on a domain-specific dataset.

---

## 🚀 Features

- 🧠 Local inference using Mistral 7B (HF Transformers)
- 📄 PDF parsing with GROBID
- 🔍 Sentence-level summarization using TF-IDF + KMeans
- ✍️ Trait extraction in YAML format
- 🧪 Fine-tuning with LoRA on custom data
- 🌐 Interactive frontend (React) + backend (Flask)

---

## ⚙️ Environment Setup (Python virtualenv)

```bash
# Clone the repo
git clone https://github.com/your-user/MoultGPT.git
cd MoultGPT

# Create and activate virtual environment
python3 -m venv mistral_env
source mistral_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧠 Running the backend (Flask + LLM)

```bash
cd backend
python app.py
```

Make sure you have the base model downloaded into:

```
/reference/LLMs/Mistral_AI/mistral-7B-Instruct-v0.3-hf/
```

You can change this path in `app.py` if needed.

---

## 📚 Parsing PDFs with GROBID (CLI mode)

Start GROBID manually (once Java is installed):

```bash
cd tools/grobid/grobid-0.7.1
./gradlew run
```

GROBID will be available on: http://localhost:8070

---

## 🔁 Running Fine-tuning with LoRA

Make sure `finetune_full.jsonl` exists in the root.

```bash
python main_generate_dataset.py
```

You can monitor the job in `finetuning/finetune_output.log`.

---

## 🖼️ Frontend (React)

```bash
cd frontend
npm install
npm start
```

The frontend will be available on: http://localhost:3000

---

## 📁 Project Structure

```
LLM/
├── backend/               # Flask server + LLM pipeline
├── frontend/              # React GUI for querying
├── finetuning/            # LoRA scripts and training data
├── paper_handler/         # PDF processing, sentence summarization
├── tools/grobid/          # GROBID installation for parsing PDFs
├── data/, output/, papers/ # Storage for input/output files
├── requirements.txt       # Pip-based environment
└── README.md              # This file
```

---

## 🤝 Citation / Acknowledgements

This tool uses:

- [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [GROBID](https://github.com/kermitt2/grobid)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [scikit-learn](https://scikit-learn.org/)

---

## 📬 Contact

For collaborations, bug reports, or questions:  
Michele Leone – [michele.leone@outlook.com]  
Project website: [moulting.org](https://www.moulting.org)
