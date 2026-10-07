# October task: policy corpus -> FAISS index -> Retriever Agent

## Run (VS Code terminal, Python 3.11)
    python -m venv .venv && .venv\Scripts\activate      # Linux/Mac: source .venv/bin/activate
    pip install -r requirements.txt
    python build_index.py                  # first run downloads multilingual-e5-small (~470 MB), then fully offline
    python retriever_agent.py "महिला व बालविकास विभागाच्या योजनेची पात्रता काय आहे?" -k 5
    python retriever_agent.py "Birsa Munda Krishi Kranti Yojana subsidy" --lang en

## Offline after first download
Set `HF_HUB_OFFLINE=1` (model is cached under ~/.cache/huggingface).

## Files
- data/chunks.jsonl : your existing 6,526 chunks (300 GRs, en+mr, 3 departments)
- common.py         : cleaning, embedder, settings (MODEL_NAME, MIN_WORDS)
- build_index.py    : clean + dedupe + embed + FAISS (IndexFlatIP, cosine) -> index/
- retriever_agent.py: RetrieverAgent.retrieve(query, k, lang, department) -> JSON evidence + latency_ms
`--dummy` flag = pipeline smoke test only (no semantics).


## October 2026 Task — Setup & Execution

### Objective

Build a local multilingual retrieval system for Maharashtra Government policy/GR documents.

Pipeline:

Policy Data → Cleaning → Chunking → Embeddings → FAISS Index → Retriever Agent

### Setup

Create virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate

