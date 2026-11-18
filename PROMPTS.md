# Prompts used in this project

## A. Runtime prompts (inside generator_agent.py) - these are what Llama actually receives

### System prompt
    You are a careful assistant for Maharashtra Government scheme information.
    Rules:
    1. Answer ONLY using the numbered EVIDENCE below. Do not use outside knowledge.
    2. After each fact write the evidence number in brackets, like [1] or [2].
    3. Copy numbers, amounts, dates and names exactly as written in the evidence.
    4. If the evidence does not contain the answer, reply with exactly: NOT_FOUND
    5. Reply in the same language as the question (Marathi question -> Marathi answer, English -> English).
    6. Keep the answer short (2-5 sentences).

### User prompt template
    EVIDENCE:
    [1] <top-1 chunk text>
    [2] <top-2 chunk text>
    [3] <top-3 chunk text>

    QUESTION: <user question>

    ANSWER:
Settings: temperature 0.1, max 256 new tokens, context 4096, top-3 Marathi evidence chunks.
Lowest-ranked chunks are dropped automatically if the prompt would exceed the context window.

## B. Prompts to rebuild the code with an AI coding assistant (Claude / ChatGPT / Copilot)
Always attach: the synopsis, the project PPT, and `data/chunks.jsonl` (or its first 5 lines).

### B1. October milestone
    Project: "Design and Implement Resource-Constrained Heterogeneous Multi-Agent Framework for Hallucination
    Detection in Multilingual Public Policy Response" (masters project, Maharashtra government GRs, English + Marathi).
    Task (October): policy document corpus collection and indexing; text cleaning, chunking and building a local
    FAISS vector index for the Retriever Agent.
    Input: data/chunks.jsonl (fields: chunk_id, doc_id, lang, department, gr_no, page_start, page_end, n_words, text).
    Environment: Windows laptop, 8 GB RAM, CPU only, Python 3.10, VS Code, must work fully offline after first setup.
    Build: (1) build_index.py: clean text (unicode NFC, zero-width chars, dash/dot runs), drop chunks <20 words and
    exact duplicates, embed with intfloat/multilingual-e5-small ("passage: " prefix, normalised), IndexFlatIP, save
    index + metadata + build_info. (2) retriever_agent.py: RetrieverAgent.retrieve(query, k, lang, department) with the
    "query: " prefix, returning JSON with ranked evidence (score, chunk_id, doc_id, pages, text), top_score and latency_ms.
    Give exact VS Code/PowerShell commands, expected output, and a pass/fail test with Marathi and English queries.

### B2. November milestone
    Continue the same project (files above already exist). Task (November): set up a locally hosted quantised Llama
    as the Generator Agent and integrate it with the Retriever Agent for evidence-grounded response generation.
    Constraints: CPU only, 8 GB RAM (also support 16 GB with Llama-3.1-8B), Windows, Python 3.10, no compiler,
    use llama-cpp-python prebuilt wheel and GGUF Q4_K_M models from Hugging Face, offline after one-time download.
    Build: nov_config.py (model registry, context, temperature), download_model.py, generator_agent.py (strict grounded
    prompt with [n] citations and NOT_FOUND abstention, context-budget trimming of evidence, latency, mock mode for
    testing), pipeline.py (retriever -> generator, JSON log per run in logs/runs.jsonl, CLI and --chat),
    evaluate_nov.py (Hit@1, Hit@3, MRR for retrieval; abstention rate on out-of-scope queries; latency) with
    eval_queries.json (in-scope Marathi/English queries with expected doc_id + 2 out-of-scope).
    Do NOT claim anything works without telling me which parts you could not run.

### B3. Debugging prompt (use whenever something fails)
    I am running <command> in VS Code PowerShell on Windows (Python 3.10, venv active, 8/16 GB RAM).
    Goal: <what you want>. Full terminal output:
    <paste everything, not only the last line>
    Explain what the error means in simple words, give the smallest fix, and tell me how to verify it worked.

### B4. Next milestone (December) starting prompt
    Same project. Add the Verifier Agent: DeBERTa NLI (entailment/contradiction/neutral) checking each sentence of the
    Generator's answer against the retrieved evidence, running on CPU within an 8 GB RAM budget, producing a
    faithfulness score and hallucination label. Use my manually labelled answers in logs/eval_results.jsonl as ground truth.
