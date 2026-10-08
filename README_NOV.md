# November task: Generator Agent (local quantised Llama) + Retriever integration

Copy these files INTO your existing oct_task folder (do not delete `index/`). Then, in the activated venv:

1. pip install -r requirements_nov.txt
2. pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
   (--only-binary forces a prebuilt wheel, so Windows will not try to compile C++)
3. python -c "from llama_cpp import Llama; print('llama OK')"
4. python download_model.py                  (one time, ~2 GB; internet needed only now)
5. python pipeline.py "अंगणवाडी सेविकांना मानधन किती मिळते?"
6. python evaluate_nov.py --no-gen           (retrieval metrics, fast)
7. python evaluate_nov.py                    (full: generation + abstention)
8. Try the synopsis model: python download_model.py llama-3.1-8b ; python pipeline.py "..." --model llama-3.1-8b

Plumbing check without any LLM: python pipeline.py "..." --mock
Logs for the thesis: logs/runs.jsonl (every query), logs/eval_summary.json, logs/eval_results.jsonl
