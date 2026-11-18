"""Fast self-test of the whole 5-agent graph with mock models (no LLM / NLI download needed).
Run from the project folder:  python tests/smoke_test.py      -> prints PASS / FAIL per check
Needs index/ (python build_index.py or --dummy)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orchestrator import Orchestrator
from text_utils import split_claims, numbers

ok = True
def check(name, cond):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name)

check("Devanagari digits + Indian commas", numbers("रु. १,००,००० व 5,500") == {"100000", "5500"})
check("claim split keeps citations", split_claims("A gets Rs. 10,000 per month [1]. B gets Rs. 5,500 [2].")[1]["cites"] == [2])
o = Orchestrator(mock_llm=True, mock_nli=True)
q = "अंगणवाडी सेविकांना मानधन किती मिळते?"
r = o.run(q, log=False)
check(f"normal run ({o.engine}) -> {r['verdict']}", r["verdict"] in ("VERIFIED", "PARTIALLY_VERIFIED", "HALLUCINATED", "ABSTAINED"))
def swap(ans, ev):
    import re
    return re.sub(r"\d[\d,]*\d", lambda m: str(int(m.group().replace(",", "")) * 3 + 7), ans, count=1)
r = o.run(q, inject=swap, log=False)
check(f"injected wrong number detected -> {r['verdict']}", r["claims"] == [] or r["verdict"] == "HALLUCINATED")
r = o.run(q, budget_s=0.0001, log=False)
check(f"tiny budget -> graceful degradation -> {r['verdict']}", r["verdict"] in ("UNVERIFIED", "HALLUCINATED", "ABSTAINED"))
r = o.run(q, policy="no_verify", log=False)
check("no_verify baseline", r["verdict"] in ("NOT_CHECKED", "ABSTAINED"))
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED"); sys.exit(0 if ok else 1)
