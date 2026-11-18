"""Sanity check of the Tier-1 NLI model on known pairs (English + Marathi), float32 vs int8.
Usage:  python nli_selftest.py
Good model: clear entailment pairs get entailment > 0.6, clear contradictions get contradiction > 0.6.
If int8 accuracy is clearly lower than float32, keep int8 OFF (default). Report both rows in the paper (ablation).
"""
import time, warnings
warnings.filterwarnings("ignore")
from hallu_config import NLI_TIER1
from verifier_agent import NLIModel

PAIRS = [  # (premise, hypothesis, expected)
 ("Anganwadi workers will receive an honorarium of Rs. 10,000 per month.", "Anganwadi workers get Rs. 10,000 per month.", "entailment"),
 ("Anganwadi workers will receive an honorarium of Rs. 10,000 per month.", "Anganwadi workers get Rs. 15,000 per month.", "contradiction"),
 ("Anganwadi workers will receive an honorarium of Rs. 10,000 per month.", "The scheme is only for Mumbai city.", "neutral"),
 ("अंगणवाडी सेविकांचे मानधन दरमहा रु. 10,000 करण्यात आले आहे.", "अंगणवाडी सेविकांना दरमहा रु. 10,000 मानधन मिळते.", "entailment"),
 ("अंगणवाडी सेविकांचे मानधन दरमहा रु. 10,000 करण्यात आले आहे.", "अंगणवाडी सेविकांना दरमहा रु. 15,000 मानधन मिळते.", "contradiction"),
 ("अंगणवाडी मदतनीस यांना दरमहा रु. 5,500 मानधन देण्यात येईल.", "मदतनीस यांना दरमहा रु. 5,500 मानधन मिळेल.", "entailment"),
 ("अंगणवाडी मदतनीस यांना दरमहा रु. 5,500 मानधन देण्यात येईल.", "मदतनीस यांना कोणतेही मानधन देण्यात येणार नाही.", "contradiction"),
 ("अंगणवाडी मदतनीस यांना दरमहा रु. 5,500 मानधन देण्यात येईल.", "ही योजना फक्त मुंबई शहरात लागू आहे.", "neutral"),
 ("अंगणवाडी सेविकांचे मानधन दरमहा रु. 10,000 करण्यात आले आहे.", "Anganwadi workers get Rs. 10,000 per month.", "entailment"),
]

for q in (False, True):
    m = NLIModel(NLI_TIER1, quantize=q)
    t = time.perf_counter()
    preds = m.predict([(p, h) for p, h, _ in PAIRS])
    ms = (time.perf_counter() - t) * 1000 / len(PAIRS)
    ok = 0
    print(f"\n=== {'int8' if m.quantized else 'float32'}  ({ms:.0f} ms/pair) ===")
    for (p, h, exp), pr in zip(PAIRS, preds):
        top = max(pr, key=pr.get); ok += top == exp
        print(f"{'OK ' if top == exp else 'XX '} exp={exp:<13} got={top:<13} "
              f"E={pr['entailment']:.2f} N={pr['neutral']:.2f} C={pr['contradiction']:.2f} | {h[:45]}")
    print(f"accuracy {ok}/{len(PAIRS)}")
