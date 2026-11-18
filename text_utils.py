"""Language-agnostic text helpers for Marathi (Devanagari) + English: claim splitting, number
normalisation, lexical support and premise selection. Pure Python, no models -> Tier-0 of the verifier.
"""
import re

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
CITE_RE = re.compile(r"\[(\d+)\]")
# sentence ends: . ? ! danda (।) double danda (॥) and newlines; keep decimals like 2.5 intact
SENT_SPLIT_RE = re.compile(r"(?<=[।॥?!])\s+|(?<=\.)\s+(?=[^\d\s])|\n+")   # "रु. 10,000" is not split
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
TOKEN_RE = re.compile(r"[\wऀ-ॿ]+")
# very common words that carry no evidence value (English + Marathi)
STOP = set("""the a an of to in and or is are was were be for on with by as at this that it its from which
will shall has have had per under scheme yojana government maharashtra
व आणि या ही हा हे ते की चा ची चे च्या ला ना ने त मध्ये साठी करिता आहे आहेत होते असे म्हणून तसेच
योजना योजनेत योजनेसाठी शासन महाराष्ट्र राज्य बाबत करण्यात येत आली""".split())


def norm_digits(t: str) -> str:
    return t.translate(DEV_DIGITS)


def numbers(t: str) -> set:
    """All numbers in a text, normalised: Devanagari -> ASCII digits, '1,00,000' -> '100000', '5.0' -> '5'."""
    out = set()
    for n in NUM_RE.findall(norm_digits(t)):
        n = n.replace(",", "").rstrip(".")
        if not n:
            continue
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


def tokens(t: str) -> list:
    return [w for w in TOKEN_RE.findall(norm_digits(t).lower()) if w not in STOP and len(w) > 1]


def stem(w: str) -> str:
    """Crude prefix stem: Marathi attaches case suffixes (योजनेसाठी, सेविकांना); first 5 chars match the root well enough."""
    return w[:5] if len(w) > 5 else w


def strip_citations(t: str) -> str:
    return re.sub(r"\s+([.।,])", r"\1", CITE_RE.sub("", t)).strip()


def split_sentences(text: str) -> list:
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s and len(s.strip()) > 2]


def split_claims(answer: str, min_tokens: int = 3) -> list:
    """Answer -> atomic-ish claims (sentence level, as in the synopsis). Keeps the [n] citations of each claim."""
    claims = []
    answer = re.sub(r"([.।])\s*((?:\[\d+\]\s*)+)", r" \2\1 ", answer)   # "आहे. [1]" -> "आहे [1]."
    for s in split_sentences(answer):
        cites = sorted({int(c) for c in CITE_RE.findall(s)})
        txt = strip_citations(s)
        if len(tokens(txt)) >= min_tokens or numbers(txt):
            claims.append({"id": len(claims), "text": txt, "cites": cites})
    return claims


def lexical_support(claim: str, sentence: str) -> float:
    """Share of the claim's content words (stemmed) present in the evidence sentence. 0..1"""
    c = {stem(w) for w in tokens(claim)}
    if not c:
        return 0.0
    s = {stem(w) for w in tokens(sentence)}
    return len(c & s) / len(c)


def evidence_sentences(evidence: list) -> list:
    """Flatten evidence chunks into sentences, remembering the 1-based evidence number."""
    out = []
    for i, e in enumerate(evidence, 1):
        for s in split_sentences(e["text"]):
            out.append({"ev": i, "chunk_id": e.get("chunk_id"), "text": s})
    return out


def select_premise(claim: str, ev_sents: list, k: int, prefer_ev=None):
    """Pick the k evidence sentences that best support/contradict the claim.
    Score = lexical overlap + number overlap bonus + small bonus if it is from a cited evidence."""
    cn = numbers(claim)
    scored = []
    for i, s in enumerate(ev_sents):
        sc = lexical_support(claim, s["text"])
        if cn:
            sc += 0.5 * len(cn & numbers(s["text"])) / len(cn)
        if prefer_ev and s["ev"] in prefer_ev:
            sc += 0.1
        scored.append((sc, i))
    scored.sort(reverse=True)
    top = sorted(i for _, i in scored[:k])          # keep document order -> readable premise
    best = scored[0][0] if scored else 0.0
    return [ev_sents[i] for i in top], best
