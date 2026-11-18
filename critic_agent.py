"""Critic Agent (Jan milestone): turns the Verifier's per-claim scores into
   (1) an error TYPE per claim (hallucination taxonomy),
   (2) an overall RISK level and VERDICT for the response, with a human-readable reason.
Detection only - nothing is rewritten. Pure rules -> fully explainable / auditable (TRiSM "explainability").

Taxonomy (reported in the paper):
  SUPPORTED              claim entailed by the evidence
  NUMERIC_FABRICATION    amount / date / count that appears in NO retrieved evidence, or a contradicted number
  CONTRADICTION          evidence states the opposite (intrinsic hallucination)
  UNSUPPORTED            evidence neither supports nor contradicts (extrinsic hallucination / outside knowledge)
  CITATION_MISMATCH      claim is supported, but not by the evidence number it cites
  UNVERIFIABLE           verifier was uncertain
  UNCHECKED              latency budget expired before this claim was checked (graceful degradation)
"""

SEVERE = {"NUMERIC_FABRICATION", "CONTRADICTION"}
SOFT = {"UNSUPPORTED", "UNVERIFIABLE"}

VERDICT_TEXT = {
    "VERIFIED": ("Verified", "सत्यापित"),
    "PARTIALLY_VERIFIED": ("Partially verified", "अंशतः सत्यापित"),
    "HALLUCINATED": ("Hallucination detected", "चुकीची माहिती आढळली"),
    "UNVERIFIED": ("Unverified (time budget exceeded)", "असत्यापित (वेळ मर्यादा संपली)"),
    "ABSTAINED": ("No answer in the policy documents", "धोरण दस्तऐवजांमध्ये उत्तर नाही"),
    "NOT_CHECKED": ("Not checked (verification off)", "तपासणी केली नाही"),
}


def claim_error_type(c):
    if c["status"] == "unchecked":
        return "UNCHECKED"
    lab, t0 = c["label"], c["tier0"]
    if t0["missing_numbers"] and lab != "SUPPORTED":
        return "NUMERIC_FABRICATION"      # amount/date/count found in NO retrieved evidence
    if lab == "CONTRADICTED":
        return "CONTRADICTION"
    if lab == "NOT_SUPPORTED":
        return "UNSUPPORTED"
    if lab == "UNCERTAIN":
        return "NUMERIC_FABRICATION" if t0["missing_numbers"] else "UNVERIFIABLE"
    if lab in ("SUPPORTED", "LIKELY_SUPPORTED"):
        return "CITATION_MISMATCH" if (t0["has_citation"] and not t0["cited_ok"]) else "SUPPORTED"
    return "UNVERIFIABLE"


def reason(c, et):
    t0 = c["tier0"]
    if et == "NUMERIC_FABRICATION":
        nums = ", ".join(t0["missing_numbers"]) or "a number"
        return f"Number(s) {nums} not found in the retrieved policy evidence."
    if et == "CONTRADICTION":
        return "The cited policy text states something different."
    if et == "UNSUPPORTED":
        return "Not stated in the retrieved policy documents (possible outside knowledge)."
    if et == "CITATION_MISMATCH":
        return "Supported by the evidence, but by a different source than the one cited."
    if et == "UNVERIFIABLE":
        return "Verifier could not decide with enough confidence."
    if et == "UNCHECKED":
        return "Skipped: latency budget exhausted (graceful degradation)."
    return "Supported by the policy evidence."


class CriticAgent:
    def assess(self, claims, abstained, budget_exceeded, policy, retrieval_top_score=None, gen_confidence=None):
        for c in claims:
            c["error_type"] = claim_error_type(c)
            c["reason"] = reason(c, c["error_type"])
        types = [c["error_type"] for c in claims]
        checked = [c for c in claims if c["status"] != "unchecked"]
        # faithfulness = mean P(entail) over checked claims (tier-0-only claims count 1 / 0.5 / 0)
        def ent(c):
            if c.get("probs"):
                return c["probs"]["entailment"]
            return {"LIKELY_SUPPORTED": 1.0, "UNCERTAIN": 0.5}.get(c["label"], 0.0)
        faith = round(sum(ent(c) for c in checked) / len(checked), 3) if checked else None
        flagged = [t for t in types if t in SEVERE | SOFT]
        hall_rate = round(len(flagged) / len(claims), 3) if claims else 0.0

        if abstained:
            verdict, risk = "ABSTAINED", "LOW"
        elif policy == "no_verify":
            verdict, risk = "NOT_CHECKED", "UNKNOWN"
        elif any(t in SEVERE for t in types):
            verdict, risk = "HALLUCINATED", "HIGH"
        elif budget_exceeded and "UNCHECKED" in types:
            verdict, risk = "UNVERIFIED", "MEDIUM"
        elif any(t in SOFT for t in types) or not claims:
            verdict, risk = "PARTIALLY_VERIFIED", "MEDIUM"
        else:
            verdict, risk = "VERIFIED", "LOW"
        # depth of verification actually achieved (honest reporting: tier-0 only != NLI-verified)
        tiers = [c.get("tier", 0) for c in checked]
        depth = "none" if not checked else ("heuristic" if max(tiers) == 0 else f"nli-tier{max(tiers)}")
        notes = []
        if abstained and retrieval_top_score and retrieval_top_score > 0.86:
            notes.append("Strong retrieval match but the generator abstained - possible over-abstention.")
        if gen_confidence is not None and gen_confidence < 0.5 and verdict == "VERIFIED":
            notes.append("Low generator confidence despite verified claims.")
        en, mr = VERDICT_TEXT[verdict]
        return {"verdict": verdict, "verdict_en": en, "verdict_mr": mr, "risk": risk,
                "faithfulness": faith, "hallucination_rate": hall_rate, "verification_depth": depth,
                "error_counts": {t: types.count(t) for t in sorted(set(types))}, "notes": notes}
