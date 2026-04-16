import json
import re
from pathlib import Path

INPUT  = "bail_corpus.jsonl"
OUTPUT = "bail_rag_ready.json"

# ── SECTION CLASSIFIER ───────────────────────────────────────────────────────
# Returns ALL matching sections, not just the first one

def classify_sections(sections: list) -> list[str]:
    """Map raw section numbers to human-readable labels."""
    mapping = {
        "436":  "436 CrPC - Bail in bailable offences",
        "437":  "437 CrPC - Bail in non-bailable offences",
        "438":  "438 CrPC - Anticipatory bail",
        "439":  "439 CrPC - HC/Sessions bail powers",
        "167":  "167 CrPC - Default bail",
        "389":  "389 CrPC - Bail pending appeal",
        "309":  "309 CrPC - Bail during trial",
        "440":  "440 CrPC - Amount of bail",
        "437(6)": "437(6) CrPC - Bail if trial not concluded in 60 days",
        "37":   "37 NDPS - Bail in NDPS (twin conditions)",
        "45":   "45 PMLA - Bail in PMLA (twin conditions)",
        "43D":  "43D UAPA - Bail in UAPA",
        "29":   "29 POCSO - Presumption of guilt (bail impact)",
        "18":   "18 SC/ST Act - No anticipatory bail",
    }
    labels = []
    for sec in sections:
        sec_str = str(sec).strip()
        for key, label in mapping.items():
            if sec_str == key or sec_str.startswith(key + "("):
                if label not in labels:
                    labels.append(label)
    return labels if labels else ["Other"]

# ── CATEGORY CLASSIFIER ───────────────────────────────────────────────────────
# Must use court_type field, NOT text content, for SC vs HC detection

def classify_category(chunk: dict) -> str:
    court_type = chunk.get("court_type", "")
    text       = chunk.get("chunk_text", "").lower()
    sections   = [str(s) for s in chunk.get("sections", [])]

    # Offence-specific — check text for specific markers
    if any(kw in text for kw in ["ndps", "narcotic", "commercial quantity", "section 37 ndps"]):
        return "NDPS"
    if any(kw in text for kw in ["pocso", "protection of children", "sexual offences act"]):
        return "POCSO"
    if any(kw in text for kw in ["pmla", "money laundering", "enforcement directorate", "section 45"]):
        return "PMLA"
    if any(kw in text for kw in ["uapa", "unlawful activities", "terrorist", "section 43d"]):
        return "UAPA"
    if any(kw in text for kw in ["498a", "498-a", "domestic violence", "dowry"]):
        return "Domestic Violence"
    if any(kw in text for kw in ["ipc 302", "section 302", "murder", "culpable homicide"]):
        return "Murder / IPC 302"
    if any(kw in text for kw in ["cheating", "420", "economic offence", "fraud", "corruption"]):
        return "Economic Offence"

    # Section-based
    if "438" in sections:
        return "Anticipatory Bail"
    if "167" in sections:
        return "Default Bail"
    if "437(6)" in sections:
        return "Default Bail - 60 Day"

    # Court-based — only use court_type, not text
    if court_type == "Supreme Court":
        return "SC - General Bail"
    if court_type == "High Court":
        return "HC - General Bail"

    return "General"

# ── BAIL OUTCOME EXTRACTOR ────────────────────────────────────────────────────
# Tries to detect whether bail was granted or denied in this chunk

def extract_bail_outcome(text: str) -> str:
    text_l = text.lower()
    granted_signals = [
        "bail is granted", "released on bail", "bail granted",
        "direct.*release.*bail", "bail application.*allowed",
        "bail is allowed", "ordered to be released"
    ]
    denied_signals = [
        "bail is rejected", "bail application.*dismissed",
        "bail denied", "rejected.*bail", "not entitled to bail",
        "bail is refused", "declined.*bail"
    ]
    for sig in granted_signals:
        if re.search(sig, text_l):
            return "granted"
    for sig in denied_signals:
        if re.search(sig, text_l):
            return "denied"
    return "unknown"

# ── KEY LEGAL PRINCIPLES TAGGER ───────────────────────────────────────────────
# Tags which bail principles this chunk discusses — critical for RAG retrieval

def tag_principles(text: str) -> list[str]:
    text_l = text.lower()
    tags = []
    principle_map = {
        "triple_test":        ["flight risk", "tampering", "repeat offence", "triple test"],
        "bail_is_rule":       ["bail is rule", "jail is exception", "personal liberty"],
        "parity":             ["parity", "co-accused", "similarly placed"],
        "anticipatory_bail":  ["anticipatory bail", "apprehending arrest", "section 438"],
        "default_bail":       ["default bail", "indefeasible right", "167(2)", "sixty days"],
        "economic_offence":   ["economic offence", "serious economic", "gravity of offence"],
        "medical_bail":       ["medical bail", "health condition", "sick", "infirm"],
        "surety":             ["surety", "bail bond", "amount of bail", "personal bond"],
        "cancellation":       ["cancellation of bail", "supervening", "misuse of bail"],
        "conditions":         ["bail condition", "condition of bail", "section 437(3)"],
        "arrest_guidelines":  ["arnesh kumar", "41a", "not arrested", "necessity of arrest"],
    }
    for tag, keywords in principle_map.items():
        if any(kw in text_l for kw in keywords):
            tags.append(tag)
    return tags

# ── MAIN ENRICHMENT PIPELINE ──────────────────────────────────────────────────

def enrich(input_path: str, output_path: str):
    records = []
    skipped = 0

    for line in Path(input_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        text = chunk.get("chunk_text", "")

        # ── Skip low-quality chunks ──
        if len(text.split()) < 60:
            skipped += 1
            continue

        # ── Enrich ──
        chunk["section_labels"] = classify_sections(chunk.get("sections", []))
        chunk["category"]       = classify_category(chunk)
        chunk["bail_outcome"]   = extract_bail_outcome(text)
        chunk["principles"]     = tag_principles(text)

        # ── RAG context string ──
        # This is what gets prepended to chunk_text when feeding to LLM
        chunk["rag_context"] = (
            f"[Case: {chunk.get('case_name','')}] "
            f"[Court: {chunk.get('court','')}] "
            f"[Date: {chunk.get('date','')}] "
            f"[Sections: {', '.join(chunk.get('section_labels',[]))}] "
            f"[Category: {chunk.get('category','')}] "
            f"[Authority: {chunk.get('authority','')}]"
        )

        records.append(chunk)

    # ── Write ──
    Path(output_path).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8"
    )

    # ── Stats ──
    from collections import Counter
    cats     = Counter(r["category"]     for r in records)
    outcomes = Counter(r["bail_outcome"] for r in records)
    auth     = Counter(r["authority"]    for r in records)
    principles_flat = [p for r in records for p in r["principles"]]
    top_principles  = Counter(principles_flat).most_common(8)

    print(f"\n{'─'*55}")
    print(f"  Total chunks enriched : {len(records)}")
    print(f"  Skipped               : {skipped}")
    print(f"\n  Category breakdown:")
    for cat, n in cats.most_common():
        print(f"    {cat:<28} {n:>4} chunks")
    print(f"\n  Bail outcome detected:")
    for k, v in outcomes.items():
        print(f"    {k:<12} {v:>4} chunks")
    print(f"\n  Authority levels:")
    for k, v in auth.items():
        print(f"    {k:<10} {v:>4} chunks")
    print(f"\n  Top legal principles tagged:")
    for p, n in top_principles:
        print(f"    {p:<25} {n:>4} chunks")
    print(f"\n  Output: {output_path}")
    print(f"{'─'*55}\n")

if __name__ == "__main__":
    enrich(INPUT, OUTPUT)