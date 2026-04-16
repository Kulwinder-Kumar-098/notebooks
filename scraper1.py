import json
import re
import time
import random
import hashlib
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY      = "467212a79c8fa202e600c963e3def90fb964094d"   # get free key at indiankanoon.org/api/
OUTPUT_FILE  = "bail.jsonl"
DELAY_MIN    = 1.5   # seconds between requests (be polite)
DELAY_MAX    = 3.0
CHUNK_SIZE   = 400   # words per chunk
CHUNK_OVERLAP= 50
MIN_WORDS    = 60    # drop chunks smaller than this
# ─────────────────────────────────────────────────────────────────────────────

BASE_API  = "https://api.indiankanoon.org"
BASE_DOC  = "https://indiankanoon.org/doc"

# ── TARGET QUERIES ────────────────────────────────────────────────────────────
# Each entry: (search_query, pages_to_fetch)
QUERIES = [
    # SC landmarks — high cited_by, authoritative
    ("Arnesh Kumar bail arrest guidelines", 1),
    ("Satender Kumar Antil CBI bail framework", 1),
    ("Sanjay Chandra CBI bail triple test", 1),
    ("Gudikanti Narasimhulu bail philosophy Krishna Iyer", 1),
    ("State Rajasthan Balchand bail rule jail exception", 1),
    ("Bikramjit Singh default bail 167 CrPC", 1),
    ("M Ravindran Intelligence Officer default bail NDPS", 1),
    ("Sushila Aggarwal anticipatory bail 438", 1),

    # Section-specific
    ("section 438 CrPC anticipatory bail conditions", 3),
    ("section 437 CrPC non bailable bail discretion", 3),
    ("section 167 CrPC default bail indefeasible right", 2),
    ("section 439 CrPC High Court bail powers", 2),

    # Offence-specific
    ("bail section 302 murder prima facie", 3),
    ("section 37 NDPS bail twin conditions commercial quantity", 3),
    ("PMLA section 45 bail twin conditions", 2),
    ("UAPA bail Article 21 prima facie", 2),
    ("POCSO bail section 29 presumption", 2),
    ("anticipatory bail 498A misuse Arnesh Kumar", 2),

    # Bail principles
    ("cancellation of bail supervening circumstances", 2),
    ("bail conditions onerous excessive Article 21", 2),
    ("parity bail co-accused granted bail", 2),
    ("bail pending appeal higher threshold", 2),
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

session = requests.Session()
session.headers.update({
    "Authorization": f"Token {API_KEY}",
    "User-Agent": "BailSense-Research-Bot/1.0 (academic project)"
})

def polite_sleep():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

def search_tids(query: str, page: int = 0) -> list[str]:
    """Return list of document TIDs from a search query."""
    try:
        r = session.post(
            f"{BASE_API}/search/",
            data={"formInput": query, "pagenum": page},
            timeout=15
        )
        r.raise_for_status()
        docs = r.json().get("docs", [])
        return [str(d["tid"]) for d in docs if "tid" in d]
    except Exception as e:
        print(f"  [search error] {query} p{page}: {e}")
        return []

def fetch_doc_html(tid: str) -> str | None:
    """Fetch full judgment HTML from Indian Kanoon."""
    try:
        r = session.get(f"{BASE_DOC}/{tid}/", timeout=20)
        r.raise_for_status()
        polite_sleep()
        return r.text
    except Exception as e:
        print(f"  [fetch error] tid={tid}: {e}")
        return None

# ── PARSER HELPERS ────────────────────────────────────────────────────────────

def extract_court(soup: BeautifulSoup, full_text: str) -> str:
    # Try common HTML containers first
    candidates = [
        soup.find("div", class_="docsource_main"),
        soup.find("div", class_="docsource"),
        soup.find("span", class_="docsource"),
    ]
    for el in candidates:
        if el:
            value = clean_meta(el.get_text(" ", strip=True))
            if value:
                return value

    # Fallback from text
    patterns = [
        r"\b(Supreme Court of India)\b",
        r"\b([A-Z][A-Za-z .&\-()]+High Court(?:\s*-\s*[A-Za-z .&\-()]+)?)\b",
        r"\b([A-Z][A-Za-z .&\-()]+District Court)\b",
        r"\b([A-Z][A-Za-z .&\-()]+Sessions Court)\b",
    ]
    for pat in patterns:
        m = re.search(pat, full_text, re.I)
        if m:
            return clean_meta(m.group(1))

    return "Unknown"

def extract_date(soup: BeautifulSoup, full_text: str) -> str:
    candidates = [
        soup.find("span", class_="doc_date"),
        soup.find("div", class_="doc_date"),
    ]
    for el in candidates:
        if el:
            raw = clean_meta(el.get_text(" ", strip=True))
            parsed = parse_date(raw)
            if parsed:
                return parsed

    patterns = [
        r"\bon\s+(\d{1,2}\s+\w+,?\s+\d{4})\b",
        r"\bDate of Order\s*:?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b",
        r"\bDate\s*:?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b",
    ]
    for pat in patterns:
        m = re.search(pat, full_text, re.I)
        if m:
            parsed = parse_date(m.group(1).strip())
            if parsed:
                return parsed

    return "Unknown"

def extract_judge(soup: BeautifulSoup, full_text: str) -> str:
    candidates = [
        soup.find("span", class_="doc_author"),
        soup.find("div", class_="doc_author"),
    ]
    for el in candidates:
        if el:
            value = clean_meta(el.get_text(" ", strip=True))
            if value:
                return value[:80]

    patterns = [
        r"\bAuthor:\s*([A-Za-z.\s]+)",
        r"\bBench:\s*([A-Za-z.\s,]+)",
        r"\bHON'?BLE\s+(?:MR\.?|MRS\.?|MS\.?|DR\.?|JUSTICE\s+)?([A-Za-z.\s]+)",
    ]
    for pat in patterns:
        m = re.search(pat, full_text, re.I)
        if m:
            value = clean_meta(m.group(1))
            if value:
                return value[:80]

    return "Unknown"

def clean_meta(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\bJUDGMENT\b.*$", "", text, flags=re.I).strip()
    text = re.sub(r"\bORDER\b.*$", "", text, flags=re.I).strip()
    return text

# ── PARSERS ───────────────────────────────────────────────────────────────────

def parse_html(html: str, tid: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text(" ", strip=True)

    # ── Title / case name ──
    title_el = soup.find("h2", class_="doc_title") or soup.find("title")
    title = title_el.get_text(strip=True) if title_el else ""
    title = re.sub(r"\s+on\s+\d{1,2}\s+\w+,?\s+\d{4}$", "", title).strip()
    title = clean_meta(title) or "Unknown"

    # ── Court / Date / Judge (fixed) ──
    court = extract_court(soup, full_text)
    date_norm = extract_date(soup, full_text)
    judge = extract_judge(soup, full_text)

    # ── Citations (cites / cited_by) ──
    cite_text = full_text
    cites_m    = re.search(r"Cites\s+(\d+)", cite_text, re.I)
    citedby_m  = re.search(r"Cited by\s+(\d+)", cite_text, re.I)
    cites    = int(cites_m.group(1))   if cites_m else 0
    cited_by = int(citedby_m.group(1)) if citedby_m else 0

    # ── Sections mentioned ──
    sections = extract_sections(cite_text)

    # ── Body text ──
    body_el = (
        soup.find("div", class_="judgments") or
        soup.find("div", id="doc_content") or
        soup.find("div", class_="doc_content")
    )
    if not body_el:
        print(f"    [skip] no body element: {tid}")
        return None

    body = clean_text(body_el.get_text(separator="\n"))

    if len(body.split()) < 100:
        print(f"    [skip] body too short: {tid}")
        return None

    yr_m = re.search(r"\b(19|20)\d{2}\b", date_norm) if date_norm != "Unknown" else None

    return {
        "tid":        tid,
        "case_name":  title,
        "date":       date_norm,
        "year":       int(yr_m.group()) if yr_m else None,
        "court":      court,
        "court_type": infer_court_type(court),
        "judge":      judge[:80],
        "cites":      cites,
        "cited_by":   cited_by,
        "authority":  score_authority(cited_by, infer_court_type(court)),
        "sections":   sections,
        "url":        f"{BASE_DOC}/{tid}/",
        "body":       body,
    }

def extract_sections(text: str) -> list[str]:
    """Extract all CrPC/IPC/NDPS/PMLA/BNSS section references."""
    patterns = [
        r"[Ss]ection\s+(\d+[A-Za-z]?(?:\(\d+\))?)\s+(?:Cr\.?P\.?C|CrPC|BNSS|IPC|NDPS|PMLA|UAPA|POCSO)",
        r"[Ss]\.?\s*(\d+[A-Za-z]?(?:\(\d+\))?)\s+(?:Cr\.?P\.?C|CrPC|BNSS)",
    ]
    found = set()
    for pat in patterns:
        for m in re.finditer(pat, text):
            found.add(m.group(1))
    return sorted(found)

def parse_date(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""

    months = {
        "january":"01","february":"02","march":"03","april":"04",
        "may":"05","june":"06","july":"07","august":"08",
        "september":"09","october":"10","november":"11","december":"12"
    }

    # Format: 24 July, 2003
    m = re.search(r"(\d{1,2})\s+(\w+),?\s+(\d{4})", raw, re.I)
    if m:
        day = m.group(1).zfill(2)
        mon = months.get(m.group(2).lower())
        yr  = m.group(3)
        if mon:
            return f"{yr}-{mon}-{day}"

    # Format: 24.09.2013 or 24/09/2013 or 24-09-2013
    m = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", raw)
    if m:
        day = m.group(1).zfill(2)
        mon = m.group(2).zfill(2)
        yr = m.group(3)
        if len(yr) == 2:
            yr = f"20{yr}" if int(yr) <= 30 else f"19{yr}"
        return f"{yr}-{mon}-{day}"

    return raw.strip()

def clean_text(text: str) -> str:
    text = re.sub(r"\[Cites\s+\d+.*?\]", "", text, flags=re.I | re.S)
    text = re.sub(r"Page\s*-?\s*\d+\s*of\s*\d+", "", text, flags=re.I)
    text = re.sub(r"::: Downloaded on.*?:::", "", text, flags=re.I | re.S)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def infer_court_type(court: str) -> str:
    c = (court or "").lower()
    if "supreme" in c:
        return "Supreme Court"
    if "high court" in c:
        return "High Court"
    if "district" in c:
        return "District Court"
    if "sessions" in c:
        return "Sessions Court"
    return "Other"

def score_authority(cited_by: int, court_type: str) -> str:
    bonus = 20 if court_type == "Supreme Court" else 0
    eff   = cited_by + bonus
    if eff >= 500:
        return "high"
    if eff >= 50:
        return "medium"
    return "low"

# ── CHUNKER ───────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Section-aware chunker — respects numbered paragraphs."""
    paras = re.split(r"\n\s*\n", text)
    chunks, buf, buf_words = [], [], 0

    overlap_paras = max(1, CHUNK_OVERLAP // 25)  # rough paragraph overlap

    for para in paras:
        para = para.strip()
        if not para:
            continue

        w = len(para.split())
        if buf_words + w > CHUNK_SIZE and buf:
            candidate = " ".join(buf)
            if len(candidate.split()) >= MIN_WORDS:
                chunks.append(candidate)

            buf = buf[-overlap_paras:] if len(buf) >= overlap_paras else buf
            buf_words = sum(len(b.split()) for b in buf)

        buf.append(para)
        buf_words += w

    if buf:
        candidate = " ".join(buf)
        if len(candidate.split()) >= MIN_WORDS:
            chunks.append(candidate)

    return chunks

# ── WRITER ────────────────────────────────────────────────────────────────────

def write_chunks(doc: dict, out_file):
    chunks = chunk_text(doc["body"])
    base = {k: v for k, v in doc.items() if k != "body"}
    for i, chunk in enumerate(chunks):
        uid = hashlib.md5(f"{doc['tid']}_{i}".encode()).hexdigest()[:12]
        record = {
            "id":           uid,
            "chunk_index":  i,
            "total_chunks": len(chunks),
            "chunk_text":   chunk,
            "word_count":   len(chunk.split()),
            **base,
        }
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    seen_tids  = set()
    total_docs = 0
    total_chunks = 0

    # Load already-scraped TIDs so we can resume safely
    out_path = Path(OUTPUT_FILE)
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as existing:
                for line in existing:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        seen_tids.add(json.loads(line)["tid"])
                    except Exception:
                        pass
            print(f"Resuming — {len(seen_tids)} TIDs already in corpus")
        except Exception as e:
            print(f"[resume warning] could not read existing output: {e}")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_file:
        for query, pages in QUERIES:
            print(f"\nQuery: '{query}'")
            for page in range(pages):
                tids = search_tids(query, page)
                print(f"  page {page} → {len(tids)} TIDs")
                polite_sleep()

                for tid in tids:
                    if tid in seen_tids:
                        print(f"    skip (duplicate): {tid}")
                        continue

                    html = fetch_doc_html(tid)
                    if not html:
                        continue

                    doc = parse_html(html, tid)
                    if not doc:
                        print(f"    skip (parse failed): {tid}")
                        continue

                    # Quality gate — skip low-authority District Court docs
                    if doc["court_type"] == "District Court" and doc["cited_by"] < 10:
                        print(f"    skip (low authority district court): {doc['case_name'][:50]}")
                        continue

                    n_chunks = len(chunk_text(doc["body"]))
                    write_chunks(doc, out_file)
                    out_file.flush()

                    seen_tids.add(tid)
                    total_docs   += 1
                    total_chunks += n_chunks
                    print(f"    saved: {doc['case_name'][:55]:<55} | cited_by={doc['cited_by']:>6} | chunks={n_chunks}")

    print(f"\n{'─'*60}")
    print(f"  Documents saved : {total_docs}")
    print(f"  Total chunks    : {total_chunks}")
    print(f"  Output file     : {OUTPUT_FILE}")
    print(f"{'─'*60}")

if __name__ == "__main__":
    main()