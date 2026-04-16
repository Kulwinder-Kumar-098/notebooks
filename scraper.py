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
OUTPUT_FILE  = "bail_corpus.jsonl"
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
        r = session.post(f"{BASE_API}/search/", data={
            "formInput": query,
            "pagenum": page
        }, timeout=15)
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

# ── PARSERS ───────────────────────────────────────────────────────────────────

def parse_html(html: str, tid: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")

    # ── Title / case name ──
    title_el = soup.find("h2", class_="doc_title") or soup.find("title")
    title = title_el.get_text(strip=True) if title_el else ""
    title = re.sub(r"\s+on\s+\d{1,2}\s+\w+,?\s+\d{4}$", "", title).strip()

    # ── Court ──
    court_el = soup.find("div", class_="docsource_main")
    court = court_el.get_text(strip=True) if court_el else ""

    # ── Date ──
    date_el = soup.find("span", class_="doc_date") or \
              soup.find("div", class_="doc_date")
    date_raw = date_el.get_text(strip=True) if date_el else ""
    date_norm = parse_date(date_raw)

    # ── Judge ──
    author_el = soup.find("span", class_="doc_author") or \
                soup.find("div", class_="doc_author")
    judge = author_el.get_text(strip=True) if author_el else ""

    # ── Citations (cites / cited_by) ──
    cite_text = soup.get_text()
    cites_m    = re.search(r"Cites\s+(\d+)", cite_text)
    citedby_m  = re.search(r"Cited by\s+(\d+)", cite_text)
    cites    = int(cites_m.group(1))    if cites_m    else 0
    cited_by = int(citedby_m.group(1)) if citedby_m  else 0

    # ── Sections mentioned ──
    sections = extract_sections(cite_text)

    # ── Body text ──
    body_el = soup.find("div", class_="judgments") or \
              soup.find("div", id="doc_content") or \
              soup.find("div", class_="doc_content")
    if not body_el:
        return None
    body = clean_text(body_el.get_text(separator="\n"))

    if len(body.split()) < 100:
        return None   # too short to be useful

    yr_m = re.search(r"\b(19|20)\d{2}\b", date_norm)

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
    months = {"january":"01","february":"02","march":"03","april":"04",
              "may":"05","june":"06","july":"07","august":"08",
              "september":"09","october":"10","november":"11","december":"12"}
    m = re.search(r"(\d{1,2})\s+(\w+),?\s+(\d{4})", raw, re.I)
    if m:
        day = m.group(1).zfill(2)
        mon = months.get(m.group(2).lower())
        yr  = m.group(3)
        if mon:
            return f"{yr}-{mon}-{day}"
    return raw.strip()

def clean_text(text: str) -> str:
    text = re.sub(r"\[Cites\s+\d+.*?\]", "", text)
    text = re.sub(r"Page\s*-?\s*\d+\s*of\s*\d+", "", text, flags=re.I)
    text = re.sub(r"::: Downloaded on.*?:::", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def infer_court_type(court: str) -> str:
    c = court.lower()
    if "supreme" in c:   return "Supreme Court"
    if "high court" in c: return "High Court"
    if "district" in c:  return "District Court"
    if "sessions" in c:  return "Sessions Court"
    return "Other"

def score_authority(cited_by: int, court_type: str) -> str:
    bonus = 20 if court_type == "Supreme Court" else 0
    eff   = cited_by + bonus
    if eff >= 500: return "high"
    if eff >= 50:  return "medium"
    return "low"

# ── CHUNKER ───────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Section-aware chunker — respects numbered paragraphs."""
    paras = re.split(r"\n\s*\n", text)
    chunks, buf, buf_words = [], [], 0
    for para in paras:
        para = para.strip()
        if not para:
            continue
        w = len(para.split())
        if buf_words + w > CHUNK_SIZE and buf:
            candidate = " ".join(buf)
            if len(candidate.split()) >= MIN_WORDS:
                chunks.append(candidate)
            buf = buf[-2:] if len(buf) >= 2 else buf  # overlap
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
        for line in out_path.read_text().splitlines():
            try:
                seen_tids.add(json.loads(line)["tid"])
            except:
                pass
        print(f"Resuming — {len(seen_tids)} TIDs already in corpus")

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