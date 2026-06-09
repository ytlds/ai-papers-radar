#!/usr/bin/env python3
"""
AI Inference Acceleration Radar
================================
Collects recent papers and releases related to LLM inference acceleration
from multiple sources, scores them, and produces a Markdown digest report
that can be emailed via Office365 / Outlook SMTP.

Zero external dependencies: stdlib only (urllib, xml.etree, email, smtplib).
Python 3.10+

Modes (set via CLI arg):
  weekly   - lightweight push, last 7 days
  monthly  - deep digest, last 30 days

Sources covered:
  - arXiv (cs.LG / cs.DC / cs.AR / cs.CL filtered by keywords)
  - Hugging Face Daily Papers (public API)
  - GitHub Releases (vllm, sglang, TensorRT-LLM, flash-attention,
    transformers, DeepSpeed, lmdeploy, llama.cpp, lightllm)
  - RSS feeds (NVIDIA Developer Blog, vLLM Blog)

Outputs:
  - reports/YYYY-MM-DD-{mode}.md  (committed back to repo)
  - email sent via SMTP if env vars provided
  - state.json updated with seen item IDs to prevent duplicates
"""

from __future__ import annotations

import json
import os
import re
import smtplib
import ssl
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / "radar" / "state.json"
REPORTS_DIR = REPO_ROOT / "reports"

USER_AGENT = "ai-infer-radar/1.0 (github-actions)"
REQ_TIMEOUT = 30  # seconds

# arXiv categories most relevant for inference acceleration
ARXIV_CATEGORIES = ["cs.LG", "cs.DC", "cs.AR", "cs.CL"]
ARXIV_MAX_RESULTS = 300  # per category per run

# Keyword scoring table. Hits add to the relevance score.
# Tuned for: attention/KV cache, quantization/sparsity, speculative/parallel
# decoding, serving/scheduling, hardware/kernel/compiler.
KEYWORDS_SCORED: dict[str, int] = {
    # Core inference verbs (cheap baseline)
    "inference": 1,
    "serving": 2,
    "decoding": 2,
    "latency": 1,
    "throughput": 1,
    # Attention / KV cache
    "kv cache": 4,
    "kv-cache": 4,
    "paged attention": 5,
    "pagedattention": 5,
    "flash attention": 5,
    "flashattention": 5,
    "prefix caching": 4,
    "chunked prefill": 4,
    "radix attention": 4,
    "attention sink": 3,
    # Quantization / sparsity
    "quantization": 3,
    "quantized": 2,
    "w4a16": 4,
    "w8a8": 4,
    "fp8": 3,
    "int4": 3,
    "int8": 2,
    "gptq": 4,
    "awq": 4,
    "smoothquant": 4,
    "sparsity": 2,
    "sparse": 1,
    "pruning": 2,
    "mixture of experts": 2,
    "moe": 2,
    # Speculative / parallel decoding
    "speculative decoding": 5,
    "speculative sampling": 5,
    "medusa": 4,
    "eagle": 3,
    "lookahead decoding": 4,
    "parallel decoding": 4,
    "multi-token prediction": 4,
    "tree attention": 3,
    # Serving / scheduling
    "continuous batching": 4,
    "vllm": 3,
    "sglang": 3,
    "tensorrt-llm": 3,
    "disaggregated": 4,
    "prefill-decode": 4,
    "pd separation": 4,
    "mooncake": 3,
    "scheduler": 1,
    # Hardware / kernel / compiler
    "cuda kernel": 3,
    "triton kernel": 3,
    "cutlass": 3,
    "torch.compile": 3,
    "tensor parallelism": 3,
    "pipeline parallelism": 2,
    "communication overlap": 3,
    "mlir": 2,
    # LLM general (lower weight to avoid flooding)
    "large language model": 1,
    "llm": 1,
    "transformer": 1,
}

# Author/affiliation hints that boost a paper. Matched against
# the abstract/authors text. Keep this short and high-precision.
AFFILIATION_BOOST: dict[str, int] = {
    "carnegie mellon": 2,
    "stanford": 2,
    "berkeley": 2,
    "mit": 2,
    "tsinghua": 2,
    "peking university": 2,
    "deepseek": 3,
    "nvidia": 2,
    "meta ai": 2,
    "google deepmind": 2,
    "microsoft research": 2,
    "anthropic": 2,
}

# GitHub repos to watch for releases
GITHUB_REPOS = [
    "vllm-project/vllm",
    "sgl-project/sglang",
    "NVIDIA/TensorRT-LLM",
    "Dao-AILab/flash-attention",
    "huggingface/transformers",
    "microsoft/DeepSpeed",
    "InternLM/lmdeploy",
    "ggml-org/llama.cpp",
    "ModelTC/lightllm",
]

# RSS feeds (public, no auth)
RSS_FEEDS = [
    ("NVIDIA Developer Blog", "https://developer.nvidia.com/blog/feed"),
    ("vLLM Blog", "https://blog.vllm.ai/feed.xml"),
]

# arXiv keyword filter applied at query time (OR-joined)
# Keep this broad; precise scoring happens after fetch.
ARXIV_QUERY_TERMS = [
    "inference", "serving", "decoding", "kv cache", "quantization",
    "speculative", "attention", "throughput", "latency",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Item:
    source: str               # "arxiv" | "hf-daily" | "github" | "rss"
    item_id: str              # globally unique key for dedup
    title: str
    url: str
    published: str            # ISO date string
    authors: str = ""
    abstract: str = ""
    extra: dict = field(default_factory=dict)
    score: int = 0
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTTP helper (stdlib only)
# ---------------------------------------------------------------------------

def http_get(url: str, accept: str = "*/*") -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": accept},
    )
    with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Source: arXiv API
# ---------------------------------------------------------------------------

ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}

def fetch_arxiv(since: datetime, until: datetime) -> list[Item]:
    """Query arXiv API for recent papers in relevant categories matching
    inference-related keywords. Returns items with published in [since, until)."""
    items: list[Item] = []
    cat_q = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    kw_q = " OR ".join(f'all:"{t}"' for t in ARXIV_QUERY_TERMS)
    search_query = f"({cat_q}) AND ({kw_q})"

    params = {
        "search_query": search_query,
        "start": "0",
        "max_results": str(ARXIV_MAX_RESULTS),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    raw = http_get(url, accept="application/atom+xml")
    root = ET.fromstring(raw)

    for entry in root.findall("a:entry", ARXIV_NS):
        arxiv_url = entry.findtext("a:id", default="", namespaces=ARXIV_NS).strip()
        # ID looks like http://arxiv.org/abs/2511.12345v1
        m = re.search(r"abs/([\w.\-]+?)(v\d+)?$", arxiv_url)
        if not m:
            continue
        arxiv_id = m.group(1)
        published = entry.findtext("a:published", default="", namespaces=ARXIV_NS).strip()
        try:
            pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            continue
        if pub_dt < since or pub_dt >= until:
            continue
        title = (entry.findtext("a:title", default="", namespaces=ARXIV_NS) or "").strip()
        title = re.sub(r"\s+", " ", title)
        abstract = (entry.findtext("a:summary", default="", namespaces=ARXIV_NS) or "").strip()
        abstract = re.sub(r"\s+", " ", abstract)
        authors = ", ".join(
            (a.findtext("a:name", default="", namespaces=ARXIV_NS) or "").strip()
            for a in entry.findall("a:author", ARXIV_NS)
        )
        items.append(Item(
            source="arxiv",
            item_id=f"arxiv:{arxiv_id}",
            title=title,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            published=pub_dt.date().isoformat(),
            authors=authors,
            abstract=abstract,
            extra={"arxiv_id": arxiv_id},
        ))
    return items


# ---------------------------------------------------------------------------
# Source: Hugging Face Daily Papers
# ---------------------------------------------------------------------------

def fetch_hf_daily(since: datetime, until: datetime) -> list[Item]:
    """HF Daily Papers public API.

    Endpoint observed in HF community code: https://huggingface.co/api/daily_papers
    Returns a JSON list. Items with published in [since, until).
    We page through until we pass the `since` cutoff (results are reverse-chronological).
    """
    items: list[Item] = []
    page = 1
    while True:
        url = f"https://huggingface.co/api/daily_papers?page={page}&limit=50"
        try:
            raw = http_get(url, accept="application/json")
            data = json.loads(raw)
        except Exception:
            break
        if not data:
            break
        stop = False
        for entry in data:
            pub_str = entry.get("publishedAt") or entry.get("submittedOnDailyAt", "")
            if not pub_str:
                continue
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            if pub_dt < since:
                stop = True
                continue
            if pub_dt >= until:
                # Newer than the window (rare given reverse-chrono ordering,
                # but possible if HF resurfaces an older paper); skip.
                continue
            paper = entry.get("paper") or {}
            arxiv_id = paper.get("id", "")
            if not arxiv_id:
                continue
            title = (entry.get("title") or paper.get("title") or "").strip()
            authors_list = paper.get("authors") or []
            authors = ", ".join(
                a.get("name", "") for a in authors_list if isinstance(a, dict)
            )
            upvotes = paper.get("upvotes", 0)
            num_comments = entry.get("numComments", 0)
            abstract = (paper.get("summary") or "").strip()
            items.append(Item(
                source="hf-daily",
                item_id=f"arxiv:{arxiv_id}",  # same key space as arxiv for dedup
                title=title,
                url=f"https://huggingface.co/papers/{arxiv_id}",
                published=pub_dt.date().isoformat(),
                authors=authors,
                abstract=abstract,
                extra={"upvotes": upvotes, "num_comments": num_comments,
                       "arxiv_id": arxiv_id},
            ))
        if stop or page >= 20:  # hard cap to avoid runaway
            break
        page += 1
    return items


# ---------------------------------------------------------------------------
# Source: GitHub Releases
# ---------------------------------------------------------------------------

def fetch_github_releases(since: datetime, until: datetime) -> list[Item]:
    """Use GitHub REST API for releases. Returns releases with published_at in
    [since, until). Unauthenticated rate limit is 60/h per IP; GitHub Actions
    runners get an authenticated token via GITHUB_TOKEN env var which raises
    this to 5000/h."""
    items: list[Item] = []
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    for repo in GITHUB_REPOS:
        url = f"https://api.github.com/repos/{repo}/releases?per_page=10"
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/vnd.github+json",
                    **({"Authorization": f"Bearer {token}"} if token else {}),
                },
            )
            with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
                releases = json.loads(resp.read())
        except Exception as e:
            print(f"[warn] github releases fetch failed for {repo}: {e}",
                  file=sys.stderr)
            continue
        for rel in releases:
            pub_str = rel.get("published_at", "")
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except ValueError:
                continue
            if pub_dt < since or pub_dt >= until:
                continue
            tag = rel.get("tag_name", "")
            name = rel.get("name", "") or tag
            body = (rel.get("body") or "").strip()
            items.append(Item(
                source="github",
                item_id=f"github:{repo}@{tag}",
                title=f"{repo} {name}",
                url=rel.get("html_url", f"https://github.com/{repo}/releases/tag/{tag}"),
                published=pub_dt.date().isoformat(),
                authors=repo,
                abstract=body[:800],
                extra={"repo": repo, "tag": tag, "prerelease": rel.get("prerelease", False)},
            ))
    return items


# ---------------------------------------------------------------------------
# Source: RSS feeds (NVIDIA Developer Blog, vLLM Blog)
# ---------------------------------------------------------------------------

def _parse_rss_date(s: str) -> datetime | None:
    """Try a few common formats found in RSS/Atom feeds."""
    s = s.strip()
    fmts = [
        "%a, %d %b %Y %H:%M:%S %z",     # RFC822
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def fetch_rss(since: datetime, until: datetime) -> list[Item]:
    items: list[Item] = []
    for feed_name, feed_url in RSS_FEEDS:
        try:
            raw = http_get(feed_url, accept="application/rss+xml, application/atom+xml, application/xml")
            root = ET.fromstring(raw)
        except Exception as e:
            print(f"[warn] RSS fetch failed for {feed_name}: {e}", file=sys.stderr)
            continue

        # Handle both RSS 2.0 (<rss><channel><item>) and Atom (<feed><entry>)
        ns_atom = {"a": "http://www.w3.org/2005/Atom"}
        entries = root.findall(".//item")  # RSS 2.0
        is_atom = False
        if not entries:
            entries = root.findall("a:entry", ns_atom)
            is_atom = True

        for e in entries:
            if is_atom:
                title = (e.findtext("a:title", default="", namespaces=ns_atom) or "").strip()
                link_el = e.find("a:link", ns_atom)
                link = link_el.get("href") if link_el is not None else ""
                pub_str = (e.findtext("a:published", default="", namespaces=ns_atom)
                           or e.findtext("a:updated", default="", namespaces=ns_atom))
                summary = (e.findtext("a:summary", default="", namespaces=ns_atom) or "").strip()
            else:
                title = (e.findtext("title") or "").strip()
                link = (e.findtext("link") or "").strip()
                pub_str = (e.findtext("pubDate") or e.findtext("{http://purl.org/dc/elements/1.1/}date") or "")
                summary = (e.findtext("description") or "").strip()

            pub_dt = _parse_rss_date(pub_str)
            if pub_dt is None or pub_dt < since or pub_dt >= until:
                continue
            # Strip HTML tags from summary
            summary = re.sub(r"<[^>]+>", " ", summary)
            summary = re.sub(r"\s+", " ", summary).strip()
            items.append(Item(
                source="rss",
                item_id=f"rss:{link}",
                title=title,
                url=link,
                published=pub_dt.date().isoformat(),
                authors=feed_name,
                abstract=summary[:800],
                extra={"feed": feed_name},
            ))
    return items


# ---------------------------------------------------------------------------
# Scoring & dedup
# ---------------------------------------------------------------------------

def score_item(item: Item) -> None:
    text = f"{item.title}\n{item.abstract}".lower()
    score = 0
    reasons: list[str] = []

    for kw, pts in KEYWORDS_SCORED.items():
        if kw in text:
            score += pts
            reasons.append(f"kw:{kw}(+{pts})")

    aff_text = f"{item.authors}\n{item.abstract}".lower()
    for aff, pts in AFFILIATION_BOOST.items():
        if aff in aff_text:
            score += pts
            reasons.append(f"aff:{aff}(+{pts})")

    if item.source == "hf-daily":
        upvotes = int(item.extra.get("upvotes", 0) or 0)
        bonus = min(upvotes // 10, 5)  # cap at +5
        if bonus:
            score += bonus
            reasons.append(f"hf-upvotes:{upvotes}(+{bonus})")

    if item.source == "github":
        # Releases are signals, not papers — give a small flat boost so they
        # show up but don't outrank scored papers. Skip prereleases noise.
        if not item.extra.get("prerelease"):
            score += 3
            reasons.append("github-release(+3)")
        else:
            score += 1
            reasons.append("github-prerelease(+1)")

    if item.source == "rss":
        # Blog posts: small boost; ranking will mostly depend on keyword hits.
        score += 2
        reasons.append("blog-post(+2)")

    item.score = score
    item.reasons = reasons


def dedup(items: Iterable[Item]) -> list[Item]:
    """Merge duplicates by item_id. When merging arxiv + hf-daily entries for
    the same paper, prefer the hf-daily entry (has upvotes) but combine
    abstracts/authors if either is empty."""
    by_id: dict[str, Item] = {}
    for it in items:
        existing = by_id.get(it.item_id)
        if existing is None:
            by_id[it.item_id] = it
            continue
        # Merge: prefer hf-daily (has community signal)
        if it.source == "hf-daily" and existing.source != "hf-daily":
            merged = it
            if not merged.abstract and existing.abstract:
                merged.abstract = existing.abstract
            if not merged.authors and existing.authors:
                merged.authors = existing.authors
            by_id[it.item_id] = merged
        else:
            # Keep existing; just enrich missing fields
            if not existing.abstract and it.abstract:
                existing.abstract = it.abstract
            if not existing.authors and it.authors:
                existing.authors = it.authors
    return list(by_id.values())


# ---------------------------------------------------------------------------
# State (seen items)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"seen_ids": [], "last_run": None}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Keep state file bounded: only retain last 5000 IDs
    state["seen_ids"] = list(dict.fromkeys(state["seen_ids"]))[-5000:]
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_markdown(mode: str, items: list[Item], failures: list[str],
                    window_label: str) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    top_n = 15 if mode == "monthly" else 10

    lines: list[str] = []
    if mode == "monthly":
        lines.append(f"# AI Inference Acceleration Radar — Monthly Digest ({window_label})")
    else:
        lines.append(f"# AI Inference Acceleration Radar — Weekly Digest")
    lines.append(f"_Generated: {today} UTC  ·  Window: {window_label}_")
    lines.append("")
    lines.append(f"Mode: **{mode}**  ·  New items this run: **{len(items)}**  ·  Top shown: **{min(top_n, len(items))}**")
    lines.append("")

    if failures:
        lines.append("> ⚠️ Source failures this run:")
        for f in failures:
            lines.append(f"> - {f}")
        lines.append("")

    # Group by source for the body, after a top-N summary
    items_sorted = sorted(items, key=lambda x: (-x.score, x.published), reverse=False)
    items_sorted = sorted(items, key=lambda x: (-x.score, x.published))
    # The above is intentional: primary key = score desc, secondary = published asc

    lines.append(f"## Top {min(top_n, len(items))} by Score")
    lines.append("")
    for i, it in enumerate(items_sorted[:top_n], 1):
        lines.append(f"### {i}. [{it.title}]({it.url})")
        lines.append(f"- **Source**: `{it.source}`  ·  **Published**: {it.published}  ·  **Score**: {it.score}")
        if it.authors:
            authors_short = it.authors if len(it.authors) < 200 else it.authors[:200] + "…"
            lines.append(f"- **Authors / Repo**: {authors_short}")
        if it.abstract:
            abs_short = it.abstract if len(it.abstract) < 600 else it.abstract[:600] + "…"
            lines.append(f"- **Abstract**: {abs_short}")
        if it.reasons:
            lines.append(f"- **Why scored**: {', '.join(it.reasons[:8])}")
        lines.append("")

    # Per-source breakdown
    lines.append("## By Source")
    lines.append("")
    by_source: dict[str, list[Item]] = {}
    for it in items_sorted:
        by_source.setdefault(it.source, []).append(it)
    for src in ["arxiv", "hf-daily", "github", "rss"]:
        bucket = by_source.get(src, [])
        if not bucket:
            continue
        lines.append(f"### {src} ({len(bucket)})")
        for it in bucket[:20]:
            lines.append(f"- [{it.title}]({it.url}) — {it.published} — score {it.score}")
        if len(bucket) > 20:
            lines.append(f"- _…and {len(bucket) - 20} more_")
        lines.append("")

    lines.append("---")
    lines.append("_This digest is generated automatically by [ai-infer-radar](https://github.com/) — pick one item to discuss in depth._")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Email via Office365 SMTP
# ---------------------------------------------------------------------------

def send_email_o365(subject: str, body_md: str, body_html: str | None = None) -> None:
    """Send email via Office365 SMTP. Requires env vars:
      SMTP_USER, SMTP_PASS, MAIL_TO
    Optional: MAIL_FROM (defaults to SMTP_USER)
    """
    user = os.environ.get("SMTP_USER", "").strip()
    pwd = os.environ.get("SMTP_PASS", "").strip()
    to_addr = os.environ.get("MAIL_TO", "").strip()
    from_addr = os.environ.get("MAIL_FROM", user).strip() or user
    if not (user and pwd and to_addr):
        print("[info] SMTP env vars not set, skipping email send.", file=sys.stderr)
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body_md)  # plain text fallback = markdown source

    host = "smtp.office365.com"
    port = 587
    ctx = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=60) as s:
        s.ehlo()
        s.starttls(context=ctx)
        s.ehlo()
        s.login(user, pwd)
        s.send_message(msg)
    print(f"[ok] email sent to {to_addr}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Time window
# ---------------------------------------------------------------------------

def compute_window(mode: str, now: datetime | None = None) -> tuple[datetime, datetime, str]:
    """Return (since, until, label) for the given mode.

    weekly:  rolling [now - 7d, now), label = "Last 7 Days"
    monthly: previous calendar month in UTC, [first_of_prev_month,
             first_of_this_month), label = "YYYY-MM" of the previous month.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if mode == "weekly":
        since = now - timedelta(days=7)
        return since, now, "Last 7 Days"
    if mode == "monthly":
        # First day of current month at 00:00 UTC
        first_this = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        # First day of previous month
        if now.month == 1:
            first_prev = datetime(now.year - 1, 12, 1, tzinfo=timezone.utc)
        else:
            first_prev = datetime(now.year, now.month - 1, 1, tzinfo=timezone.utc)
        label = first_prev.strftime("%Y-%m")
        return first_prev, first_this, label
    raise ValueError(f"unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "weekly"
    if mode not in ("weekly", "monthly"):
        print(f"[error] mode must be 'weekly' or 'monthly', got {mode!r}",
              file=sys.stderr)
        return 2

    since, until, window_label = compute_window(mode)
    print(f"[info] mode={mode}, window={since.isoformat()} ~ {until.isoformat()} ({window_label})",
          file=sys.stderr)

    state = load_state()
    seen_ids: set[str] = set(state.get("seen_ids", []))

    failures: list[str] = []
    all_items: list[Item] = []

    fetchers = [
        ("arxiv", fetch_arxiv),
        ("hf-daily", fetch_hf_daily),
        ("github", fetch_github_releases),
        ("rss", fetch_rss),
    ]
    for name, fn in fetchers:
        try:
            batch = fn(since, until)
            print(f"[info] {name}: {len(batch)} items", file=sys.stderr)
            all_items.extend(batch)
        except Exception as e:
            msg = f"{name} fetch failed: {type(e).__name__}: {e}"
            print(f"[warn] {msg}", file=sys.stderr)
            failures.append(msg)

    # Dedup before scoring
    merged = dedup(all_items)

    # Filter policy:
    # - weekly: incremental — exclude items already pushed in past runs (avoid
    #   showing the same paper week after week)
    # - monthly: by default include everything in the calendar month (the
    #   monthly digest is a self-contained snapshot of that month)
    # Overrides:
    #   INCLUDE_SEEN=1 -> always include all items (no seen filter)
    #   EXCLUDE_SEEN=1 -> always apply seen filter (e.g. for monthly too)
    include_seen_env = os.environ.get("INCLUDE_SEEN", "").strip() == "1"
    exclude_seen_env = os.environ.get("EXCLUDE_SEEN", "").strip() == "1"
    if include_seen_env:
        apply_seen_filter = False
    elif exclude_seen_env:
        apply_seen_filter = True
    else:
        apply_seen_filter = (mode == "weekly")

    if apply_seen_filter:
        fresh = [it for it in merged if it.item_id not in seen_ids]
    else:
        fresh = merged
    print(f"[info] merged={len(merged)}, fresh={len(fresh)} (seen_filter={apply_seen_filter})",
          file=sys.stderr)

    # Score
    for it in fresh:
        score_item(it)

    # Drop very low-score items to reduce noise
    min_score = 2
    fresh = [it for it in fresh if it.score >= min_score]
    print(f"[info] after score>={min_score} filter: {len(fresh)}",
          file=sys.stderr)

    # Render
    md = render_markdown(mode, fresh, failures, window_label)

    # Write report
    # Filename includes the window label so monthly reports for May (run on
    # Jun 1) become "2026-06-01-monthly-2026-05.md" — easy to identify.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date().isoformat()
    if mode == "monthly":
        report_path = REPORTS_DIR / f"{today}-monthly-{window_label}.md"
    else:
        report_path = REPORTS_DIR / f"{today}-weekly.md"
    report_path.write_text(md)
    print(f"[ok] wrote report: {report_path}", file=sys.stderr)

    # Update state with newly-seen IDs
    for it in fresh:
        seen_ids.add(it.item_id)
    state["seen_ids"] = sorted(seen_ids)
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    # Email
    if mode == "monthly":
        subject = f"[ai-infer-radar] Monthly Digest {window_label} ({len(fresh)} items)"
    else:
        subject = f"[ai-infer-radar] Weekly Digest — {today} ({len(fresh)} items)"
    try:
        send_email_o365(subject, md)
    except Exception as e:
        print(f"[warn] email send failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        # don't fail the whole job on email error
    return 0


if __name__ == "__main__":
    sys.exit(main())
