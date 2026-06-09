# ai-infer-radar

A zero-dependency Python script + GitHub Actions workflow that produces a weekly
and monthly digest of recent work in **LLM inference acceleration**, sourced
from arXiv, Hugging Face Daily Papers, GitHub releases, and selected RSS
feeds. The digest is committed to this repository as Markdown and (optionally)
emailed to you via Office365 SMTP.

## What it covers

| Source           | What                                                                  |
|------------------|------------------------------------------------------------------------|
| arXiv            | cs.LG / cs.DC / cs.AR / cs.CL filtered by inference-related keywords  |
| HF Daily Papers  | Community-curated papers via `huggingface.co/api/daily_papers`        |
| GitHub Releases  | vllm, sglang, TensorRT-LLM, flash-attention, transformers, DeepSpeed, lmdeploy, llama.cpp, lightllm |
| RSS              | NVIDIA Developer Blog, vLLM Blog                                      |

Items are deduplicated by `arxiv_id` (so arXiv and HF entries for the same
paper merge), scored against a keyword table biased toward attention/KV cache,
quantization, speculative decoding, serving/scheduling, and kernels/compilers,
and ranked into a Top-N list (10 weekly, 15 monthly).

## Schedule

- **Weekly**: every Monday 08:00 UTC (= JST 17:00 / Beijing 16:00) — rolling last 7 days
- **Monthly**: 1st of each month 09:00 UTC — **previous calendar month** in UTC
  (e.g. triggered on Jun 1 covers May 1 ~ May 31; correctly handles 28/29/30/31-day months and Dec→Jan year boundary)
- **Manual**: `Actions → radar → Run workflow`, choose `weekly` or `monthly`

### Filter semantics

- **Weekly** runs apply a "seen-IDs" filter — items already pushed in a previous
  weekly run are excluded, so each weekly is incremental.
- **Monthly** runs do **not** filter by seen-IDs by default — the monthly digest
  is a self-contained snapshot of that calendar month, even if some items were
  already in earlier weekly digests.
- Overrides via env vars: `INCLUDE_SEEN=1` forces include-all; `EXCLUDE_SEEN=1`
  forces apply-seen-filter (e.g. if you want monthly to also be incremental).

## Setup

### 1. Create a public GitHub repo and push these files

```
.github/workflows/radar.yml
radar/run.py
README.md
```

### 2. Add repository secrets

Go to `Settings → Secrets and variables → Actions → New repository secret`
and add:

| Secret name | Value                                                                    | Required |
|-------------|--------------------------------------------------------------------------|----------|
| `SMTP_USER` | Your Office365 email address, e.g. `you@example.com`                     | optional |
| `SMTP_PASS` | Office365 password **or** App Password if your tenant requires MFA       | optional |
| `MAIL_TO`   | Destination email, e.g. `you@example.com`                                | optional |
| `MAIL_FROM` | From-address; defaults to `SMTP_USER` if omitted                         | optional |

`GITHUB_TOKEN` is provided automatically by Actions — you do **not** need to
create it. It raises the GitHub API rate limit from 60/h to 5000/h for the
release fetcher.

**If SMTP env vars are not set, the script will skip email and only commit
the Markdown report to the repo** — this is a valid first-step deployment.

### 3. Office365 SMTP notes

- Host: `smtp.office365.com`, port `587`, STARTTLS — these are hard-coded in
  the script.
- If your tenant has MFA enabled or "Modern Authentication only", basic SMTP
  AUTH may be disabled. In that case you must either ask your admin to enable
  SMTP AUTH for your mailbox, use an App Password (if your tenant allows), or
  switch to a Microsoft Graph-based sender (not implemented here — would add
  external dependencies).
- For a personal Outlook.com account, the same host/port works with your
  account password or App Password.

### 4. First run

After pushing, trigger a manual run from the Actions tab to verify the
pipeline end-to-end. Check `reports/` for the generated Markdown file.

## Customization

All knobs are at the top of `radar/run.py`:

- `KEYWORDS_SCORED` — keyword → points mapping for relevance scoring
- `AFFILIATION_BOOST` — author/affiliation hints for boosting
- `GITHUB_REPOS` — add or remove repos to watch
- `RSS_FEEDS` — add more blog feeds (any RSS 2.0 or Atom URL)
- `ARXIV_CATEGORIES`, `ARXIV_QUERY_TERMS` — arXiv query shape
- `min_score = 2` in `main()` — raise to reduce noise, lower to widen recall

State is persisted in `radar/state.json` (committed back to repo) so items
already pushed in a previous digest are not re-pushed. To force a "show
everything in the window regardless of seen-state", set `INCLUDE_SEEN=1` as a
workflow env var.

## Known limitations (honest list)

- **HF Daily Papers API is unofficial.** The endpoint
  `huggingface.co/api/daily_papers` is observed in HF community code and
  publicly fetchable, but it has no documented stability contract. If HF
  changes it, that source will start returning 0 items and the script will
  fall through with a warning rather than crashing.
- **RSS feed URLs may move.** The two URLs in `RSS_FEEDS` are best-effort; if
  one changes, you'll see a `[warn]` line in the run log and the report will
  carry a "Source failures" notice.
- **arXiv keyword filter is a recall/precision tradeoff.** The query is
  intentionally broad (any of ~9 terms in title/abstract). Most filtering
  happens in the post-fetch scoring stage. You will still see some
  off-topic results in the lower ranks — that is by design.
- **Scoring is heuristic, not learned.** It's a transparent additive system
  you can audit and tune. Don't treat the Top-N as authoritative; treat it as
  a triage queue.
- **No semantic dedup across non-arXiv sources.** A blog post about
  PagedAttention and the PagedAttention paper itself will appear as two items
  unless their URLs/IDs are identical.

## What's NOT in scope here

- No LLM-based summarization (you asked for English original, zero external
  dependencies). If you later want Chinese summaries, you can pipe the
  `reports/*.md` through any LLM API offline.
- No Twitter/X scraping (no stable public API).
- No conference-proceedings scraping (e.g. MLSys, SOSP). Add manually each
  cycle, or extend with per-conference scrapers later.
