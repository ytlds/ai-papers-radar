#!/usr/bin/env python3
"""
Configuration for AI Inference Acceleration Radar
=================================================
All tunable parameters live here. radar.py contains execution logic only and
imports everything from this module. This file is plain Python (no external
parser / zero dependencies).

If this file is missing or fails to import, radar.py aborts (fail-fast) — there
are no built-in defaults in the execution code.

NOTE on weekly vs monthly:
  WEEKLY and MONTHLY are intentionally kept as two fully independent dicts.
  They currently hold the same numbers, but you can tune one without touching
  the other.
"""

# ---------------------------------------------------------------------------
# HTTP / network
# ---------------------------------------------------------------------------

USER_AGENT = "ai-papers-radar/1.0 (github-actions)"
REQ_TIMEOUT = 30  # seconds

# Retry policy for all network calls.
# Strategy: retry on ANY exception (including HTTP 4xx), exponential backoff.
REQ_MAX_RETRIES = 3          # total attempts = REQ_MAX_RETRIES
REQ_BACKOFF_BASE = 1.0       # seconds; delays are 1s, 2s, 4s, ...

# ---------------------------------------------------------------------------
# Per-mode settings (independent; currently identical values)
# ---------------------------------------------------------------------------
# top_quota_per_source : per-source cap in the "Top" section of the report.
# top_quota_default    : quota for any source not explicitly listed.
# by_source_truncate   : max items listed per source in the "By Source" section.

WEEKLY = {
    "top_quota_per_source": {
        "arxiv": 10,
        "hf-daily": 5,
        "github": 5,
        "rss": 5,
    },
    "top_quota_default": 5,
    "by_source_truncate": 50,
}

MONTHLY = {
    "top_quota_per_source": {
        "arxiv": 10,
        "hf-daily": 5,
        "github": 5,
        "rss": 5,
    },
    "top_quota_default": 5,
    "by_source_truncate": 50,
}

# ---------------------------------------------------------------------------
# Scoring: minimum score threshold to keep an item
# ---------------------------------------------------------------------------

MIN_SCORE = 2

# ---------------------------------------------------------------------------
# arXiv source
# ---------------------------------------------------------------------------

# arXiv categories most relevant for inference acceleration
ARXIV_CATEGORIES = ["cs.LG", "cs.DC", "cs.AR", "cs.CL"]
ARXIV_MAX_RESULTS = 300  # per query per run

# arXiv keyword filter applied at query time (OR-joined).
# Keep this broad; precise scoring happens after fetch.
ARXIV_QUERY_TERMS = [
    "inference", "serving", "decoding", "kv cache", "quantization",
    "speculative", "attention", "throughput", "latency",
]

# ---------------------------------------------------------------------------
# Keyword scoring table. Hits add to the relevance score.
# Tuned for: attention/KV cache, quantization/sparsity, speculative/parallel
# decoding, serving/scheduling, hardware/kernel/compiler.
# ---------------------------------------------------------------------------

KEYWORDS_SCORED = {
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
AFFILIATION_BOOST = {
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

# ---------------------------------------------------------------------------
# GitHub releases source
# ---------------------------------------------------------------------------

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

# Per-source scoring tweaks (used in score_item).
GITHUB_RELEASE_BOOST = 3       # flat boost for a normal release
GITHUB_PRERELEASE_BOOST = 1    # smaller boost for prereleases
RSS_BOOST = 2                  # flat boost for blog posts
HF_UPVOTE_DIVISOR = 10         # upvotes // divisor = bonus
HF_UPVOTE_CAP = 5              # max bonus from upvotes

# ---------------------------------------------------------------------------
# RSS feeds (public, no auth)
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    ("NVIDIA Developer Blog", "https://developer.nvidia.com/blog/feed"),
    ("vLLM Blog", "https://blog.vllm.ai/feed.xml"),
]

# ---------------------------------------------------------------------------
# Email (Office365 SMTP)
# ---------------------------------------------------------------------------

SMTP_HOST = "smtp.office365.com"
SMTP_PORT = 587
SMTP_TIMEOUT = 60  # seconds

# ---------------------------------------------------------------------------
# State / report paths (relative to repo root computed in radar.py)
# ---------------------------------------------------------------------------

STATE_SUBPATH = ("radar", "state.json")   # under REPO_ROOT
REPORTS_SUBPATH = ("reports",)            # under REPO_ROOT
STATE_MAX_IDS = 5000                       # bound on seen_ids retained
