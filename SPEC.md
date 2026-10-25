# SPEC: OpenAI embedding provider (minimal)

## Goal

Let `ndup` vectorize chunks with OpenAI `text-embedding-3-small` or `text-embedding-3-large` as an explicit opt-in. Local `BAAI/bge-base-en-v1.5` stays the default. Selecting an OpenAI model sends chunk text to api.openai.com.

## Scope

- `openai` SDK as an optional extra (`uv sync --extra openai`) and a dev dependency.
- `src/ndup/embed/openai_provider.py`: `OpenAIEmbeddingProvider` implementing the `EmbeddingProvider` protocol. Batches of 256 inputs, results ordered by `index`, float32, L2-normalized. Empty inputs are sent as a single space. Key from `/run/secrets/openai_api_key`, `NDUP_OPENAI_API_KEY`, keyring, then `OPENAI_API_KEY`.
- `registry.py`: `build_provider` recognizes the two OpenAI model names; `runtime_for` returns `openai-api` for them so they register as distinct model rows.
- `thresholds.py`: provisional `OPENAI_SHIPPED` prior (tau_chunk 0.80, tau_doc 0.76, source `provisional-openai`).
- `secrets.py`: `openai_api_key` entry so the key is resolved and redacted like the Notion token.
- README: provider is documented; local-only claim is softened.

Out of scope: doctor `openai-key` check, egress proxy allow-list, compose wiring, golden-set threshold fit, broader doc edits.

## Incidental fixes needed to run against the live workspace

- Notion API `2025-09-03` rejects `search` filter `database`; `search_databases` now filters `data_source` and yields each parent database once.
- The Notion client now retries `httpx.TransportError` (connection reset mid-read) the same way it retries timeouts.

## Done check

- `uv run ruff check .`, `uv run mypy`, `uv run pytest -m unit` pass.
- Live run with the Notion key and the OpenAI key from `.env`: `ndup init --model text-embedding-3-small`, `sync --root <page>`, `embed`, `analyze --full`, `show clusters` complete and `ndup models` shows dims 1536, runtime `openai-api`.
