# Laptop quickstart

Phase 1 is a solo terminal workflow. Embeddings stay on the machine that runs `ndup`. The default model is `BAAI/bge-base-en-v1.5`. Tests and offline development use `fake-bow-v1`, which does not download weights.

## Install

```bash
uv sync --group dev
uv run pytest -m unit
```

To embed with the real model later, install the local extra and let sentence-transformers read a cache you already populated. The process sets `HF_HUB_OFFLINE=1`.

```bash
uv sync --group dev --extra local
```

## Database

A memory URL is enough to try the commands. Postgres 17 with pgvector 0.8 is the supported store.

```bash
docker compose -f docker-compose.solo.yml up -d
```

Or point `storage.database_url` at a Postgres you already run. `ndup db up` prints the compose command when Docker is missing.

## First run

```bash
export NDUP_NOTION_TOKEN=secret_...
uv run ndup init --database-url postgresql://ndup:ndup@localhost:5432/ndup --model fake-bow-v1
uv run ndup sync
uv run ndup embed
uv run ndup analyze --full
uv run ndup show clusters
```

`ndup init` and `ndup sync --estimate` print an ETA from a sample. Without a token they print the default 3–6 requests/page prior. `ndup run` does sync, embed, and analyze once.

`ndup web` and `ndup worker` exit 2. Those servers are Phase 2.

Switch `--model` to `BAAI/bge-base-en-v1.5` after the weights are on disk. `ndup doctor --bench` times whatever model the config names.
