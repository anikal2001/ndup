# ndup

Notion semantic duplication detector. It finds pages and database rows that cover the same issue, shows passage-level evidence, and marks the oldest and newest copy. There is no canonical winner and no write-back to Notion.

Embeddings are local by default. The default model is `BAAI/bge-base-en-v1.5`. Tests use a deterministic local fake embedder. OpenAI `text-embedding-3-small` and `text-embedding-3-large` are available as an explicit opt-in; selecting one sends chunk text to api.openai.com.

This repository currently contains Phase 0 and Phase 1: a solo terminal workflow (`init`, `sync`, `embed`, `analyze`, `show`). Pod productionization, Cloudflare/SSO, and the web app are Phase 2.

Design docs live in `docs/`:

- [Architecture v3](docs/architecture-v3.md)
- [Implementation plan](docs/implementation-plan.md)
- [Project context](docs/project-context.md)
- [Implementation notes](docs/implementation-notes.md)
- [Laptop quickstart](docs/quickstart.md)

```bash
uv sync --group dev
uv run pytest -m unit
uv run ndup --help
```

## Run locally

This is a CLI application, not a web app. The solo workflow uses a local Postgres
database. Install Python 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and Docker Desktop, then run these commands from the repository directory in PowerShell:

```powershell
uv sync --group dev
docker compose -f docker-compose.solo.yml up -d
uv run ndup init --database-url "postgresql://ndup:ndup@localhost:5432/ndup" --model fake-bow-v1
$env:NDUP_NOTION_TOKEN = (Get-Content .env | ConvertFrom-StringData).NOTION_API_KEY
uv run ndup sync
uv run ndup embed
uv run ndup analyze --full
uv run ndup show clusters
```

Create a Notion integration and share the pages or databases you want to scan
with it. The CLI does not load `.env` automatically; this command reads
`NOTION_API_KEY` from that file and sets the `NDUP_NOTION_TOKEN` environment
variable the CLI expects. Keep `.env` private and never commit its contents. The
`fake-bow-v1` model is useful for trying the workflow without downloading model
weights; for real semantic embeddings, install `uv sync --group dev --extra local`
and configure the real model as described in the
[quickstart](docs/quickstart.md).

### OpenAI embeddings

Install the extra, provide a key, and pick an OpenAI model at `init`. Shipped thresholds for
these models are a provisional prior; run `ndup calibrate` once you have labeled pairs.

```powershell
uv sync --group dev --extra openai
$env:OPENAI_API_KEY = (Get-Content .env | ConvertFrom-StringData).OPENAI_API_KEY
uv run ndup init --database-url "postgresql://ndup:ndup@localhost:5432/ndup" --model text-embedding-3-small
```

`NDUP_OPENAI_API_KEY` or `/run/secrets/openai_api_key` also work. Without Docker, a pickled
memory store works for a single-user run: `--database-url memory://C:/path/to/ndup-state.pkl`.

To stop the local database, run:

```powershell
docker compose -f docker-compose.solo.yml down
```

The web app and worker commands are Phase 2 and are not available in this build.
