# ndup

Notion semantic duplication detector. It finds pages and database rows that cover the same issue, shows passage-level evidence, and marks the oldest and newest copy. There is no canonical winner and no write-back to Notion.

## Local setup

This runs on the laptop. No RunPod and no Docker.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is missing. `main` is enough.

```bash
git clone https://github.com/anikal2001/ndup.git
cd ndup
uv sync --group dev
export NDUP_NOTION_TOKEN=secret_...
uv run ndup init --database-url memory://.ndup-store --model fake-bow-v1
uv run ndup sync
uv run ndup embed
uv run ndup analyze --full
uv run ndup show clusters
```

`memory://.ndup-store` is a local file store for trying the commands. Postgres 17 + pgvector is the supported store: pass `--database-url postgresql://ndup:ndup@localhost:5432/ndup` when a local Postgres is already running.

`fake-bow-v1` needs no model download. For real embeddings: `uv sync --group dev --extra local`, then `--model BAAI/bge-base-en-v1.5`.

`uv run ndup run` does sync, embed, and analyze once.

Embeddings stay on the machine that runs `ndup`. The default model is `BAAI/bge-base-en-v1.5`. Tests use `fake-bow-v1`. No OpenAI or other remote embedding provider is implemented.

## Postgres

Install Postgres and pgvector with the operating system package manager. Point ndup at it with `NDUP_DATABASE_URL`. When `pg_isready` finds a local server, `ndup init` creates the database and the `vector` extension.

```bash
# Debian or Ubuntu. Use the pgvector package that matches postgres -V
# when postgresql-17-pgvector is not in the distro archive.
sudo apt update
sudo apt install postgresql postgresql-17-pgvector
sudo -u postgres createuser --pwprompt ndup

# macOS
brew install postgresql@17 pgvector
brew services start postgresql@17
```

```bash
export NDUP_DATABASE_URL=postgresql://ndup:ndup@localhost:5432/ndup
uv run ndup init --model fake-bow-v1
uv run ndup run
```

`pip install .` installs the same package when uv is not the tool you use.

## RunPod

The laptop calls the RunPod API. GitHub Actions builds and publishes `ghcr.io/anikal2001/ndup-pod:<tag>` on version tags, and RunPod runs that image. Embeddings stay on the pod or the laptop. Page text is not sent to the RunPod API.

```bash
export RUNPOD_API_KEY=...
uv run ndup runpod connect
uv run ndup runpod up
uv run ndup runpod status
uv run ndup runpod down
```

`ndup runpod connect` stores the key in the OS keychain (or a mode-600 file under `~/.config/ndup/secrets/` when the keychain is unavailable) and checks the account. `ndup runpod up` creates or reuses a Secure Cloud network volume and a GPU pod, starts it, and prints the URL. `ndup runpod up --cpu` uses a CPU pod on the same volume. `status` shows the pod. `down` stops it and keeps the volume.

Config keys match architecture v3:

```toml
[deploy]
target = "pod"
[deploy.pod]
ingress = "cloudflare-tunnel"   # cloudflare-tunnel | runpod-proxy
backup_url = ""
```

`ingress = "runpod-proxy"` prints `https://<pod-id>-<port>.proxy.runpod.net`. `cloudflare-tunnel` opens no inbound port. The API key is `RUNPOD_API_KEY` or the OS keychain. It is never written to the repo, the config file, or logs.

## This build

Solo commands (`init`, `run`, `sync`, `embed`, `analyze`, `show`), native Postgres, and the RunPod connection are in this build. The web app and Cloudflare Access are not.

Design docs:

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
