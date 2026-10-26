# Laptop quickstart

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

`pip install .` installs the same package when uv is not the tool you use.

## Postgres

Install Postgres and pgvector with the operating system package manager. `NDUP_DATABASE_URL` selects the server. When `pg_isready` finds a local server, `ndup init` creates the database and the `vector` extension.

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
uv run ndup sync
uv run ndup embed
uv run ndup analyze --full
uv run ndup show clusters
```

## RunPod

Optional, and separate from the local setup above. The laptop calls the API and does not build an image. GitHub Actions publishes `ghcr.io/anikal2001/ndup-pod:<tag>` on version tags. RunPod runs that image. Embeddings stay on the pod or the laptop.

```bash
export RUNPOD_API_KEY=...
uv run ndup runpod connect
uv run ndup runpod up
uv run ndup runpod status
uv run ndup runpod down
```

`ndup runpod up --cpu` starts a CPU pod on the same Secure Cloud network volume. `[deploy] target = "pod"` and `[deploy.pod]` are the config keys. `ingress = "runpod-proxy"` prints `https://<pod-id>-<port>.proxy.runpod.net`. `cloudflare-tunnel` opens no inbound port. The API key stays in the OS keychain or `RUNPOD_API_KEY`.

Compose files in the repository are an optional extra. This quickstart does not use them.
