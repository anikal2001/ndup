# ndup

Notion semantic duplication detector. It finds pages and database rows that cover the same issue, shows passage-level evidence, and marks the oldest and newest copy. There is no canonical winner and no write-back to Notion.

Embeddings stay on the machine that runs `ndup`. The default model is `BAAI/bge-base-en-v1.5`. Tests use a deterministic local fake embedder. No OpenAI or other remote embedding provider is implemented.

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
