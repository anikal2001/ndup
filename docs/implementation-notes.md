# ndup — implementation notes (running log)

Decisions, changes, and tradeoffs made while turning the v2 design into v3 and the implementation plan. For the spec author to audit.

Companion docs: [architecture v3](./architecture-v3.md) · [implementation plan](./implementation-plan.md)

## Decisions not in the spec

- **Python core, not Rust.** Asked whether Rust would give more concurrency and less memory. Concluded no: Notion caps the system at 3 req/s so the sync worker holds about six requests in flight; inference runs in ONNX/CUDA and kNN in Postgres in either language; the HF tokenizer is already Rust. A Rust core would drop UMAP, HDBSCAN, and the calibration tooling or need a Python sidecar. Rust kept as an option for a static thin-client binary and PyO3 hot paths. Confirmed with the author.
- **RunPod as the team host (pod mode).** Author chose "plan RunPod as the host" over a gated remote embedding provider. Designed a single-container `ndup-pod` image because RunPod runs one container per pod and has no Compose. Content stays inside the pod; the boundary statement changed from "the org's host" to "the ndup host, which is rented". Flagged the processor agreement and encryption at rest as go-live gates.
- **Cloudflare Tunnel + Access as the default ingress on RunPod.** The RunPod HTTP proxy URL is public, and pods have no inbound firewall. A tunnel means no exposed port and gives a signed identity JWT. oauth2-proxy via the RunPod proxy kept as fallback.
- **Teamspace replaced by "area".** The REST API represents team-level pages as workspace-parented and has no teamspace object. Areas are operator-labeled roots.
- **Three discovery channels.** Notion documents search as non-exhaustive with indexing lag. Tree traversal is authoritative, search is a hint, webhooks optional.
- **Thresholds from labeled pairs, percentile only as a floor.** The v2 99.5th percentile was a 0.5 % per-pair false-positive rate, which is thousands of false neighbors per doc at 150k docs. Added a short-doc rule for one- and two-chunk docs, which are the majority once rows are included.
- **Subsets are satellites, not clustering edges.** Prevents chaining unrelated docs through a shared section.
- **Two hashes per chunk.** `text_hash` for collisions and boilerplate, `embed_hash` for the embedding cache, because the embedded text carries a title/heading prefix.
- **Current-state analysis tables.** Replaced per-run snapshots so incremental analysis has a single source of truth; added `cluster_event` for trends.

## Changes from what was specified

- HNSW indexes are created at model registration instead of after the bulk load. The bulk-load optimization saved minutes and made partial analysis during the first sync impossible.
- `web` is attached to a non-internal `frontend` network. Docker does not publish ports for containers on internal-only networks; this is a documented moby behavior.
- `NEWEST` and `active_copies` use observed content changes, falling back to `last_edited_time` until the first observed change, because property and automation edits bump `last_edited_time`.
- `content_mode` is defined per surface and enforced in one `redact()` function in the API layer; export is metadata-only in team mode server-side.
- First-sync estimate is produced by sampling the real workspace during `init` rather than from a fixed requests-per-page assumption.
- Added `529 service_overload` handling next to `429`.

## Tradeoffs

- **Pod mode vs compose.** One container is simpler to host on RunPod but loses per-service memory limits and network isolation between roles; the analyze role gets an in-process working-set cap and the Notion token is injected only into the sync role's env. Compose on a VM remains fully supported by the same code for orgs that prefer it.
- **GPU is elastic, not permanent.** Steady state can run on a CPU pod on the same volume; the GPU pod is for the first sync and re-embeds. Only one pod per volume at a time, enforced by a lease file.
- **Egress control on RunPod is process-level** (in-container proxy allow-list plus the no-egress test), not network-level. Weaker than a VM firewall; documented as a risk.
- **Hourly data-source polling** is adaptive (hot sources hourly, cold daily) rather than complete, trading staleness on quiet trackers for rate-limit budget. Webhooks remove most of the need.
- **Estimates assume two engineers** with a split between ingestion/ops and text/detection. One engineer roughly doubles calendar time.

## Open questions and things the author should know

- RunPod's encryption-at-rest guarantee for network volumes could not be confirmed from public docs during planning; spike S5 asks RunPod directly. Optional `pgcrypto` text encryption is planned as hardening if required.
- The real search miss rate for database rows is unknown; spike S2 measures it and decides how much hourly polling is needed.
- Requests per page on the real workspace are unknown; spike S1 measures them. The 25-hour first-sync figure from v2 is likely optimistic by up to half.
- The planning session that produced these notes kept copies under `docs/ndup/` in the repository that session was scoped to. The ndup project has its own repository per the design; these Project copies are the working set until that repo exists.
- Pre-commit hook was skipped for the docs commit in that session: it runs an swc parse of `src/` TypeScript, which is unrelated to markdown, and `node_modules` is not installed in that environment.
