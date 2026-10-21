# Project context — Notion semantic duplication detector (`ndup`)

Stable constraints. Spec: [architecture v3](./architecture-v3.md). Build plan: [implementation plan](./implementation-plan.md). Decision log: [implementation notes](./implementation-notes.md). Earlier draft: [architecture v2](./architecture.md).

## Product
- Finds Notion docs that cover approximately the same issue, with passage-level evidence, workspace-wide and per doc.
- Each duplicate cluster shows **oldest** and **newest** plus creator, last editor, and dates. No canonical scoring. The user decides what to keep.
- Python 3.12. Viewers use a browser. Solo mode is a CLI for a scoped subset.

## Hard constraints
- **Scope:** all pages and all database rows the integration can see.
- **Scale:** org of ~300–500 people. Design for ≤ 300k documents and ≤ 1M chunks. Sync is resumable and incremental.
- **Privacy:** page content never goes to OpenAI or any remote embedding API. Default model `BAAI/bge-base-en-v1.5`, on the host.
- **Host:** team mode runs on a RunPod Secure Cloud pod (rented hardware, one container, content stays in the pod). Compose on a VM is the second supported target. A processor agreement and an encryption-at-rest decision gate go-live.
- **Language:** English only for detection. Other languages are synced and reported separately, not clustered.
- **Storage:** Postgres 17 + pgvector ≥ 0.8, on the pod volume or the VM. Not exposed off-host.
- **Read-only:** no write-back to Notion.
- **Multi-user:** one shared database. SSO in front (Cloudflare Access by default on RunPod). Notion has no per-user permissions, so the boundary is which roots are connected. Those roots are labeled **areas**.
- **Ingress on RunPod:** Cloudflare Tunnel, no inbound ports. RunPod's public HTTP proxy is the fallback.

## Still open before later phases
- SSO / IdP, Cloudflare account, and whether `/hooks/notion` is exposed (Phase 2).
- RunPod data center and GPU pod class (spikes S4 and S5).
- Processor agreement and whether body text must be encrypted at rest (before the unattended week).
- Which roots form the first audience (before the first full sync).
