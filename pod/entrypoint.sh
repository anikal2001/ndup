#!/bin/sh
# Pod boot: one lease, then Postgres on a unix socket, then migrations.
# s6-overlay, cloudflared, and the RunPod volume are Phase 2. The lease is the
# part that can be tested without those packages.
set -eu

LEASE="${NDUP_LEASE:-/workspace/ndup.lease}"
if [ -z "${NDUP_POD_ID:-}" ]; then
  NDUP_POD_ID="${RUNPOD_POD_ID:-local}"
  export NDUP_POD_ID
fi
PGDATA="${PGDATA:-/workspace/pgdata}"

python - <<PY
from datetime import datetime, timezone
from pathlib import Path
import os
from ndup.pod_lease import acquire_lease
acquire_lease(
    Path(os.environ.get("NDUP_LEASE", "/workspace/ndup.lease")),
    os.environ.get("NDUP_POD_ID", "local"),
    datetime.now(timezone.utc),
)
PY

if command -v initdb >/dev/null 2>&1; then
  if [ ! -f "$PGDATA/PG_VERSION" ]; then
    mkdir -p "$PGDATA"
    initdb -D "$PGDATA" --username=ndup --auth=trust >/dev/null
    printf "listen_addresses = ''\nunix_socket_directories = '/var/run/postgresql'\n" >> "$PGDATA/postgresql.conf"
  fi
fi

exec "$@"
