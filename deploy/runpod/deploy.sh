#!/bin/sh
# Create or swap a RunPod pod. Refuses to call the API without a key so a
# laptop checkout cannot spend money by accident. The live boot is Phase 2.
set -eu

if [ -z "${RUNPOD_API_KEY:-}" ]; then
  echo "RUNPOD_API_KEY is not set. Refusing to call the RunPod API." >&2
  echo "Would POST https://rest.runpod.io/v1/networkvolumes and /v1/pods using deploy/runpod/template.json." >&2
  exit 2
fi

ACTION="${1:-create}"
echo "RunPod action ${ACTION} is not executed from Phase 1 CI." >&2
exit 2
