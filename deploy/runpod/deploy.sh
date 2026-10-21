#!/bin/sh
# The operator command is `ndup runpod`. This script does not build an image
# and does not call the RunPod API. GitHub Actions publishes the pod image
# on version tags; RunPod pulls that image.
set -eu

echo "Use ndup runpod connect, up, status, and down. The laptop does not build images." >&2
exit 2
