#!/usr/bin/env bash
# Focused carrier-free ODSMF follow-up. All scheduling, training, evaluation
# and summary logic is shared with the completed ODSMF20 driver.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_SUITE=purestate8 exec bash "${SCRIPT_DIR}/run_duke_fdmf_odsmf20.sh" "$@"
