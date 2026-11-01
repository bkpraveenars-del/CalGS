#!/bin/bash
# Two parallel lanes (one CPU thread each) for the revision experiments.
cd "$(dirname "$0")"
lane() { for job in "$@"; do eval "$job"; done; }
lane "python3 run_revision.py cv loso" \
     "CALGS_SEED=20261102 CALGS_TAG=_r2 python3 run_revision.py cv loso" \
     "CALGS_SEED=20261104 CALGS_TAG=_r4 python3 run_revision.py cv loso" \
     "python3 run_revision.py loeo" > results/logs/laneA.log 2>&1 &
lane "CALGS_SEED=20261101 CALGS_TAG=_r1 python3 run_revision.py cv loso" \
     "CALGS_SEED=20261103 CALGS_TAG=_r3 python3 run_revision.py cv loso" \
     "CALGS_SEED=20261105 CALGS_TAG=_r5 python3 run_revision.py cv loso" \
     "CALGS_SEED=20261110 CALGS_SCALE=3 CALGS_TAG=_big python3 run_revision.py bigtp" > results/logs/laneB.log 2>&1 &
wait
