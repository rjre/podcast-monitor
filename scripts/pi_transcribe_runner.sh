#!/bin/bash
# Runs on the Raspberry Pi, working the OTHER half of the podcast backlog so
# it never duplicates work the cloud sandbox is already doing. Small batches
# (15 episodes) with a git commit+push checkpoint after each one, since a
# reboot/network drop should lose at most one small batch.
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root
source venv/bin/activate

PI_IDS="best-stocks-now,bloomberg-surveillance,bogleheads-on-investing-podcast,capital-allocators,catching-up-to-fi,choosefi,early-retirement-financial-freedom,earn-invest,equity-mates-investing-podcast,financial-audit,financial-tea-with-mrs-dow-jones,halftime-report,how-to-money,invest-like-the-best,investtalk,macro-voices,mad-money-w-jim-cramer,masters-in-business,money-guy-show,moody-s-talks-inside-economics,morning-call,motley-fool-hidden-gems,optimal-finance-daily,personal-finance-for-long-term-investors,planet-money,ramsey-everyday-millionaires,real-estate-the-ramsey-way,real-estate-this-week,retire-sooner,riskreversal-pod,rule-breaker-investing,sound-investing,stacking-benjamins,talking-wealth,the-acquirers-podcast,the-dividend-cafe,the-intrinsic-value-podcast,the-investing-for-beginners-podcast,the-investor-s-podcast-we-study-billionaires,the-meb-faber-show,the-millionaire-real-estate-agent,the-ramsey-show,the-rundown"

LOG=/tmp/pi_transcribe.log

while true; do
  # Always pull the latest before picking the next batch, so we don't
  # re-transcribe something the sandbox (or the scheduled GitHub Action)
  # already finished.
  git pull origin main --no-edit >> "$LOG" 2>&1

  remaining=$(python3 - "$PI_IDS" <<'PYEOF'
import json, sys
ids = set(sys.argv[1].split(","))
eps = json.load(open("data/episodes.json"))
print(sum(1 for e in eps if e.get("podcast_id") in ids and e.get("transcript_status") != "done" and e.get("audio_url")))
PYEOF
)
  echo "=== $(date -u +%FT%TZ): $remaining pending in Pi's half ===" >> "$LOG"
  if [ "$remaining" -eq 0 ]; then
    echo "=== Pi's half of the queue is fully transcribed ===" >> "$LOG"
    break
  fi

  # Use the "tiny" model if this Pi is a 4/8 (older/less RAM) board, "base"
  # if it's a Pi 5 or has 4GB+ free -- base matches what the cloud side uses,
  # so prefer it if your Pi can keep up. Override by exporting WHISPER_MODEL.
  MODEL="${WHISPER_MODEL:-base}"

  python3 pipeline/transcribe.py --limit 15 --model "$MODEL" --podcast-ids "$PI_IDS" >> "$LOG" 2>&1

  git add data/episodes.json data/aggregates.json data/transcripts/ >> "$LOG" 2>&1
  if ! git diff --cached --quiet; then
    git commit -m "Transcribe batch (Pi, $MODEL model)" >> "$LOG" 2>&1
    git fetch origin main >> "$LOG" 2>&1
    if [ "$(git rev-list HEAD..origin/main --count 2>/dev/null || echo 0)" != "0" ]; then
      git merge origin/main --no-edit >> "$LOG" 2>&1
      if [ $? -ne 0 ]; then
        echo "=== MERGE CONFLICT -- pausing, needs a human/Claude to resolve ===" >> "$LOG"
        exit 1
      fi
      # aggregates.json is a derived file -- never trust git's line-merge of
      # it, always rebuild fresh from the merged episodes.json.
      python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, "pipeline")
from run import load_json, save_json, build_aggregates
eps = load_json(os.path.join("data", "episodes.json"), [])
taxonomy_raw = load_json(os.path.join("config", "taxonomy.json"), {})
aggregates = build_aggregates(eps, taxonomy_raw)
save_json(os.path.join("data", "aggregates.json"), aggregates)
PYEOF
      git add data/aggregates.json >> "$LOG" 2>&1
      if ! git diff --cached --quiet; then
        git commit -m "Rebuild aggregates after merge (Pi)" >> "$LOG" 2>&1
      fi
    fi
    git push origin main >> "$LOG" 2>&1
  fi
done
