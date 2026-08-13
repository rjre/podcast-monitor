#!/bin/bash
# Runs transcription in priority order: UK podcasts -> Odd Lots -> Money Stuff -> everything else.
# Each batch is small (15 episodes) so episodes.json/aggregates.json are checkpointed and
# committed/pushed frequently -- this environment periodically idle-restarts and kills
# background processes, so we don't want a single giant run that only saves at the end.
set -uo pipefail
cd /workspace/podcast-monitor

UK_IDS="unhedged,money-to-the-masses,economist-podcasts,patrick-boyle-on-finance,reuters-morning-bid"

run_batches() {
  local label="$1"
  local podcast_ids_flag="$2"
  local max_batches="$3"
  for i in $(seq 1 "$max_batches"); do
    remaining=$(python3 - "$podcast_ids_flag" <<'PYEOF'
import json, sys
ids_arg = sys.argv[1]
eps = json.load(open("data/episodes.json"))
if ids_arg:
    ids = set(ids_arg.split(","))
    pending = [e for e in eps if e.get("podcast_id") in ids and e.get("transcript_status") != "done" and e.get("audio_url")]
else:
    pending = [e for e in eps if e.get("transcript_status") != "done" and e.get("audio_url")]
print(len(pending))
PYEOF
)
    echo "=== $label: batch $i, $remaining pending ===" >> /tmp/priority_transcribe.log
    if [ "$remaining" -eq 0 ]; then
      echo "=== $label: queue empty, moving on ===" >> /tmp/priority_transcribe.log
      break
    fi
    if [ -n "$podcast_ids_flag" ]; then
      python3 pipeline/transcribe.py --limit 15 --podcast-ids "$podcast_ids_flag" >> /tmp/priority_transcribe.log 2>&1
    else
      python3 pipeline/transcribe.py --limit 15 >> /tmp/priority_transcribe.log 2>&1
    fi
    git add data/episodes.json data/aggregates.json data/transcripts/ >> /tmp/priority_transcribe.log 2>&1
    if ! git diff --cached --quiet; then
      git commit -m "Transcribe batch ($label $i)" >> /tmp/priority_transcribe.log 2>&1
      git fetch origin main >> /tmp/priority_transcribe.log 2>&1
      if [ "$(git rev-list HEAD..origin/main --count 2>/dev/null || echo 0)" != "0" ]; then
        git merge origin/main --no-edit >> /tmp/priority_transcribe.log 2>&1
        if [ $? -ne 0 ]; then
          echo "=== $label: batch $i MERGE CONFLICT, needs manual reconciliation, pausing runner ===" >> /tmp/priority_transcribe.log
          exit 1
        fi
        python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, "pipeline")
from run import load_json, save_json, build_aggregates
eps = load_json(os.path.join("data", "episodes.json"), [])
taxonomy_raw = load_json(os.path.join("config", "taxonomy.json"), {})
aggregates = build_aggregates(eps, taxonomy_raw)
save_json(os.path.join("data", "aggregates.json"), aggregates)
PYEOF
        git add data/aggregates.json >> /tmp/priority_transcribe.log 2>&1
        if ! git diff --cached --quiet; then
          git commit -m "Rebuild aggregates after merge ($label $i)" >> /tmp/priority_transcribe.log 2>&1
        fi
      fi
      git push origin main >> /tmp/priority_transcribe.log 2>&1
    fi
  done
}

run_batches "UK" "$UK_IDS" 20
run_batches "OddLots" "odd-lots" 6
run_batches "MoneyStuff" "money-stuff" 2

echo "=== PRIORITY QUEUE COMPLETE, continuing with this machine's half of the general queue ===" >> /tmp/priority_transcribe.log

# A Raspberry Pi is working the other half of the podcast backlog in parallel
# (non-overlapping podcast_id partition, see /tmp/split_result.txt) -- stick to
# our half so neither machine wastes compute re-transcribing the other's work.
SANDBOX_IDS="afford-anything,alternative-allocations,barron-s-live,biggerpockets-money,biggerpockets-real-estate,brew-markets,brown-ambition,count-me-in,diy-money,eurodollar-university,excess-returns,forward-guidance,freakonomics-radio,get-started-investing,hermoney-with-jean-chatzky,journey-to-launch,m-a-science,market-mondays,money-for-couples-with-ramit-sethi,money-girl,morningstar-investing-insights,motley-fool-money,one-rental-at-a-time,prof-g-markets,rational-reminder,real-estate-rookie,rebel-capitalist-news,retire-with-style,retirement-answers,retirement-power-play,retirement-starts-today,simple-passive-cashflow,slate-money,so-money-with-farnoosh-torabi,squawk-pod,stock-market-today-with-ibd,the-clark-howard-podcast,the-crypto-podcast,the-long-term-investor,the-morning-filter,the-retirement-and-ira-show,the-wall-street-skinny,the-wolf-of-all-streets,thoughts-on-the-market,unchained"

while true; do
  remaining=$(python3 - "$SANDBOX_IDS" <<'PYEOF'
import json, sys
ids = set(sys.argv[1].split(","))
eps = json.load(open("data/episodes.json"))
print(sum(1 for e in eps if e.get("podcast_id") in ids and e.get("transcript_status") != "done" and e.get("audio_url")))
PYEOF
)
  if [ "$remaining" -eq 0 ]; then
    echo "=== SANDBOX HALF OF QUEUE TRANSCRIBED ===" >> /tmp/priority_transcribe.log
    break
  fi
  python3 pipeline/transcribe.py --limit 15 --podcast-ids "$SANDBOX_IDS" >> /tmp/priority_transcribe.log 2>&1
  git add data/episodes.json data/aggregates.json data/transcripts/ >> /tmp/priority_transcribe.log 2>&1
  if ! git diff --cached --quiet; then
    git commit -m "Transcribe batch (general queue)" >> /tmp/priority_transcribe.log 2>&1
    git fetch origin main >> /tmp/priority_transcribe.log 2>&1
    if [ "$(git rev-list HEAD..origin/main --count 2>/dev/null || echo 0)" != "0" ]; then
      git merge origin/main --no-edit >> /tmp/priority_transcribe.log 2>&1
      if [ $? -ne 0 ]; then
        echo "=== general queue: MERGE CONFLICT, needs manual reconciliation, pausing runner ===" >> /tmp/priority_transcribe.log
        exit 1
      fi
      python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, "pipeline")
from run import load_json, save_json, build_aggregates
eps = load_json(os.path.join("data", "episodes.json"), [])
taxonomy_raw = load_json(os.path.join("config", "taxonomy.json"), {})
aggregates = build_aggregates(eps, taxonomy_raw)
save_json(os.path.join("data", "aggregates.json"), aggregates)
PYEOF
      git add data/aggregates.json >> /tmp/priority_transcribe.log 2>&1
      if ! git diff --cached --quiet; then
        git commit -m "Rebuild aggregates after merge (general queue)" >> /tmp/priority_transcribe.log 2>&1
      fi
    fi
    git push origin main >> /tmp/priority_transcribe.log 2>&1
  fi
done
