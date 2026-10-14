#!/usr/bin/env bash
# Run a GLM worker on one task.
# Usage: scripts/delegate.sh <implementer|reviewer|bench> <task-id> [extra-notes-file]
# Creates ../jbm-wt/<task-id> on branch task/<task-id> (from main) if missing,
# fills docs/briefs/<role>.md, runs the `glm` CLI there, and writes the final
# report to .agents/reports/<task-id>-<role>.md (stderr to .agents/logs/).
# New branches start from $JBM_BASE (default: main).
set -euo pipefail

role="${1:?role}"; id="${2:?task id}"; extra_file="${3:-}"
repo="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
wt="$(cd "$repo/.." && pwd)/jbm-wt/$id"
brief_tpl="$repo/docs/briefs/$role.md"
spec="$repo/docs/tasks/$id.md"

[[ -f "$brief_tpl" ]] || { echo "no brief template: $brief_tpl" >&2; exit 1; }
[[ -f "$spec" ]] || { echo "no task spec: $spec" >&2; exit 1; }

if [[ ! -d "$wt" ]]; then
  [[ "$role" == "reviewer" ]] && { echo "no worktree to review: $wt" >&2; exit 1; }
  mkdir -p "$(dirname "$wt")"
  if git -C "$repo" show-ref --quiet "refs/heads/task/$id"; then
    git -C "$repo" worktree add -q "$wt" "task/$id"
  else
    git -C "$repo" worktree add -q -b "task/$id" "$wt" "${JBM_BASE:-main}"
  fi
fi
# The spec may be newer than the branch point; keep the worktree copy current.
mkdir -p "$wt/docs/tasks"
cp "$spec" "$wt/docs/tasks/$id.md"

extra=""
[[ -n "$extra_file" ]] && extra="$(cat "$extra_file")"

brief="$(ID="$id" WT="$wt" EXTRA="$extra" python3 - "$brief_tpl" <<'PY'
import os, sys
t = open(sys.argv[1]).read()
t = t.replace("{{ID}}", os.environ["ID"]).replace("{{WORKTREE}}", os.environ["WT"])
print(t.replace("{{EXTRA}}", os.environ["EXTRA"]))
PY
)"

mkdir -p "$repo/.agents/reports" "$repo/.agents/logs"
report="$repo/.agents/reports/$id-$role.md"
log="$repo/.agents/logs/$id-$role-$(date +%Y%m%d-%H%M%S).log"

allowed=(
  "Bash(cargo:*)" "Bash(rustup:*)" "Bash(rustc:*)" "Bash(git:*)"
  "Bash(ls:*)" "Bash(cat:*)" "Bash(head:*)" "Bash(tail:*)" "Bash(wc:*)"
  "Bash(find:*)" "Bash(grep:*)" "Bash(rg:*)" "Bash(mkdir:*)" "Bash(cp:*)"
  "Bash(mv:*)" "Bash(touch:*)" "Bash(uname:*)" "Bash(sysctl:*)" "Bash(lscpu:*)"
  "Bash(diff:*)" "Bash(file:*)" "Bash(python3:*)"
)
flags=(--allowedTools "${allowed[@]}")
[[ "$role" != "implementer" ]] && flags+=(--disallowedTools Edit Write NotebookEdit)

echo "[$role $id] worktree $wt" >&2
cd "$wt"
if glm "$brief" "${flags[@]}" >"$report" 2>"$log"; then
  echo "[$role $id] done -> $report" >&2
else
  code=$?
  echo "[$role $id] glm exited $code (log: $log)" >&2
  exit "$code"
fi
