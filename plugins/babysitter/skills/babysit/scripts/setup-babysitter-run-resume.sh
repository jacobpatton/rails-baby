#!/bin/bash

# Babysitter Resume Run Setup Script
# Resumes an existing babysitter run by reading the run ID from arguments

set -euo pipefail

# Parse arguments (check for --help early before requiring session ID)
RUN_ID=""
MAX_ITERATIONS=0

# Parse options and positional arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      cat << 'HELP_EOF'
Babysitter Resume - Resume an existing babysitter run

USAGE:
  /babysitter-resume <run-id> [OPTIONS]

ARGUMENTS:
  <run-id>     The run ID to resume (e.g., run-20260119-example)

OPTIONS:
  --max-iterations <n>           Override max iterations (default: keep existing)
  -h, --help                     Show this help message

DESCRIPTION:
  Resumes an existing Babysitter run in your CURRENT session. The run must
  exist in .a5c/runs/<run-id>/ directory.

  The stop hook prevents exit and uses run:iterate to drive orchestration
  until completion or iteration limit.

EXAMPLES:
  /babysitter-resume --claude-session-id "${CLAUDE_SESSION_ID}" --run-id run-20260119-example
  /babysitter-resume --claude-session-id "${CLAUDE_SESSION_ID}" --run-id run-20260119-example --max-iterations 20

STOPPING:
  Only by reaching --max-iterations or completion secret detection
  No manual stop - Babysitter runs infinitely by default!
HELP_EOF
      exit 0
      ;;
    --claude-session-id)
      if [[ -z "${2:-}" ]]; then
        echo "❌ Error: --claude-session-id requires a session ID argument" >&2
        exit 1
      fi
      CLAUDE_SESSION_ID="$2"
      shift 2
      ;;
    --run-id)
      if [[ -z "${2:-}" ]]; then
        echo "❌ Error: --run-id requires a run ID argument" >&2
        exit 1
      fi
      RUN_ID="$2"
      shift 2
      ;;
    --max-iterations)
      if [[ -z "${2:-}" ]]; then
        echo "❌ Error: --max-iterations requires a number argument" >&2
        exit 1
      fi
      if ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "❌ Error: --max-iterations must be a positive integer or 0, got: $2" >&2
        exit 1
      fi
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    *)
      # First non-option argument is the run ID
      if [[ -z "$RUN_ID" ]]; then
        RUN_ID="$1"
        shift
      else
        echo "❌ Error: Unexpected argument: $1" >&2
        echo "   Usage: /babysitter-resume <run-id> [OPTIONS]" >&2
        exit 1
      fi
      ;;
  esac
done

# Validate run ID is provided
if [[ -z "$RUN_ID" ]]; then
  echo "❌ Error: No run ID provided" >&2
  echo "" >&2
  echo "   Usage: /babysitter-resume <run-id>" >&2
  echo "" >&2
  echo "   Examples:" >&2
  echo "     /babysitter-resume run-20260119-example" >&2
  echo "     /babysitter-resume run-20260119-example --max-iterations 20" >&2
  echo "" >&2
  echo "   For all options: /babysitter-resume --help" >&2
  exit 1
fi

# Now check for session ID (after --help has been processed)
if [[ -z "${CLAUDE_SESSION_ID:-}" ]]; then
  echo "❌ Error: CLAUDE_SESSION_ID not available" >&2
  echo "   Babysitter requires session isolation to work correctly." >&2
  exit 1
fi

# Determine state directory (plugin-relative for session isolation)
if [[ -n "${CLAUDE_PLUGIN_ROOT:-}" ]]; then
  STATE_DIR="$CLAUDE_PLUGIN_ROOT/state"
else
  # Fallback: derive from script location
  STATE_DIR="$(dirname "$(dirname "$0")")/state"
fi

# Verify run exists
RUN_DIR=".a5c/runs/$RUN_ID"
if [[ ! -d "$RUN_DIR" ]]; then
  echo "❌ Error: Run not found: $RUN_ID" >&2
  echo "   Expected directory: $RUN_DIR" >&2
  echo "" >&2
  echo "   Available runs:" >&2
  if [[ -d ".a5c/runs" ]]; then
    ls -1 .a5c/runs/ | head -10
  else
    echo "   (no .a5c/runs directory found)" >&2
  fi
  exit 1
fi

# Get run status using CLI
CLI="npx -y @a5c-ai/babysitter-sdk@latest"

RUN_STATUS=$($CLI run:status "$RUN_ID" --json 2>/dev/null || echo '{}')
STATE="unknown"
PROCESS_ID="unknown"
if command -v jq >/dev/null 2>&1; then
  STATE=$(echo "$RUN_STATUS" | jq -r '.state // "unknown"')
  PROCESS_ID=$(echo "$RUN_STATUS" | jq -r '.metadata.processId // "unknown"')
fi

# Check if run is in a resumable state
if [[ "$STATE" == "completed" ]]; then
  echo "❌ Error: Run is already completed" >&2
  echo "   Run ID: $RUN_ID" >&2
  echo "   Cannot resume a completed run." >&2
  exit 1
fi

if [[ "$STATE" == "unknown" ]]; then
  echo "⚠️  Warning: Could not determine run state" >&2
  echo "   Run ID: $RUN_ID" >&2
  echo "   Proceeding anyway..." >&2
fi

# Create prompt from run status
PROMPT="Resume Babysitter run: $RUN_ID

Process: $PROCESS_ID
Current state: $STATE

Continue orchestration using run:iterate loop."

# Create state file for stop hook (markdown with YAML frontmatter)
mkdir -p "$STATE_DIR"
BABYSITTER_STATE_FILE="$STATE_DIR/${CLAUDE_SESSION_ID}.md"

cat > "$BABYSITTER_STATE_FILE" <<EOF
---
active: true
iteration: 1
max_iterations: $MAX_ITERATIONS
started_at: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_id: "$RUN_ID"
---

$PROMPT
EOF

# Output setup message
cat <<EOF
🔄 Babysitter run resumed in this session!

Run ID: $RUN_ID
Process: $PROCESS_ID
Current state: $STATE
Iteration: 1
Max iterations: $(if [[ $MAX_ITERATIONS -gt 0 ]]; then echo $MAX_ITERATIONS; else echo "unlimited"; fi)
Completion promise: secret (emitted only when run completes)

The stop hook is now active. When you try to exit, run:iterate will be called
to drive the next orchestration step. You'll see your previous work in journal,
state.json, and task results.

To monitor: head -10 "$BABYSITTER_STATE_FILE"

⚠️  WARNING: This loop cannot be stopped manually! It will run infinitely
    unless you set --max-iterations or the run completes.

🔄
EOF

# Output the resume prompt
echo ""
echo "$PROMPT"

