#!/usr/bin/env bash
# Calibration for the Vangja audit harness (plan/07_AUDIT_HARNESS.md §7).
#
# The calibration subset is workstreams S, M, and R against the frozen historical
# snapshot (main@8028c51, extracted read-only under .pi/state/calibration/). The
# scorer — never the finders — holds the expected findings (WS-S-001/002/003,
# WS-M-001/002, WS-R-001/004) and the four rejected Phase-A leads (F-012, F-014,
# F-015, F-016) as negative controls.
#
# A run with recall < 0.6 or a confirmed rejected lead writes UNCALIBRATED; the
# orchestrator then refuses to grade (exit 10).
#
# Usage:
#   .pi/calibrate.sh --fixture <findings.json> [--out <dir>]
#   .pi/calibrate.sh --findings-dir <dir> [--out <dir>]
#   .pi/calibrate.sh --run-finders [--narrowed] [--out <dir>]   # uses the pinned model
#   .pi/calibrate.sh --weakened                       # fixture: deliberately weak brief
#   .pi/calibrate.sh --dry-run                        # print the plan, change nothing
#   .pi/calibrate.sh --snapshot-check                 # verify the read-only worktree
#
# Exit codes: 0 calibrated, 10 UNCALIBRATED, 2 usage, 3 environment/snapshot.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_SHA="8028c5108d926c58ec4e9b29f72842f2b9d8d095"
SNAP_DIR="$ROOT/.pi/state/calibration/snapshot"
EXPECTED="$ROOT/.pi/audit/calibration/expected.json"
MODE=""
FINDINGS=""
OUT=""
DRY_RUN=0
SNAPSHOT_CHECK=0
TS="$(date -u +%Y%m%dT%H%M%SZ)"
CAL_ROOT="$ROOT/.pi/audit/calibration"
NODE_BIN="${NODE_BIN:-node}"
NARROWED=0
FINDER_TIMEOUT="${FINDER_TIMEOUT:-1200}"

usage() {
	sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--fixture)
			MODE="fixture"
			FINDINGS="$2"
			shift 2
			;;
		--findings-dir)
			MODE="collect"
			FINDINGS="$2"
			shift 2
			;;
		--run-finders)
			MODE="finder"
			shift
			;;
		--weakened)
			MODE="fixture"
			FINDINGS="$ROOT/.pi/tests/audit/fixtures/calibration/weak.json"
			shift
			;;
		--out)
			OUT="$2"
			shift 2
			;;
		--narrowed)
			NARROWED=1
			shift
			;;
		--dry-run)
			DRY_RUN=1
			shift
			;;
		--snapshot-check)
			SNAPSHOT_CHECK=1
			shift
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "calibrate: unknown argument $1" >&2
			usage >&2
			exit 2
			;;
	esac
done

OUT="${OUT:-$CAL_ROOT/$TS}"

ensure_snapshot() {
	if [[ -d "$SNAP_DIR" ]] && git -C "$SNAP_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
		local actual
		actual="$(git -C "$SNAP_DIR" rev-parse HEAD)"
		if [[ "$actual" != "$SNAPSHOT_SHA" ]]; then
			echo "calibrate: snapshot worktree is at $actual, expected $SNAPSHOT_SHA" >&2
			exit 3
		fi
		return 0
	fi
	mkdir -p "$(dirname "$SNAP_DIR")"
	echo "calibrate: creating read-only snapshot worktree at $SNAP_DIR ($SNAPSHOT_SHA)"
	git -C "$ROOT" worktree add --detach "$SNAP_DIR" "$SNAPSHOT_SHA" >/dev/null
}

snapshot_clean() {
	local status
	status="$(git -C "$SNAP_DIR" status --porcelain 2>/dev/null || true)"
	if [[ -n "$status" ]]; then
		echo "calibrate: the snapshot worktree was modified during calibration:" >&2
		echo "$status" >&2
		return 1
	fi
	return 0
}

if [[ "$SNAPSHOT_CHECK" -eq 1 ]]; then
	ensure_snapshot
	snapshot_clean
	actual="$(git -C "$SNAP_DIR" rev-parse HEAD)"
	echo "snapshot: $SNAP_DIR @ $actual (read-only check passed)"
	exit 0
fi

if [[ -z "$MODE" ]]; then
	# Default to the real calibration subset. Use --dry-run to see the pinned
	# commands without spending model calls.
	MODE="finder"
fi

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cp "$EXPECTED" "$OUT/expected.json"

run_finders() {
	if [[ "$DRY_RUN" -eq 0 ]]; then
		ensure_snapshot
	fi
	local collector="$OUT/collector"
	mkdir -p "$collector/.pi/state" "$OUT/sessions" "$OUT/briefs"
	cp "$ROOT/AGENTS.md" "$collector/AGENTS.md"
	for config in model.json budgets.json project.json agents.json tools.json; do
		if [[ -f "$ROOT/.pi/config/$config" ]]; then
			mkdir -p "$collector/.pi/config"
			cp "$ROOT/.pi/config/$config" "$collector/.pi/config/$config"
		fi
	done
	awk 'BEGIN{n=0} /^---$/{n++; next} n>=2' "$ROOT/.pi/agents/audit_finder.md" > "$OUT/audit_finder.body.md"

	local before_sha
	before_sha="$(git -C "$SNAP_DIR" rev-parse HEAD 2>/dev/null || echo "$SNAPSHOT_SHA")"
	if [[ "$DRY_RUN" -eq 0 ]]; then
		snapshot_clean
	fi
	for ws in S M R; do
		local brief="$OUT/briefs/WS-$ws.md"
		local focus=""
		if [[ "$NARROWED" -eq 1 ]]; then
			case "$ws" in
				S) focus="**Narrowed focus after a failed calibration:** the estimator actually used by parametric transfer (is it the MAP/mode the paper argues for?), covariance preservation for prior_from_idata, and the regularization scale across pooling modes.";;
				M) focus="**Narrowed focus after a failed calibration:** the configuration-selection procedure (validation split vs test horizon) and the identity of the Prophet/TimeSeers baselines (do the runners import those packages?).";;
				R) focus="**Narrowed focus after a failed calibration:** dependency pinning (is PyMC a declared dependency?) and the seed policy (is any random_seed recorded in fit(), runners, SLURM, or the container?).";;
			esac
		fi
		cat > "$brief" <<BRIEF
## Calibration finder — WS-$ws (read-only historical snapshot)

Snapshot: $SNAP_DIR at $before_sha (main@$SNAPSHOT_SHA). **Never modify the snapshot.**
You are an independent finder for workstream $ws only. Re-derive problems from the
code, configs, runners, artifacts, and paper text in that snapshot. Do not read
Vangja-project/audit/** or .pi/audit/calibration/**; the historical audit is the
scorer's reference, not your source.

Workstream boundary (plan/07 §3): $(
		case "$ws" in
			S) echo "statistics & math — formulas vs code, priors, pooling, transfer moments, potentials, estimator correctness";;
			M) echo "methodology & experimental design — selection, baselines, horizon framing, attribution, generalisation limits";;
			R) echo "reproducibility & provenance — env pinning, seeds, SHAs, manifests, ledger integrity, one-command reproduction";;
		esac
	).

$focus

Rules: nine-field evidence per finding; exact command plus observed output; no
command → not a finding; cap 25 findings; cross-boundary leads are cross-refs.
Effort class: medium — at most 40 tool calls and 15 minutes; register each
finding with auditkit.finding_register(cycle=900, workstream="$ws") **as soon as
it is ready**, not in one batch at the end. Stop after the coverage checklist.
BRIEF
		local session="$OUT/sessions/$ws"
		mkdir -p "$session"
		local args=(
			--mode json -p -na -nc
			--provider opencode-go
			--model opencode-go/deepseek-v4.1-flash
			--thinking max
			-e "$ROOT/.pi/extensions/verify/index.ts"
			-e "$ROOT/.pi/extensions/auditkit/index.ts"
			--skill scientific-audit-and-adversarial-review
			--skill reproducible-experiments
			--skill experiment-protocol-design
			--skill data-provenance
			--append-system-prompt "$OUT/audit_finder.body.md"
			--tools read,bash,grep,find,ls,finding_validate,finding_register,findings_render,verify_tests,verify_artifact,verify_reproduce,verify_traceability,verify_guards
			--session-dir "$session"
			--name "calibration__${ws}__1"
			"TASK: $(cat "$brief")"
		)
		if [[ "$DRY_RUN" -eq 1 ]]; then
			printf 'cd %q && PI_LOOP_ROOT=%q PI_LOOP_ROLE=audit_finder PI_AGENT_NAME=%q pi' \
				"$SNAP_DIR" "$collector" "calibration__${ws}" >&2
			printf ' %q' "${args[@]}" >&2
			printf '\n' >&2
			continue
		fi
		(
			cd "$SNAP_DIR"
			PI_LOOP_ROOT="$collector" \
			PI_LOOP_ROLE=audit_finder \
			PI_AGENT_NAME="calibration__${ws}" \
			PI_LOOP_STATE="AUDIT" \
				pi "${args[@]}" > "$OUT/finder-$ws.json" 2> "$OUT/finder-$ws.err"
		)
	done
	if [[ "$DRY_RUN" -eq 1 ]]; then
		return 0
	fi
	after_sha="$(git -C "$SNAP_DIR" rev-parse HEAD)"
	if [[ "$before_sha" != "$after_sha" ]] || ! snapshot_clean; then
		echo "calibrate: the snapshot was modified; refusing to score" >&2
		exit 3
	fi
	FINDINGS="$collector/.pi/audit/cycle-900"
	if [[ ! -d "$FINDINGS" ]]; then
		echo "calibrate: finders produced no findings under $FINDINGS" >&2
		exit 3
	fi
	MODE="finder"
}

if [[ "$MODE" == "finder" ]]; then
	run_finders
	if [[ "$DRY_RUN" -eq 1 ]]; then
		echo "calibrate: dry run — $( [[ -d "$SNAP_DIR" ]] && echo "snapshot present" || echo "snapshot will be created" )"
		exit 0
	fi
fi

if [[ ! -e "$FINDINGS" ]]; then
	echo "calibrate: findings not found: $FINDINGS" >&2
	exit 2
fi

set +e
"$NODE_BIN" "$ROOT/.pi/extensions/auditkit/calibrate_cli.ts" \
	--findings "$FINDINGS" \
	--out "$OUT" \
	--mode "$MODE" \
	--snapshot "$SNAPSHOT_SHA" \
	--expected "$OUT/expected.json" \
	--note "calibration subset S/M/R; rejected leads F-012/F-014/F-015/F-016 are negative controls"
status=$?
set -e

if [[ "$status" -eq 10 ]]; then
	echo "calibrate: UNCALIBRATED — the orchestrator must refuse to grade this audit run (see $OUT/REPORT.md)" >&2
fi
exit "$status"
