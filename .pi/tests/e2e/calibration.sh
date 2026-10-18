#!/usr/bin/env bash
# S7 calibration target (`make -C .pi calibration`).
#
# Runs the audit calibration in its three offline fixture modes and asserts the
# refusal paths: a calibrated fixture produces a recall, a weakened brief and a
# confirmed rejected lead both exit 10 with an UNCALIBRATED marker. No model
# calls, no network; outputs go to a temp dir that is removed on exit.
#
# The model-backed historical run is NOT part of this target:
#   .pi/calibrate.sh --run-finders --narrowed --out /tmp/cal-real
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/vangja-calibration-XXXXXX")"
STATUS=0

cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

pass() { echo "PASS  $*"; }
fail() { STATUS=1; echo "FAIL  $*"; }

echo "== calibration: strong fixture (must be calibrated) =="
if bash "$ROOT/.pi/calibrate.sh" --fixture "$ROOT/.pi/tests/audit/fixtures/calibration/strong.json" --out "$TMP/strong" >/dev/null 2>&1; then
	recall="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['recall'])" "$TMP/strong/recall.json" 2>/dev/null || echo "?")"
	if [[ "$recall" == "1.0" || "$recall" == "1" ]]; then
		pass "strong fixture calibrated (recall=$recall)"
	else
		fail "strong fixture recall=$recall (expected 1.0)"
	fi
else
	fail "strong fixture run"
fi

echo "== calibration: weakened brief (must refuse with exit 10) =="
bash "$ROOT/.pi/calibrate.sh" --weakened --out "$TMP/weak" >/dev/null 2>&1
weak_status=$?
if [[ "$weak_status" -eq 10 && -f "$TMP/weak/UNCALIBRATED" ]]; then
	pass "weakened brief → exit 10 + UNCALIBRATED"
else
	fail "weakened brief → exit $weak_status (expected 10 + UNCALIBRATED)"
fi

echo "== calibration: confirmed rejected lead (must refuse with exit 10) =="
bash "$ROOT/.pi/calibrate.sh" --fixture "$ROOT/.pi/tests/audit/fixtures/calibration/rejected-lead.json" --out "$TMP/rejected" >/dev/null 2>&1
rejected_status=$?
if [[ "$rejected_status" -eq 10 && -f "$TMP/rejected/UNCALIBRATED" ]]; then
	pass "confirmed rejected lead → exit 10 + UNCALIBRATED"
else
	fail "confirmed rejected lead → exit $rejected_status (expected 10 + UNCALIBRATED)"
fi

echo
if [[ "$STATUS" -eq 0 ]]; then
	echo "calibration: all three modes behave as specified"
else
	echo "calibration: FAILED"
fi
exit "$STATUS"
