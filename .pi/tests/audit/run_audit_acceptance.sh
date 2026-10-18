#!/usr/bin/env bash
# S6 audit-harness acceptance transcript (offline; no model calls, no network).
#
# Covers the prompt's acceptance tests:
#   1 rubric_validate on all five rubrics                     (node suite)
#   2 run_gates on the seeded fixture + tampered artifact     (probe: gates/tamper)
#   3 seeded audit detection ≥80% and grade ≤ weak reject     (probe: grade)
#   4 finding schema rejections + dedupe                      (probe: findings)
#   5 audit_grade blocking rules + boundaries                 (node suite)
#   6 calibration recall + UNCALIBRATED refusal               (calibrate.sh)
#   7 verify-env includes the audit tests                     (checked by `make -C .pi verify-env`)
#
# Every fixture is copied to a temp dir; the committed fixture is never modified.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NODE="${NODE:-node}"
FIXTURE="$ROOT/.pi/tests/audit/fixtures/seeded"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/vangja-audit-acceptance-XXXXXX")"
STATUS=0
PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "PASS  $*"; }
fail() { FAIL=$((FAIL + 1)); STATUS=1; echo "FAIL  $*"; }
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

run_node_tests() {
	echo "== 1/5. node audit suite (schema, grade, rubric, verify, canary, calibration, templates, wiring) =="
	if (cd "$ROOT" && "$NODE" --test .pi/tests/audit/test_*.test.ts); then
		pass "node audit suite"
	else
		fail "node audit suite"
	fi
}

run_gates_probe() {
	echo
	echo "== 2. gates on the seeded fixture + tampered artifact =="
	local copy="$TMP/gates-fixture"
	cp -R "$FIXTURE" "$copy"
	if (cd "$ROOT" && "$NODE" --experimental-strip-types .pi/tests/audit/acceptance_probe.ts gates "$copy"); then
		pass "run_gates on the seeded fixture"
	else
		fail "run_gates on the seeded fixture"
	fi

	local tampered="$TMP/tampered-fixture"
	cp -R "$FIXTURE" "$tampered"
	local output
	output="$(cd "$ROOT" && "$NODE" --experimental-strip-types .pi/tests/audit/acceptance_probe.ts tamper "$tampered" 2>&1)"
	echo "$output"
	if echo "$output" | grep -q "PASS  tampered artifact fails hash_ok" && echo "$output" | grep -q "PASS  reproduction prints numeric drift"; then
		pass "tampered artifact fails hash_ok and verify_reproduce prints drift"
	else
		fail "tampered artifact detection"
	fi
}

run_grade_probe() {
	echo
	echo "== 3. seeded audit: detection ≥80%, grade ≤ weak reject, GRADE cites gates =="
	local copy="$TMP/grade-fixture"
	cp -R "$FIXTURE" "$copy"
	if (cd "$ROOT" && "$NODE" --experimental-strip-types .pi/tests/audit/acceptance_probe.ts grade "$copy"); then
		pass "seeded audit grade"
	else
		fail "seeded audit grade"
	fi
}

run_findings_probe() {
	echo
	echo "== 4. finding schema rejections and dedupe =="
	if (cd "$ROOT" && "$NODE" --experimental-strip-types .pi/tests/audit/acceptance_probe.ts findings "$TMP"); then
		pass "finding validation"
	else
		fail "finding validation"
	fi
}

run_calibration() {
	echo
	echo "== 6. calibration: strong fixture calibrated, weakened + rejected-lead UNCALIBRATED =="
	local strong="$TMP/cal-strong"
	if bash "$ROOT/.pi/calibrate.sh" --fixture "$ROOT/.pi/tests/audit/fixtures/calibration/strong.json" --out "$strong" >/dev/null 2>&1; then
		local recall
		recall="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['recall'])" "$strong/recall.json" 2>/dev/null || echo "?")"
		pass "strong calibration (recall=$recall, calibrated)"
	else
		fail "strong calibration fixture"
	fi

	local weak="$TMP/cal-weak"
	bash "$ROOT/.pi/calibrate.sh" --weakened --out "$weak" >/dev/null 2>&1
	local weak_status=$?
	if [[ "$weak_status" -eq 10 && -f "$weak/UNCALIBRATED" ]]; then
		pass "weakened brief → exit 10 + UNCALIBRATED marker"
	else
		fail "weakened brief refusal (exit $weak_status)"
	fi

	local rejected="$TMP/cal-rejected"
	bash "$ROOT/.pi/calibrate.sh" --fixture "$ROOT/.pi/tests/audit/fixtures/calibration/rejected-lead.json" --out "$rejected" >/dev/null 2>&1
	local rejected_status=$?
	if [[ "$rejected_status" -eq 10 && -f "$rejected/UNCALIBRATED" ]]; then
		pass "confirmed rejected lead → exit 10 + UNCALIBRATED marker"
	else
		fail "rejected-lead refusal (exit $rejected_status)"
	fi
}

run_node_tests
run_gates_probe
run_grade_probe
run_findings_probe
run_calibration

echo
echo "== summary =="
echo "PASS=$PASS FAIL=$FAIL"
if [[ "$STATUS" -eq 0 ]]; then
	echo "audit acceptance: all checks passed"
else
	echo "audit acceptance: FAILED"
fi
exit "$STATUS"
