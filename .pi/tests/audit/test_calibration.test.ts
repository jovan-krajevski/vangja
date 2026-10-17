/**
 * Calibration: recall over the historical expected list, false positives on the
 * rejected Phase-A leads, and the UNCALIBRATED refusal path (including the
 * deliberately weakened brief).
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { test } from "node:test";
import {
	CALIBRATION_EXPECTED,
	loadCalibrationExpected,
	loadFindingsFile,
	scoreCalibration,
	writeCalibrationReport,
} from "../../extensions/auditkit/calibrate.ts";
import { auditGrade } from "../../extensions/auditkit/grade.ts";
import { CALIBRATION_FIXTURES, CLEAN_RUBRIC, gateResults, readText, REPO_ROOT } from "./helpers.ts";

function findings(name: "strong" | "weak" | "rejected-lead") {
	return loadFindingsFile(path.join(CALIBRATION_FIXTURES, `${name}.json`));
}

function tempOut(prefix: string): string {
	return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

test("the expected list is the plan/07 §7 list with four rejected leads", () => {
	const expected = loadCalibrationExpected(REPO_ROOT);
	assert.equal(expected.snapshot, "8028c5108d926c58ec4e9b29f72842f2b9d8d095");
	assert.deepEqual(expected.workstreams, ["S", "M", "R"]);
	assert.deepEqual(
		expected.cases.map((entry) => entry.id),
		["WS-S-001", "WS-S-002", "WS-S-003", "WS-M-001", "WS-M-002", "WS-R-001", "WS-R-004"],
	);
	assert.deepEqual(
		expected.rejected_leads.map((entry) => entry.id),
		["F-012", "F-014", "F-015", "F-016"],
	);
	assert.equal(expected.threshold, 0.6);
});

test("a strong calibration run is calibrated at recall 1.0 with no rejected lead", () => {
	const result = scoreCalibration(findings("strong"), CALIBRATION_EXPECTED);
	assert.equal(result.calibrated, true, result.detail);
	assert.equal(result.recall, 1);
	assert.equal(result.hit_count, 7);
	assert.equal(result.miss_count, 0);
	assert.equal(result.rejected_confirmed, 0);
	assert.deepEqual(result.narrow_briefs, {});
	assert.deepEqual(result.by_workstream, { S: { expected: 3, hit: 3 }, M: { expected: 2, hit: 2 }, R: { expected: 2, hit: 2 } });
});

test("the weakened brief falls below 0.6 and names the missed findings", () => {
	const result = scoreCalibration(findings("weak"), CALIBRATION_EXPECTED);
	assert.equal(result.calibrated, false);
	assert.ok(result.recall < 0.6);
	assert.deepEqual(result.misses.map((miss) => miss.expected), [
		"WS-S-002",
		"WS-S-003",
		"WS-M-002",
		"WS-R-001",
		"WS-R-004",
	]);
	assert.deepEqual(result.narrow_briefs.S, ["WS-S-002", "WS-S-003"]);
	assert.deepEqual(result.narrow_briefs.M, ["WS-M-002"]);
	assert.deepEqual(result.narrow_briefs.R, ["WS-R-001", "WS-R-004"]);
});

test("confirming a rejected Phase-A lead is uncalibrated even at recall 1.0", () => {
	const result = scoreCalibration(findings("rejected-lead"), CALIBRATION_EXPECTED);
	assert.equal(result.recall, 1);
	assert.equal(result.rejected_confirmed, 1);
	assert.equal(result.rejected_hits[0].lead, "F-014");
	assert.equal(result.calibrated, false);
});

test("the scorer does not count a corrected statement as the rejected lead", () => {
	const corrected = findings("strong").map((finding) =>
		finding.id === "M-002"
			? {
					...finding,
					observed:
						"the baseline is labelled Prophet but is a Vangja ablation: it does not import prophet; the fairness framing (F-015) is the wrong claim",
				}
			: finding,
	);
	const result = scoreCalibration(corrected, CALIBRATION_EXPECTED);
	assert.equal(result.calibrated, true, result.detail);
	assert.equal(result.rejected_confirmed, 0);
});

test("writeCalibrationReport writes REPORT.md, recall.json, and UNCALIBRATED", () => {
	const out = tempOut("vangja-cal-weak-");
	const result = scoreCalibration(findings("weak"), CALIBRATION_EXPECTED);
	const files = writeCalibrationReport({
		outDir: out,
		result,
		mode: "fixture",
		snapshot: CALIBRATION_EXPECTED.snapshot,
		snapshot_readonly: true,
		findings_source: "weak.json",
		expected_source: ".pi/audit/calibration/expected.json",
	});
	assert.ok(fs.existsSync(files.report));
	assert.ok(fs.existsSync(files.recall));
	assert.ok(files.uncalibrated && fs.existsSync(files.uncalibrated));
	const report = readText(files.report);
	assert.match(report, /# Calibration — UNCALIBRATED/);
	assert.match(report, /Recall:\*\* 2\/7 = 0\.286/);
	assert.match(report, /F-014 \| not confirmed \(correct\)/);
	const recall = JSON.parse(readText(files.recall));
	assert.equal(recall.calibrated, false);
	assert.equal(recall.recall, 2 / 7);
	assert.equal(recall.rejected_confirmed, 0);

	const strong = tempOut("vangja-cal-strong-");
	const strongFiles = writeCalibrationReport({
		outDir: strong,
		result: scoreCalibration(findings("strong"), CALIBRATION_EXPECTED),
		mode: "fixture",
		snapshot: CALIBRATION_EXPECTED.snapshot,
		snapshot_readonly: true,
		findings_source: "strong.json",
		expected_source: ".pi/audit/calibration/expected.json",
	});
	assert.equal(strongFiles.uncalibrated, null);
	assert.match(readText(strongFiles.report), /# Calibration — CALIBRATED/);
});

test("an uncalibrated result refuses the grade even with a perfect rubric", () => {
	const result = scoreCalibration(findings("weak"), CALIBRATION_EXPECTED);
	const grade = auditGrade({
		findings: [],
		gates: gateResults(),
		rubric: CLEAN_RUBRIC,
		calibration: {
			recall: result.recall,
			rejected_confirmed: result.rejected_confirmed,
			calibrated: result.calibrated,
		},
		traceability: { ok: true },
		limitations_disclosed: true,
	});
	assert.equal(grade.usable, false);
	assert.equal(grade.grade, "uncalibrated");
	assert.match(grade.reason, /UNCALIBRATED/);
});

test("calibrate.sh --weakened exits 10 and writes the UNCALIBRATED marker", () => {
	const out = tempOut("vangja-cal-script-weak-");
	const result = spawnSync(path.join(REPO_ROOT, ".pi", "calibrate.sh"), ["--weakened", "--out", out], {
		cwd: REPO_ROOT,
		encoding: "utf8",
	});
	assert.equal(result.status, 10, `${result.stdout}\n${result.stderr}`);
	assert.ok(fs.existsSync(path.join(out, "UNCALIBRATED")));
	assert.ok(fs.existsSync(path.join(out, "REPORT.md")));
	assert.ok(fs.existsSync(path.join(out, "recall.json")));
	assert.match(readText(path.join(out, "REPORT.md")), /UNCALIBRATED/);
});

test("calibrate.sh --fixture strong exits 0 and reports recall 1.0", () => {
	const out = tempOut("vangja-cal-script-strong-");
	const result = spawnSync(
		path.join(REPO_ROOT, ".pi", "calibrate.sh"),
		["--fixture", path.join(CALIBRATION_FIXTURES, "strong.json"), "--out", out],
		{ cwd: REPO_ROOT, encoding: "utf8" },
	);
	assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
	const recall = JSON.parse(readText(path.join(out, "recall.json")));
	assert.equal(recall.calibrated, true);
	assert.equal(recall.recall, 1);
	assert.equal(recall.rejected_confirmed, 0);
});

test("calibrate.sh --dry-run prints the pinned command and writes nothing", () => {
	const out = fs.mkdtempSync(path.join(os.tmpdir(), "vangja-cal-dry-"));
	const result = spawnSync(
		path.join(REPO_ROOT, ".pi", "calibrate.sh"),
		["--run-finders", "--dry-run", "--out", out],
		{ cwd: REPO_ROOT, encoding: "utf8" },
	);
	assert.equal(result.status, 0, result.stderr);
	const output = `${result.stdout}${result.stderr}`;
	assert.match(output, /--provider opencode-go/);
	assert.match(output, /--model opencode-go\/deepseek-v4.1-flash/);
	assert.match(output, /--thinking max/);
	assert.match(output, /PI_LOOP_ROLE=audit_finder/);
	assert.match(output, /WP-S|WS-S/);
	assert.ok(!fs.existsSync(path.join(out, "REPORT.md")));
});
