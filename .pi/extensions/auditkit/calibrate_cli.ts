#!/usr/bin/env node
/**
 * Calibration scorer CLI (plan/07_AUDIT_HARNESS.md §7).
 *
 * Usage:
 *   node .pi/extensions/auditkit/calibrate_cli.ts \
 *     --findings <file|dir> --out <dir> [--mode fixture|collect|finder] \
 *     [--snapshot <sha>] [--expected <path>] [--note <text>]
 *
 * Exit codes: 0 = calibrated; 10 = UNCALIBRATED (the orchestrator must refuse to
 * grade); 2 = usage/input error.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import {
	loadCalibrationExpected,
	loadFindingsDir,
	loadFindingsFile,
	scoreCalibration,
	writeCalibrationReport,
} from "./calibrate.ts";

interface Args {
	findings?: string;
	out?: string;
	mode?: "fixture" | "collect" | "finder";
	snapshot?: string;
	expected?: string;
	note?: string;
}

function parseArgs(argv: string[]): Args {
	const args: Args = {};
	for (let index = 0; index < argv.length; index += 1) {
		const flag = argv[index];
		const value = argv[index + 1];
		switch (flag) {
			case "--findings":
				args.findings = value;
				index += 1;
				break;
			case "--out":
				args.out = value;
				index += 1;
				break;
			case "--mode":
				args.mode = value as Args["mode"];
				index += 1;
				break;
			case "--snapshot":
				args.snapshot = value;
				index += 1;
				break;
			case "--expected":
				args.expected = value;
				index += 1;
				break;
			case "--note":
				args.note = value;
				index += 1;
				break;
			default:
				if (flag.startsWith("--")) throw new Error(`unknown argument ${flag}`);
		}
	}
	return args;
}

function main(): number {
	const root = process.cwd();
	let args: Args;
	try {
		args = parseArgs(process.argv.slice(2));
	} catch (error) {
		console.error((error as Error).message);
		return 2;
	}
	if (!args.findings || !args.out) {
		console.error("usage: calibrate_cli.ts --findings <file|dir> --out <dir> [--mode fixture|collect|finder]");
		return 2;
	}
	const findingsPath = path.resolve(root, args.findings);
	if (!fs.existsSync(findingsPath)) {
		console.error(`findings not found: ${findingsPath}`);
		return 2;
	}
	const findings = fs.statSync(findingsPath).isDirectory()
		? loadFindingsDir(findingsPath)
		: loadFindingsFile(findingsPath);
	const expected = args.expected
		? (JSON.parse(fs.readFileSync(path.resolve(root, args.expected), "utf8")) as ReturnType<typeof loadCalibrationExpected>)
		: loadCalibrationExpected(root);
	const result = scoreCalibration(findings, expected);
	const files = writeCalibrationReport({
		outDir: path.resolve(root, args.out),
		result,
		mode: args.mode ?? "collect",
		snapshot: args.snapshot ?? expected.snapshot,
		snapshot_readonly: true,
		findings_source: args.findings,
		expected_source: args.expected ?? ".pi/audit/calibration/expected.json",
		note: args.note,
	});
	// `latest` is a deterministic pointer for rubric_autofill and audit_grade.
	// Fixture-mode runs must not overwrite it: they are tests of the scorer, not
	// the calibration of the current audit cycle.
	if ((args.mode ?? "collect") !== "fixture") {
		const latest = path.join(root, ".pi", "audit", "calibration", "latest");
		fs.mkdirSync(latest, { recursive: true });
		fs.copyFileSync(files.report, path.join(latest, "REPORT.md"));
		fs.copyFileSync(files.recall, path.join(latest, "recall.json"));
		const marker = path.join(latest, "UNCALIBRATED");
		if (files.uncalibrated) fs.writeFileSync(marker, `${result.detail}\n`);
		else if (fs.existsSync(marker)) fs.rmSync(marker);
	}

	console.log(
		JSON.stringify(
			{
				calibrated: result.calibrated,
				recall: result.recall,
				hit_count: result.hit_count,
				cases: result.cases,
				rejected_confirmed: result.rejected_confirmed,
				misses: result.misses.map((miss) => miss.expected),
				rejected: result.rejected_hits.map((hit) => hit.lead),
				report: files.report,
				recall_file: files.recall,
				uncalibrated: files.uncalibrated,
			},
			null,
			2,
		),
	);
	return result.calibrated ? 0 : 10;
}

process.exit(main());
