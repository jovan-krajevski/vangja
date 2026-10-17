/**
 * Rubric validation, weight invariants, autofill from gates, and evidence
 * pointers on every LLM score.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { AUTO_SOURCES, autofillRubric, loadRubric, renderRubric, scoreRubric, validateRubric, rubricStats, type RubricTree } from "../../extensions/rubric/lib.ts";
import { gateResults, fullScores, REPO_ROOT } from "./helpers.ts";

const NAMES = ["pr_review", "readiness", "paper", "audit", "decision"] as const;

test("all five rubrics load and validate", () => {
	for (const name of NAMES) {
		const rubric = loadRubric(REPO_ROOT, name);
		const validation = validateRubric(rubric);
		assert.equal(validation.ok, true, `${name}: ${validation.errors.join("; ")}`);
		if (name === "decision") {
			assert.equal(validation.kind, "decision");
		} else {
			assert.equal(validation.kind, "rubric");
			assert.ok(validation.branches >= 8, `${name} has ${validation.branches} branches`);
			assert.ok(validation.leaves >= 8, `${name} has ${validation.leaves} leaves`);
		}
	}
});

test("every branch and the top level sum to 1", () => {
	for (const name of NAMES) {
		const rubric = loadRubric(REPO_ROOT, name);
		if (rubric.kind !== "rubric") continue;
		const tree = rubric as RubricTree;
		const top = tree.branches.reduce((sum, branch) => sum + branch.weight, 0);
		assert.ok(Math.abs(top - 1) < 1e-9, `${name}: top-level weights sum to ${top}`);
		for (const branch of tree.branches) {
			const sum = branch.children.reduce((total, leaf) => total + leaf.weight, 0);
			assert.ok(Math.abs(sum - 1) < 1e-9, `${name}/${branch.id}: weights sum to ${sum}`);
		}
	}
});

test("every auto leaf names a known deterministic source and every leaf has a criterion", () => {
	for (const name of NAMES) {
		const rubric = loadRubric(REPO_ROOT, name);
		if (rubric.kind !== "rubric") continue;
		for (const branch of (rubric as RubricTree).branches) {
			for (const leaf of branch.children) {
				assert.ok(leaf.criterion.length > 10, `${name}/${leaf.id} has no criterion`);
				if (leaf.type === "auto") assert.ok(leaf.auto && leaf.auto in AUTO_SOURCES, `${leaf.id}: ${leaf.auto}`);
			}
		}
	}
});

test("the audit rubric has one branch per workstream plus five cross-cutting branches", () => {
	const tree = loadRubric(REPO_ROOT, "audit") as RubricTree;
	const ids = tree.branches.map((branch) => branch.id);
	for (const ws of ["M", "S", "P", "D", "I", "R", "H", "E", "C", "L"]) {
		assert.ok(ids.includes(`ws_${ws}`), `missing ws_${ws}`);
	}
	for (const cross of ["evidence_quality", "coverage", "refutation_effort", "reproducibility", "integrity"]) {
		assert.ok(ids.includes(cross), `missing ${cross}`);
	}
	assert.equal(ids.length, 15);
	const stats = rubricStats(tree);
	assert.equal(stats.branches, 15);
});

test("weight drift and duplicate ids are rejected", () => {
	const tree = JSON.parse(JSON.stringify(loadRubric(REPO_ROOT, "pr_review"))) as RubricTree;
	tree.branches[0].weight = 0.99;
	const drift = validateRubric(tree);
	assert.equal(drift.ok, false);
	assert.match(drift.errors.join(" "), /top-level branches: weights sum/);

	const duplicate = JSON.parse(JSON.stringify(loadRubric(REPO_ROOT, "pr_review"))) as RubricTree;
	duplicate.branches[1].children[0].id = duplicate.branches[0].children[0].id;
	const dup = validateRubric(duplicate);
	assert.equal(dup.ok, false);
	assert.match(dup.errors.join(" "), /duplicate leaf id/);
});

test("an auto leaf without a known source is rejected", () => {
	const tree = JSON.parse(JSON.stringify(loadRubric(REPO_ROOT, "readiness"))) as RubricTree;
	const leaf = tree.branches[0].children[0];
	leaf.type = "auto";
	leaf.auto = "vibes";
	const validation = validateRubric(tree);
	assert.equal(validation.ok, false);
	assert.match(validation.errors.join(" "), /auto leaf needs one of/);
});

test("autofill reads gate results, including hash_ok", () => {
	const tree = loadRubric(REPO_ROOT, "pr_review") as RubricTree;
	const gates = gateResults();
	gates.gates = gates.gates.map((gate) => (gate.id === "test-diff" ? { ...gate, status: "fail" } : gate));
	gates.gates = gates.gates.map((gate) => (gate.id === "artifacts-complete" ? { ...gate, hash_ok: false } : gate));
	const filled = autofillRubric(tree, { gates });
	const leaf = (id: string) => filled.tree.branches.flatMap((branch) => branch.children).find((entry) => entry.id === id);
	assert.equal(leaf("tests_pass")?.score, 1);
	assert.equal(leaf("test_diff_clean")?.score, 0);
	assert.equal(leaf("hash_ok")?.score, 0);
	assert.equal(leaf("hash_ok")?.evidence, "gate:artifacts-complete#hash_ok=false");
	assert.ok(filled.filled.includes("tests_pass"));
	assert.deepEqual(filled.unresolved, []);
});

test("a skipped gate stays unresolved — a skip proves nothing", () => {
	const tree = loadRubric(REPO_ROOT, "readiness") as RubricTree;
	const gates = gateResults();
	gates.gates = gates.gates.map((gate) => (gate.id === "gaming-scan" ? { ...gate, status: "skip" } : gate));
	const filled = autofillRubric(tree, { gates });
	assert.ok(filled.unresolved.some((entry) => entry.id === "gaming_scan_clean" || entry.id === "tests_pass") === false);
	const audit = loadRubric(REPO_ROOT, "audit") as RubricTree;
	const auditFilled = autofillRubric(audit, { gates });
	assert.ok(auditFilled.unresolved.some((entry) => entry.id === "gaming_scan_clean"));
});

test("a complete autofill + full evidence scores 1.0", () => {
	const tree = loadRubric(REPO_ROOT, "pr_review") as RubricTree;
	const filled = autofillRubric(tree, { gates: gateResults() });
	assert.deepEqual(filled.unresolved, []);
	const result = scoreRubric(filled.tree, fullScores(filled.tree));
	assert.equal(result.ok, true, result.errors.join("; "));
	assert.equal(result.complete, true);
	assert.equal(result.total, 1);
	assert.ok(result.branches.every((branch) => branch.score === 1));
});

test("an LLM score without an evidence pointer is rejected", () => {
	const tree = loadRubric(REPO_ROOT, "pr_review") as RubricTree;
	const filled = autofillRubric(tree, { gates: gateResults() });
	const scores = fullScores(filled.tree);
	scores[0].evidence = "   ";
	const result = scoreRubric(filled.tree, scores);
	assert.equal(result.ok, false);
	assert.match(result.errors.join(" "), /requires an evidence pointer/);
	assert.equal(result.total, null);
});

test("an unknown leaf, duplicate score, or binary 0.5 is rejected", () => {
	const tree = loadRubric(REPO_ROOT, "pr_review") as RubricTree;
	const filled = autofillRubric(tree, { gates: gateResults() });
	const scores = fullScores(filled.tree);

	const unknown = scoreRubric(filled.tree, [...scores, { leaf_id: "not_a_leaf", score: 1, evidence: "x" }]);
	assert.equal(unknown.ok, false);
	assert.match(unknown.errors.join(" "), /unknown leaf/);

	const duplicate = scoreRubric(filled.tree, [...scores, scores[0]]);
	assert.equal(duplicate.ok, false);
	assert.match(duplicate.errors.join(" "), /scored more than once/);

	const fractional = scoreRubric(
		filled.tree,
		scores.map((entry, index) => (index === 0 ? { ...entry, score: 0.5 } : entry)),
	);
	assert.equal(fractional.ok, false);
	assert.match(fractional.errors.join(" "), /binary leaf accepts only 0 or 1/);
});

test("an auto leaf may not be scored manually", () => {
	const tree = loadRubric(REPO_ROOT, "pr_review") as RubricTree;
	const filled = autofillRubric(tree, { gates: gateResults() });
	const autoLeaf = filled.tree.branches.flatMap((branch) => branch.children).find((leaf) => leaf.type === "auto");
	assert.ok(autoLeaf);
	const result = scoreRubric(filled.tree, [
		...fullScores(filled.tree),
		{ leaf_id: autoLeaf!.id, score: 0, evidence: "manual override" },
	]);
	assert.equal(result.ok, false);
	assert.match(result.errors.join(" "), /auto leaf is filled by rubric_autofill/);
});

test("an unresolved leaf makes the total null, never a silent zero", () => {
	const tree = loadRubric(REPO_ROOT, "readiness") as RubricTree;
	const filled = autofillRubric(tree, { gates: null });
	const scores = fullScores(filled.tree);
	const result = scoreRubric(filled.tree, scores);
	assert.equal(result.total, null);
	assert.equal(result.complete, false);
	assert.ok(result.unresolved.length > 0);
});

test("renderRubric emits the branch/leaf/weight/score/evidence table", () => {
	const tree = loadRubric(REPO_ROOT, "pr_review") as RubricTree;
	const filled = autofillRubric(tree, { gates: gateResults() });
	const result = scoreRubric(filled.tree, fullScores(filled.tree));
	const markdown = renderRubric(filled.tree, result);
	assert.match(markdown, /### Rubric: PR review rubric/);
	assert.match(markdown, /\| Branch \| Leaf \| Type \| Weight \| Score \| Evidence \|/);
	assert.match(markdown, /\*\*Total:\*\* 1\.000/);
});
