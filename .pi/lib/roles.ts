/**
 * Role capability matrix. This is the single source of truth for both the
 * guard extension (intercepting tool calls) and the ledger tools (refusing an
 * operation at its handler). Keep it in sync with
 * `research_agent/plan/05_EXTENSIONS_AND_TOOLS.md` §7.
 */

export const LOOP_ROLES = [
	"conductor",
	"phd_candidate",
	"collector",
	"supervisor",
	"audit_finder",
	"audit_verifier",
	"audit_synthesizer",
	"audit_adjudicator",
	"meta_review",
	"scout",
	"planner",
	"methodologist",
	"experimenter",
	"repro_engineer",
	"literature_scout",
	"paper_writer",
	"critic",
	"human",
] as const;

export type LoopRole = (typeof LOOP_ROLES)[number];

export type Capability =
	| "merge_pr"
	| "hpc_submit"
	| "hpc_cancel"
	| "hpc_requeue"
	| "hpc_read"
	| "state_set"
	| "review_write"
	| "waiver_record"
	| "amendment_record"
	| "task_enqueue"
	| "task_close"
	| "pr_open"
	| "experiment_queue"
	| "experiment_update"
	| "rubric_score"
	| "auditkit_write"
	| "dispatch"
	| "verify_write";

const AUDIT_ROLES: LoopRole[] = [
	"audit_finder",
	"audit_verifier",
	"audit_synthesizer",
	"audit_adjudicator",
	"meta_review",
];

export const ROLE_MATRIX: Record<Capability, LoopRole[]> = {
	// plan/05 §7: merge_pr = supervisor (executes via conductor)
	merge_pr: ["supervisor", "conductor"],
	// hpc_submit = phd_candidate (gate-checked)
	hpc_submit: ["phd_candidate", "conductor"],
	// hpc_cancel = supervisor
	hpc_cancel: ["supervisor", "conductor"],
	// hpc_requeue = collector only
	hpc_requeue: ["collector", "conductor"],
	// hpc_status/fetch/tail = every role
	hpc_read: [...LOOP_ROLES],
	state_set: ["conductor"],
	review_write: ["supervisor", "conductor"],
	waiver_record: ["supervisor", "conductor"],
	amendment_record: ["supervisor", "conductor"],
	task_enqueue: ["phd_candidate", "supervisor", "conductor"],
	task_close: ["conductor", "supervisor"],
	pr_open: ["phd_candidate", "conductor"],
	experiment_queue: ["phd_candidate", "conductor"],
	experiment_update: ["collector", "conductor"],
	rubric_score: ["supervisor", ...AUDIT_ROLES, "conductor"],
	auditkit_write: ["supervisor", ...AUDIT_ROLES, "conductor"],
	dispatch: ["phd_candidate", "supervisor", ...AUDIT_ROLES],
	verify_write: ["phd_candidate", "collector", "supervisor", ...AUDIT_ROLES, "conductor"],
};

/** Map tool names (or planned tool names) to the capability they need. */
export const TOOL_CAPABILITIES: Record<string, Capability> = {
	merge_pr: "merge_pr",
	hpc_submit: "hpc_submit",
	hpc_cancel: "hpc_cancel",
	hpc_requeue: "hpc_requeue",
	hpc_status: "hpc_read",
	hpc_fetch: "hpc_read",
	hpc_tail: "hpc_read",
	hpc_config: "hpc_read",
	hpc_checkpoints: "hpc_read",
	state_set: "state_set",
	review_write: "review_write",
	waiver_record: "waiver_record",
	amendment_record: "amendment_record",
	task_enqueue: "task_enqueue",
	task_close: "task_close",
	pr_open: "pr_open",
	experiment_queue: "experiment_queue",
	experiment_update: "experiment_update",
	rubric_score: "rubric_score",
	auditkit_write: "auditkit_write",
	dispatch: "dispatch",
};

export function isLoopRole(value: string): value is LoopRole {
	return (LOOP_ROLES as readonly string[]).includes(value);
}

export function parseRole(value: string | undefined): LoopRole {
	if (value === undefined || value.trim() === "") return "human";
	if (!isLoopRole(value)) {
		throw new Error(
			`unknown PI_LOOP_ROLE="${value}". Valid roles: ${LOOP_ROLES.join(", ")}. ` +
				`See .pi/config/agents.json for the roster.`,
		);
	}
	return value;
}

export interface RoleDecision {
	allowed: boolean;
	reason?: string;
}

export function roleGate(capability: Capability, role: LoopRole): RoleDecision {
	const allowed = ROLE_MATRIX[capability];
	if (allowed.includes(role)) return { allowed: true };
	return {
		allowed: false,
		reason:
			`role "${role}" may not use ${capability}; allowed roles: ${allowed.join(", ")} ` +
			`(matrix: research_agent/plan/05_EXTENSIONS_AND_TOOLS.md §7). ` +
			`If this is wrong, the conductor must set a different PI_LOOP_ROLE for this session — ` +
			`do not widen the matrix from inside a session.`,
	};
}

export function roleGateForTool(toolName: string, role: LoopRole): RoleDecision {
	const capability = TOOL_CAPABILITIES[toolName];
	if (!capability) return { allowed: true };
	return roleGate(capability, role);
}

export function isAuditRole(role: LoopRole): boolean {
	return AUDIT_ROLES.includes(role);
}
