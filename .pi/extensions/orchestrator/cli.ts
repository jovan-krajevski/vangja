#!/usr/bin/env node
/**
 * The orchestrator CLI — the single implementation behind `/loop:*` and the
 * `.pi/loop/*.sh` shims (plan/05_EXTENSIONS_AND_TOOLS.md §2).
 *
 * Usage: node .pi/extensions/orchestrator/cli.ts <command> [flags]
 *   start | step | pause [reason] | resume | status | verify-env
 *   gate list | gate approve <id> <note> | gate reject <id> <why>
 *   collect (S5 placeholder; exits 2 until the collector exists)
 *
 * Flags: --sandbox <path> --dry-run --headless --json --forever --max-steps <n> --command <pi>
 */

import { parseFlags, resolveRunContext, type CliFlags, type RunContext } from "./sandbox.ts";
import {
	commandGate,
	commandPause,
	commandResume,
	commandStart,
	commandStatus,
	commandStep,
	commandVerifyEnv,
	loadLoopOptions,
} from "./commands.ts";
import { hpcGate } from "./gates.ts";
import { describeAction } from "./state-machine.ts";

const USAGE = `orchestrator CLI (plan/05_EXTENSIONS_AND_TOOLS.md §2)

  node .pi/extensions/orchestrator/cli.ts <command> [flags]

commands:
  start            step until the loop pauses, blocks, finishes, or hits --max-steps
  step             perform exactly one action
  pause [reason]   write PAUSED with the reason (HPC jobs keep running)
  resume           return to the state recorded before the pause
  status [--json]  state, task/PR/experiment, budget, gates, last 5 events, nextAction
  verify-env       run the environment checks plus the orchestrator's own checks
  gate list|approve <id> <note>|reject <id> <why>
  collect          S5 collector entry point (not implemented yet; exits 2)

flags:
  --sandbox <path>  run against a scratch clone (never the real repository)
  --dry-run         log the action to state/DRYRUN.jsonl; no other side effect
  --headless        no TUI; plain stdout (the .pi/loop/*.sh shims pass this)
  --json            machine-readable output (status)
  --forever         keep stepping up to the cap
  --max-steps <n>   cap for start/--forever (default 60)
  --command <pi>    child \`pi\` executable (tests use a stub)
`;

const TERMINAL = new Set(["PAUSED", "BLOCKED", "DONE"]);

async function runStatus(context: RunContext, flags: CliFlags): Promise<number> {
	const output = commandStatus(context);
	if (flags.json) {
		process.stdout.write(`${JSON.stringify(output.report, null, 2)}\n`);
		return 0;
	}
	for (const line of output.lines) process.stdout.write(`${line}\n`);
	return 0;
}

async function main(argv: string[]): Promise<number> {
	const command = argv[0];
	const flags = parseFlags(argv.slice(1));
	if (!command || command === "help" || command === "--help" || command === "-h") {
		process.stdout.write(USAGE);
		return command === undefined ? 2 : 0;
	}
	let context: RunContext;
	try {
		context = resolveRunContext(flags);
	} catch (error) {
		process.stderr.write(`${(error as Error).message}\n`);
		return 1;
	}
	switch (command) {
		case "start":
		case "step": {
			if (command === "step") {
				const result = await commandStep({ context });
				process.stdout.write(
					`${result.skipped ? `skipped(${result.skipped}) ` : ""}${result.message}\n`,
				);
				if (!result.performed && result.skipped === "child-alive") return 1;
				return 0;
			}
			const steps = await commandStart({
				context,
				maxSteps: flags.forever ? Math.max(flags.maxSteps, 1000) : flags.maxSteps,
				forever: flags.forever,
				onStep: (line) => process.stdout.write(`${line}\n`),
			});
			process.stdout.write(`loop stopped: ${steps.stopped} after ${steps.steps.length} step(s)\n`);
			return 0;
		}
		case "pause": {
			const reason = flags.positional.join(" ").trim() || "paused by human (no reason given)";
			const result = commandPause(context, reason);
			process.stdout.write(`${result.message}\n`);
			return 0;
		}
		case "resume": {
			const result = commandResume(context);
			process.stdout.write(`${result.message}\n`);
			return 0;
		}
		case "status":
			return runStatus(context, flags);
		case "verify-env": {
			const { checks, ok } = commandVerifyEnv(context);
			for (const check of checks) {
				process.stdout.write(`${check.ok ? "PASS" : "FAIL"}  ${check.name}: ${check.detail}\n`);
			}
			process.stdout.write(`${checks.filter((check) => check.ok).length}/${checks.length} orchestrator checks passed\n`);
			return ok ? 0 : 1;
		}
		case "gate": {
			const result = commandGate(context, flags.positional);
			process.stdout.write(`${result.message}\n`);
			if (flags.json && result.gates) process.stdout.write(`${JSON.stringify(result.gates, null, 2)}\n`);
			return result.ok ? 0 : 1;
		}
		case "peek": {
			// Read-only: the action a step would perform (used by tests and humans).
			const { peekAction } = await import("./commands.ts");
			process.stdout.write(`${describeAction(peekAction(context))}\n`);
			return 0;
		}
		case "collect": {
			const hpc = hpcGate(context.root);
			process.stderr.write(
				`collect: the collector arrives in stage S5 (research_agent/prompts/05_hpc_and_collector.md).\n` +
					`hpc gate: ${hpc.status} — ${hpc.reason}\n`,
			);
			return 2;
		}
		default:
			process.stderr.write(`unknown command "${command}"\n\n${USAGE}`);
			return 2;
	}
}

main(process.argv.slice(2))
	.then((code) => {
		process.exitCode = code;
	})
	.catch((error) => {
		process.stderr.write(`${(error as Error).message}\n`);
		process.exitCode = 1;
	});

export { main, TERMINAL, loadLoopOptions };
