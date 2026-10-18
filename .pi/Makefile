# Environment verification for the Vangja research-agent loop.
#
# Run from the repository root (`make -C .pi <target>`) or from here.
# Targets:
#   test               unit + integration suite (offline, no model calls)
#   skills-test        validate .pi/skills + run the skill-script tests
#   env-check          config/settings/roster/secrets checks with PASS lines
#   model-pin          audit .pi/state/EVENTS.jsonl for pinned model/thinking
#   verify-env         test + skills-test + env-check + orchestrator-check + model-pin
#   smoke              end-to-end sandbox loop (BOOT → audit → decision; no model calls)
#   resilience         kill/restart matrix with exactly-once assertions
#   calibration        offline audit-calibration modes (strong/weakened/rejected)
#   all                verify-env test smoke
#   accept             model-backed acceptance checks (small token cost)
#   help               this list

SHELL := /bin/bash
ROOT := $(abspath $(CURDIR)/..)
NODE ?= node

TEST_FILES := .pi/tests/test_settings.test.ts .pi/tests/test_schemas.test.ts \
	.pi/tests/test_guards_policy.test.ts .pi/tests/test_guards_wiring.test.ts \
	.pi/tests/test_ledger.test.ts .pi/tests/test_ledger_concurrency.test.ts \
	.pi/tests/test_status.test.ts .pi/tests/test_understand_loop.test.ts .pi/tests/test_child_pi.test.ts \
	.pi/tests/agents/test_agents_manifest.test.ts .pi/tests/agents/test_tools_registry.test.ts \
	.pi/tests/agents/test_dispatch_unit.test.ts .pi/tests/agents/test_prompt_templates.test.ts \
	.pi/tests/orchestrator/test_state_machine.test.ts .pi/tests/orchestrator/test_budget.test.ts \
	.pi/tests/orchestrator/test_gates.test.ts .pi/tests/orchestrator/test_cli_status.test.ts \
	.pi/tests/orchestrator/test_pause_resume.test.ts .pi/tests/orchestrator/test_sandbox_cycle.test.ts \
	.pi/tests/hpc/test_hpc_disabled.test.ts .pi/tests/hpc/test_hpc_config.test.ts \
	.pi/tests/hpc/test_hpc_submit_gate.test.ts .pi/tests/hpc/test_hpc_lifecycle.test.ts \
	.pi/tests/hpc/test_hpc_retry_rules.test.ts \
	.pi/tests/audit/test_findings.test.ts .pi/tests/audit/test_grade.test.ts \
	.pi/tests/audit/test_rubrics.test.ts .pi/tests/audit/test_verify.test.ts \
	.pi/tests/audit/test_canary.test.ts .pi/tests/audit/test_calibration.test.ts \
	.pi/tests/audit/test_templates.test.ts .pi/tests/audit/test_audit_wiring.test.ts \
	.pi/tests/audit/test_register_concurrency.test.ts \
	.pi/tests/e2e/test_model_pin_audit.test.ts

.PHONY: all test skills-test env-check model-pin verify-env verify-env-strict orchestrator-check orchestrator-accept hpc-accept audit-accept audit-calibrate calibration smoke resilience accept agents-accept help

all: verify-env test smoke

test:
	@echo "== node --test (offline) =="
	cd $(ROOT) && $(NODE) --test $(TEST_FILES)

skills-test:
	@echo "== skills (offline) =="
	cd $(ROOT) && bash .pi/tests/run.sh skills

env-check:
	@echo "== environment checks =="
	cd $(ROOT) && $(NODE) .pi/tests/check_env.ts

verify-env: test skills-test env-check orchestrator-check model-pin
	@echo
	@echo "verify-env: environment tests green (every stage's suite, skills, schemas, model pin)."

verify-env-strict: verify-env

# Offline: every recorded child argv must use the pinned model and thinking level.
model-pin:
	@echo
	@echo "== model-pin audit (.pi/state/EVENTS.jsonl) =="
	cd $(ROOT) && $(NODE) .pi/tests/e2e/audit_model_pin.ts

# Offline: reads the environment, runs the orchestrator's own checks, writes nothing.
orchestrator-check:
	@echo
	@echo "== orchestrator verify-env =="
	cd $(ROOT) && $(NODE) .pi/extensions/orchestrator/cli.ts verify-env --headless

# The full S4 acceptance transcript (sandbox only; no model calls).
orchestrator-accept:
	bash $(ROOT)/.pi/tests/orchestrator/run_orchestrator_acceptance.sh

# The full S5 mock-cluster acceptance transcript (mock shims only; no network,
# no model calls). Not part of verify-env: it takes ~1 minute.
hpc-accept:
	bash $(ROOT)/.pi/tests/hpc/run_mock_cluster.sh

# The full S6 audit acceptance transcript (offline fixtures; writes only temp dirs).
audit-accept:
	bash $(ROOT)/.pi/tests/audit/run_audit_acceptance.sh

# Offline calibration scorer on the committed strong fixture (no finder/model calls).
audit-calibrate:
	bash $(ROOT)/.pi/calibrate.sh --fixture .pi/tests/audit/fixtures/calibration/strong.json --out /tmp/vangja-calibration-strong

# The three offline calibration modes + refusal paths (no model calls).
calibration:
	bash $(ROOT)/.pi/tests/e2e/calibration.sh

# End-to-end sandbox loop: BOOT → work → review → merge → readiness → paper →
# audit → decision, with the tampered-artifact and seeded-defect checks.
# Deterministic child stub only: no model calls, no network, no HPC.
smoke:
	cd $(ROOT) && $(NODE) .pi/tests/e2e/smoke.ts

# Kill/restart matrix with exactly-once assertions (no model calls).
resilience:
	cd $(ROOT) && $(NODE) .pi/tests/e2e/resilience.ts

# Not part of verify-env: these make real model calls.
accept:
	@echo "== acceptance checks (uses the pinned model; small token cost) =="
	bash $(ROOT)/.pi/tests/acceptance/run_acceptance.sh
	@echo
	echo "== S3 agent/dispatch acceptance (uses the pinned model) =="
	bash $(ROOT)/.pi/tests/agents/run_agent_acceptance.sh

agents-accept:
	bash $(ROOT)/.pi/tests/agents/run_agent_acceptance.sh

help:
	@grep -E '^[a-z-]+:' $(CURDIR)/Makefile | sed 's/:.*//' | sort -u
