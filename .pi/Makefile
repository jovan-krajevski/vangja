# Environment verification for the Vangja research-agent loop.
#
# Run from the repository root (`make -C .pi <target>`) or from here.
# Targets:
#   test               unit + integration suite (offline, no model calls)
#   skills-test        validate .pi/skills + run the skill-script tests
#   env-check          config/settings/roster/secrets checks with PASS lines
#   verify-env         test + skills-test + env-check + orchestrator-check
#   verify-env-strict  alias of verify-env (kept for the S0..S3 reports)
#   accept             model-backed acceptance checks (small token cost)
#   all                verify-env
#   help               this list

SHELL := /bin/bash
ROOT := $(abspath $(CURDIR)/..)
NODE ?= node

TEST_FILES := .pi/tests/test_settings.test.ts .pi/tests/test_schemas.test.ts \
	.pi/tests/test_guards_policy.test.ts .pi/tests/test_guards_wiring.test.ts \
	.pi/tests/test_ledger.test.ts .pi/tests/test_ledger_concurrency.test.ts \
	.pi/tests/test_status.test.ts .pi/tests/test_child_pi.test.ts \
	.pi/tests/agents/test_agents_manifest.test.ts .pi/tests/agents/test_tools_registry.test.ts \
	.pi/tests/agents/test_dispatch_unit.test.ts .pi/tests/agents/test_prompt_templates.test.ts \
	.pi/tests/orchestrator/test_state_machine.test.ts .pi/tests/orchestrator/test_budget.test.ts \
	.pi/tests/orchestrator/test_gates.test.ts .pi/tests/orchestrator/test_cli_status.test.ts \
	.pi/tests/orchestrator/test_pause_resume.test.ts .pi/tests/orchestrator/test_sandbox_cycle.test.ts \
	.pi/tests/hpc/test_hpc_disabled.test.ts .pi/tests/hpc/test_hpc_config.test.ts \
	.pi/tests/hpc/test_hpc_submit_gate.test.ts .pi/tests/hpc/test_hpc_lifecycle.test.ts \
	.pi/tests/hpc/test_hpc_retry_rules.test.ts

.PHONY: all test skills-test env-check verify-env verify-env-strict orchestrator-check orchestrator-accept hpc-accept accept agents-accept help

all: verify-env

test:
	@echo "== node --test (offline) =="
	cd $(ROOT) && $(NODE) --test $(TEST_FILES)

skills-test:
	@echo "== skills (offline) =="
	cd $(ROOT) && bash .pi/tests/run.sh skills

env-check:
	@echo "== environment checks =="
	cd $(ROOT) && $(NODE) .pi/tests/check_env.ts

verify-env: test skills-test env-check orchestrator-check
	@echo
	@echo "verify-env: environment tests green (the orchestrator is implemented; see loop/README.md)."

verify-env-strict: verify-env

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
