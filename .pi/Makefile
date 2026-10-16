# Environment verification for the Vangja research-agent loop.
#
# Run from the repository root (`make -C .pi <target>`) or from here.
# Targets:
#   test               unit + integration suite (offline, no model calls)
#   env-check          config/settings/roster/secrets checks with PASS lines
#   verify-env         test + env-check, plus an explicit STUB banner for the S4 command
#   verify-env-strict  same, but fails until /loop:verify-env exists (S4)
#   accept             model-backed acceptance checks (small token cost)
#   all                verify-env
#   help               this list

SHELL := /bin/bash
ROOT := $(abspath $(CURDIR)/..)
NODE ?= node

TEST_FILES := .pi/tests/test_settings.test.ts .pi/tests/test_schemas.test.ts \
	.pi/tests/test_guards_policy.test.ts .pi/tests/test_guards_wiring.test.ts \
	.pi/tests/test_ledger.test.ts .pi/tests/test_ledger_concurrency.test.ts \
	.pi/tests/test_status.test.ts .pi/tests/test_child_pi.test.ts

.PHONY: all test env-check verify-env verify-env-strict accept help

all: verify-env

test:
	@echo "== node --test (offline) =="
	cd $(ROOT) && $(NODE) --test $(TEST_FILES)

env-check:
	@echo "== environment checks =="
	cd $(ROOT) && $(NODE) .pi/tests/check_env.ts

verify-env: test env-check stub-orchestrator
	@echo
	@echo "verify-env: environment tests green."
	@echo "verify-env: NOTE the S4 stub above — the orchestrator does not exist yet."

verify-env-strict: test env-check
	@echo
	@echo "STUB (S4): '/loop:verify-env' and .pi/loop/step.sh are not implemented yet." >&2
	@echo "           This target fails on purpose until research_agent/prompts/04 lands." >&2
	@exit 1

# Not part of verify-env: these make real model calls.
accept:
	@echo "== acceptance checks (uses the pinned model; small token cost) =="
	bash $(ROOT)/.pi/tests/acceptance/run_acceptance.sh

stub-orchestrator:
	@echo
	@echo "STUB (S4): 'pi --print \"/loop:verify-env\"' is not available; the orchestrator"
	@echo "           (research_agent/prompts/04_orchestrator_and_state.md) has not been built."
	@echo "           Use 'make -C .pi verify-env-strict' in CI to fail on this stub."

help:
	@grep -E '^[a-z-]+:' $(CURDIR)/Makefile | sed 's/:.*//' | sort -u
