# PR-### · {{TITLE}}

## Summary

{{SUMMARY}}

- Task: {{TASK_ID}} — {{TASK_TITLE}}
- Branch: `{{BRANCH}}` → `{{BASE_SHA}}`…`{{HEAD_SHA}}`
- State at open: `{{STATUS}}`

## Acceptance criteria → evidence

| Criterion | Command | Observed |
|---|---|---|
{{ACCEPTANCE_ROWS}}

## Artifacts

{{ARTIFACTS}}

## Claims

{{CLAIMS}}

## Tests run

{{TESTS}}

## HPC jobs

{{HPC_JOBS}}

## Diff scan

```
{{DIFF_SCAN}}
```

## Risks and residual uncertainty

{{RISKS}}

## Reviewer checklist (for the supervisor)

- [ ] Acceptance criteria each have a command and an observed result.
- [ ] Every number in the body cites an artifact path + manifest SHA.
- [ ] Tests were added/updated, none weakened or deleted.
- [ ] Protected paths: none touched, or an amendment ID is recorded.
- [ ] No force push, no history rewrite, single logical change set.
- [ ] `make -C .pi verify-env` (or the narrowest named suite) is green.

<!-- Machine-readable footer: pass this object to pr_open. -->
<!-- PR_FOOTER_JSON
{{FOOTER_JSON}}
-->
