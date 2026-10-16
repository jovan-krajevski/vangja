# {{WS}}-{{NNN}} · {{TITLE}}

- **Location:** {{LOCATION}}
- **Snapshot:** {{SNAPSHOT}} (audit-cycle tag @ SHA)
- **Command / test:** {{COMMAND}}
- **Expected:** {{EXPECTED}}
- **Observed:** {{OBSERVED}}
- **Consequence:** {{CONSEQUENCE}}
- **Severity:** {{SEVERITY}}
- **Confidence:** {{CONFIDENCE}}
- **Status:** {{STATUS}}
- **Reproducible:** {{REPRODUCIBLE}}
- **Evidence:** {{EVIDENCE}}
- **Cross-refs:** {{CROSS_REFS}}
- **Owner:** {{OWNER}}
- **Found in cycle:** {{CYCLE}}

<!-- Machine form (findings.json entry) -->
```json
{
  "id": "{{WS}}-{{NNN}}",
  "title": "{{TITLE}}",
  "location": "{{LOCATION}}",
  "snapshot": "{{SNAPSHOT}}",
  "command": "{{COMMAND}}",
  "expected": "{{EXPECTED}}",
  "observed": "{{OBSERVED}}",
  "consequence": "{{CONSEQUENCE}}",
  "severity": "{{SEVERITY}}",
  "confidence": "{{CONFIDENCE}}",
  "status": "{{STATUS}}",
  "reproducible": "{{REPRODUCIBLE}}",
  "evidence": ["{{EVIDENCE}}"],
  "cross_refs": [],
  "owner": "{{OWNER}}",
  "found_in_cycle": 1
}
```

Checklist before writing the finding:

- [ ] The command was actually run in the snapshot, or the file:line was
      opened in this session.
- [ ] `Observed` quotes the output/number, not a paraphrase.
- [ ] `Expected` cites the paper/spec/sound practice, not preference.
- [ ] Severity matches `references/severity-examples.md`.
- [ ] Not a duplicate: search the register for the location and the mechanism.
- [ ] If `null`, `unresolved` or a deferred dataset is involved: record a
      tagged question instead of asserting.
