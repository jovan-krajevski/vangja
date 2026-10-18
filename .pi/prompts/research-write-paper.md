---
description: Write one paper section from verified artifacts
argument-hint: "<section>"
---
Section: `${1:-<section>}`. Run this as a `paper_writer` work unit in the PAPER
state; read the current `paper/` state and `paper/CLAIMS.json` first.

Write prose/tables from artifacts only. Every number cites an artifact path plus
its manifest SHA; every table/figure regenerates from a committed generator (no
hand-typed values). Add or update the `claims` entry for each reported number and
run the claims check. State dataset, horizon, exclusions, seeds, and uncertainty,
and keep the limitations section honest (negative results included).

Acceptance evidence: the section diff + regenerated `paper/generated/**` +
updated `CLAIMS.json` + the build command with observed output. If an artifact is
missing or its SHA is not an ancestor of the base branch, stop with `BLOCKED`;
propose the narrower claim the artifacts do support instead of typing a number.
Use `literature_scout` for any positioning claim; never invent a citation.
