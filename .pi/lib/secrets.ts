/**
 * Secret scanning and redaction.
 *
 * Used on write/edit tool inputs (block) and on bash tool results (redact).
 * The patterns are conservative on purpose: a false positive is a blocked
 * write the agent can explain, a false negative is a credential in the repo.
 *
 * Key material is never echoed back in full: findings carry a kind and a
 * character offset, and redaction replaces the match with `[REDACTED:<kind>]`.
 */

export interface SecretFinding {
	kind: string;
	index: number;
	length: number;
}

export interface SecretScanResult {
	findings: SecretFinding[];
	redacted: string;
}

interface Rule {
	kind: string;
	pattern: RegExp;
}

// PEM headers are assembled from parts so this file's own source never contains a
// contiguous "BEGIN … PRIVATE KEY" marker that repo-wide secret greps would flag.
const PEM_MARKERS = {
	private: `${["PRIVATE", "KEY"].join(" ")}`,
	certificate: `${["CERTIFICATE"].join("")}`,
};

/** A header without a base64 body is not usable key material; require the body. */
function pemPattern(marker: string): RegExp {
	return new RegExp(`-----BEGIN (?:[A-Z0-9]+ )*${marker}-----[\\s]*[A-Za-z0-9+/=]{32,}`, "g");
}

const RULES: Rule[] = [
	// PEM private keys (RSA/EC/OPENSSH/DSA/PKCS8 and generic "PRIVATE KEY") with a body.
	{ kind: "pem-private-key", pattern: pemPattern(PEM_MARKERS.private) },
	{ kind: "pem-certificate", pattern: pemPattern(PEM_MARKERS.certificate) },
	// Provider/API tokens with recognisable prefixes.
	{ kind: "openai-key", pattern: /\bsk-[A-Za-z0-9_-]{20,}\b/g },
	{ kind: "github-token", pattern: /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b/g },
	{ kind: "github-pat", pattern: /\bgithub_pat_[A-Za-z0-9_]{20,}\b/g },
	{ kind: "slack-token", pattern: /\bxox[baprs]-[A-Za-z0-9-]{10,}\b/g },
	{ kind: "aws-access-key-id", pattern: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g },
	{ kind: "anthropic-key", pattern: /\bsk-ant-[A-Za-z0-9_-]{20,}\b/g },
	{ kind: "google-key", pattern: /\bAIza[0-9A-Za-z_-]{35}\b/g },
	// Bearer tokens in headers and prose.
	{ kind: "bearer-token", pattern: /\bBearer\s+[A-Za-z0-9._~+/=-]{16,}/gi },
	// JWT-shaped values.
	{ kind: "jwt", pattern: /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g },
	// Generic `secret = "long literal"` assignments (>= 24 chars of key-ish data).
	{
		kind: "secret-assignment",
		pattern:
			/\b(?:api[_-]?key|secret|password|passwd|token|private[_-]?key)\b\s*[:=]\s*["'][A-Za-z0-9._/+~=-]{24,}["']/gi,
	},
];

/** Paths that must never be created or overwritten, whatever their content. */
const SECRET_PATH_PATTERNS: RegExp[] = [
	/(^|\/)\.env(\.[^/]*)?$/i,
	/(^|\/)auth\.json$/i,
	/(^|\/)hpc\.json$/i,
	/(^|\/)credentials(\.json)?$/i,
	/(^|\/)secrets?\.(json|ya?ml|toml|ini)$/i,
	/\.pem$/i,
	/\.p12$/i,
	/\.pfx$/i,
	/(^|\/)id_rsa(\.pub)?$/i,
	/(^|\/)id_ed25519(\.pub)?$/i,
	/(^|\/)id_ecdsa(\.pub)?$/i,
];

export function secretPathDecision(relPath: string): { blocked: boolean; reason?: string } {
	for (const pattern of SECRET_PATH_PATTERNS) {
		if (pattern.test(relPath)) {
			return {
				blocked: true,
				reason:
					`"${relPath}" matches a secret-bearing path pattern (${pattern}). ` +
					`Credential files stay outside the repository (e.g. ~/.ssh, or an OS keyring); ` +
					`the committed configuration is .pi/config/hpc.example.json with placeholders.`,
			};
		}
	}
	return { blocked: false };
}

export function scanForSecrets(text: string): SecretScanResult {
	if (text === "") return { findings: [], redacted: text };

	type Match = { kind: string; start: number; end: number };
	const matches: Match[] = [];
	for (const rule of RULES) {
		// Fresh regex state per call for /g patterns.
		const pattern = new RegExp(rule.pattern.source, rule.pattern.flags);
		let match: RegExpExecArray | null;
		while ((match = pattern.exec(text)) !== null) {
			matches.push({ kind: rule.kind, start: match.index, end: match.index + match[0].length });
			if (match[0].length === 0) break;
		}
	}
	if (matches.length === 0) return { findings: [], redacted: text };

	matches.sort((a, b) => a.start - b.start || b.end - a.end);
	// Drop overlapping matches (keep the earliest/longest).
	const kept: Match[] = [];
	for (const match of matches) {
		const previous = kept[kept.length - 1];
		if (previous && match.start < previous.end) continue;
		kept.push(match);
	}

	let redacted = "";
	let cursor = 0;
	const findings: SecretFinding[] = [];
	for (const match of kept) {
		redacted += text.slice(cursor, match.start);
		redacted += `[REDACTED:${match.kind}]`;
		findings.push({ kind: match.kind, index: match.start, length: match.end - match.start });
		cursor = match.end;
	}
	redacted += text.slice(cursor);
	return { findings, redacted };
}

export function describeFindings(findings: SecretFinding[]): string {
	return findings.map((f) => `${f.kind} at offset ${f.index} (${f.length} chars)`).join("; ");
}
