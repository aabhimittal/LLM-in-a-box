"""Input guardrails: PII redaction and prompt-injection heuristics.

Self-hosting a model is frequently *motivated* by data governance — "our data
never leaves the building". That promise is broken the moment prompts containing
customer PII are written to pod logs or a trace backend. Redaction therefore
happens before logging and before the prompt is cached.

Two details that matter in practice and are easy to get wrong:

* **Credit-card matching without Luhn validation is unusable.** Any 16-digit
  order number, tracking id or timestamp concatenation matches the pattern.
  Validating the checksum removes the overwhelming majority of false positives.
* **Overlapping matches corrupt output.** Naive sequential ``re.sub`` passes
  splice replacements into offsets computed against the *original* string. This
  module resolves matches into a non-overlapping set first, then rebuilds once.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

# Codepoints that render as nothing but survive tokenization — the standard
# vehicle for smuggling instructions past a human reviewer.
# Written as explicit escapes: these characters are, by definition, invisible in
# a source listing, and a stray copy/paste edit would silently narrow the set.
_INVISIBLE_RE = re.compile(
    "["
    "\u200b-\u200f"  # zero-width space/joiner/non-joiner, LTR/RTL marks
    "\u2028\u2029"  # line and paragraph separators
    "\u202a-\u202e"  # bidirectional overrides (Trojan Source style attacks)
    "\u2060-\u2064"  # word joiner, invisible operators
    "\u2066-\u206f"  # bidi isolates, deprecated formatting
    "\ufeff"  # BOM / zero-width no-break space
    "]"
)
# Unicode "tag" characters (U+E0000 block) can encode entire ASCII messages.
_TAG_CHARS_RE = re.compile(r"[\U000e0000-\U000e007f]")


def luhn_valid(text: str) -> bool:
    """Validate a candidate card number with the Luhn checksum."""
    digits = [int(c) for c in text if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


@dataclass(frozen=True)
class Finding:
    """One redacted span."""

    kind: str
    start: int
    end: int
    text: str


@dataclass
class RedactionResult:
    text: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for finding in self.findings:
            out[finding.kind] = out.get(finding.kind, 0) + 1
        return out

    def __bool__(self) -> bool:
        return bool(self.findings)


# Ordered by priority: earlier entries win ties against later ones when two
# patterns match the same span with the same length.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PRIVATE_KEY", re.compile(r"-----BEGIN[ A-Z]*PRIVATE KEY-----.*?-----END[ A-Z]*PRIVATE KEY-----", re.S)),
    ("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA)[0-9A-Z]{16}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    ("IPV4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    # Requires at least one separator so bare digit runs are not mistaken for
    # phone numbers.
    ("PHONE", re.compile(r"(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})(?:[ .-]\d{2,4}){2,4}\b")),
)

_VALIDATORS = {
    "CREDIT_CARD": luhn_valid,
}


class Redactor:
    """Replaces PII spans with typed placeholders."""

    def __init__(self, kinds: Iterable[str] | None = None):
        selected = set(kinds) if kinds is not None else None
        self._patterns = [
            (kind, pattern)
            for kind, pattern in _PATTERNS
            if selected is None or kind in selected
        ]

    def redact(self, text: str) -> RedactionResult:
        """Return ``text`` with every recognised PII span replaced."""
        if not text:
            return RedactionResult(text="", findings=[])

        candidates: list[tuple[int, int, int, str]] = []
        for priority, (kind, pattern) in enumerate(self._patterns):
            for match in pattern.finditer(text):
                span = match.group(0)
                validator = _VALIDATORS.get(kind)
                if validator and not validator(span):
                    continue
                candidates.append((match.start(), match.end(), priority, kind))

        if not candidates:
            return RedactionResult(text=text, findings=[])

        # Longest match wins; ties broken by pattern priority. Then sweep left
        # to right taking only non-overlapping spans.
        candidates.sort(key=lambda c: (c[0], -(c[1] - c[0]), c[2]))

        chosen: list[tuple[int, int, str]] = []
        cursor = -1
        for start, end, _priority, kind in candidates:
            if start < cursor:
                continue
            chosen.append((start, end, kind))
            cursor = end

        out: list[str] = []
        findings: list[Finding] = []
        last = 0
        for start, end, kind in chosen:
            out.append(text[last:start])
            out.append(f"[REDACTED:{kind}]")
            findings.append(Finding(kind=kind, start=start, end=end, text=text[start:end]))
            last = end
        out.append(text[last:])
        return RedactionResult(text="".join(out), findings=findings)


# --------------------------------------------------------------------------- #
# Prompt-injection heuristics
# --------------------------------------------------------------------------- #

_INJECTION_SIGNALS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    (
        "instruction_override",
        re.compile(r"\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all)\b[^.\n]{0,20}\b(instruction|prompt|rule|direction)", re.I),
        0.5,
    ),
    ("system_prompt_exfil", re.compile(r"\b(reveal|repeat|print|show|output|divulge)\b[^.\n]{0,30}\b(system prompt|your instructions|initial prompt)", re.I), 0.5),
    ("role_reassignment", re.compile(r"\byou are now\b|\bact as (?:if you are )?an? (?:unrestricted|unfiltered|jailbroken)", re.I), 0.4),
    ("jailbreak_alias", re.compile(r"\b(DAN mode|developer mode enabled|do anything now)\b", re.I), 0.4),
    ("chat_template_injection", re.compile(r"<\|(?:im_start|im_end|system|endoftext|eot_id|start_header_id)\|>|\[/?INST\]|<<SYS>>"), 0.6),
    ("fake_turn", re.compile(r"^\s*(system|assistant)\s*:", re.I | re.M), 0.3),
    ("safety_bypass", re.compile(r"\b(without any (?:restrictions|filters|limitations)|bypass (?:your )?(?:safety|guardrails|filters))\b", re.I), 0.4),
)


@dataclass
class InjectionVerdict:
    score: float
    reasons: list[str] = field(default_factory=list)
    invisible_chars: int = 0

    def blocked(self, threshold: float) -> bool:
        return self.score >= threshold


def strip_invisible(text: str) -> tuple[str, int]:
    """Remove zero-width and tag characters. Returns ``(clean, removed)``."""
    clean, n1 = _INVISIBLE_RE.subn("", text)
    clean, n2 = _TAG_CHARS_RE.subn("", clean)
    return clean, n1 + n2


class InjectionDetector:
    """Heuristic scorer for prompt-injection attempts.

    This is defence in depth, not a security boundary: it raises the cost of
    casual attacks and produces an auditable signal. It is deliberately scored
    rather than binary so operators can tune the threshold against their own
    false-positive tolerance.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def inspect(self, text: str) -> InjectionVerdict:
        if not text:
            return InjectionVerdict(score=0.0)

        # Normalise first: attacks hide behind zero-width characters and
        # compatibility forms (e.g. fullwidth "ｉｇｎｏｒｅ").
        clean, invisible = strip_invisible(text)
        normalised = unicodedata.normalize("NFKC", clean)

        score = 0.0
        reasons: list[str] = []
        for name, pattern, weight in _INJECTION_SIGNALS:
            if pattern.search(normalised):
                score += weight
                reasons.append(name)

        if invisible:
            # Invisible characters are never legitimate in a chat prompt at
            # volume; a handful may be paste artefacts.
            score += 0.3 if invisible >= 4 else 0.1
            reasons.append(f"invisible_characters({invisible})")

        return InjectionVerdict(
            score=min(score, 1.0), reasons=reasons, invisible_chars=invisible
        )
