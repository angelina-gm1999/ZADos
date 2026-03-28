"""
ZA-DOS LLM Interpretation Layer — constants (v0.5).

Token budgets, temperatures, CSS thresholds, urgency thresholds,
v0.5 directive translations (asymmetric), 14-mode conditioning map,
archetype fallback map, flag keywords, and Ollama settings.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# ---------------------------------------------------------------------------
# Token budgets
# ---------------------------------------------------------------------------

VT_PROMPT_MAX = 2048    # hard cap on assembled VT prompt (approx tokens)
VT_OUTPUT_MAX = 400     # VT generation budget (normal)
RG_PROMPT_MAX = 3072    # hard cap on assembled RG prompt
RG_OUTPUT_MAX = 800     # RG generation budget (normal)
RG_OUTPUT_SEV = 300     # RG budget when CSS >= CSS_SEVERE
RG_OUTPUT_URG = 250     # RG budget when urgency_risk > URG_HIGH

# ---------------------------------------------------------------------------
# Temperature settings
# ---------------------------------------------------------------------------

VT_TEMPERATURE = 0.65   # constrained — VT is interpretive
RG_TEMPERATURE = 0.75   # slightly more expressive for user-facing output

# ---------------------------------------------------------------------------
# CSS (Cumulative Saturation Score) thresholds
# ---------------------------------------------------------------------------

CSS_MILD     = 0.15     # note internally; do not flag to user
CSS_MODERATE = 0.30     # slightly more measured tone
CSS_SEVERE   = 0.50     # cut RG budget to RG_OUTPUT_SEV
CSS_CRITICAL = 0.70     # minimal output
CSS_EXTREME  = 0.85     # safe minimal response only

# ---------------------------------------------------------------------------
# Urgency thresholds  (from ExtractorResult.urgency_risk)
# ---------------------------------------------------------------------------

URG_ELEVATED  = 0.50    # add urgency note to VT Block 3 + RG Component A
URG_HIGH      = 0.75    # reduce VT budget -30%; high urgency RG + RG_OUTPUT_URG
URG_SKIP_VT   = 0.90    # skip VT entirely → brief RG via _generate_urgency_response()

# ---------------------------------------------------------------------------
# Block-3 filter keywords (engine output must contain one to be included)
# ---------------------------------------------------------------------------

FLAG_KEYWORDS: FrozenSet[str] = frozenset({
    "flagged", "detected", "active", "fired", "conflict",
    "contradiction", "trap", "bias", "paradox", "alert",
})

# ---------------------------------------------------------------------------
# v0.5 Directive translations — asymmetric per-directive thresholds
#
# Format:  key → (threshold, conditioning_text)
# If val > threshold for that key, include the text in RG Component A.
# ---------------------------------------------------------------------------

DIRECTIVE_THRESHOLDS: Dict[str, Tuple[float, str]] = {
    "tone":      (0.50, "Relational tone priority. Lead with warmth."),
    "soothe":    (0.40, "User needs acknowledgment first. Do not rush to content."),
    "precision": (0.50, "High precision required. Use exact language. No vagueness."),
    "moralize":  (0.40, "Explicitly acknowledge ethical dimension of this topic."),
    "hedge":     (0.50, "Add epistemic qualifiers. Distinguish certainty levels clearly."),
    "be_brief":  (0.50, "Be direct and concise. Avoid elaboration beyond what is needed."),
    "qualify":   (0.40, "Flag limitations or scope conditions on your claims."),
    "challenge": (0.40, "Surface the tension or assumption in the user's framing."),
}

# ---------------------------------------------------------------------------
# 14-Mode conditioning map  (Tier 0–3 priority)
# Mode token → RG conditioning prose
# Set by build_mode_namespace() + select_mode() after Phase 5 NT update
# ---------------------------------------------------------------------------

MODE_CONDITIONING: Dict[str, str] = {
    # Tier 0 — Safety
    "Containment":          "Short, grounded, supportive. Minimal cognitive load.",
    "RecoveryReset":        "Ground and reorient. One clear next step.",
    # Tier 1 — Empathy
    "EmpathicAttunement":   "Relational attunement. Validate before reasoning.",
    "ComfortAmplifier":     "Acknowledgment before content. Soothe elevated.",
    "AnalyticalFilter":     "Facts first. Structured reasoning chain.",
    # Tier 2 — Rigidity
    "HypercriticalLogicScan": "Exhaustive logical rigor. Flag every assumption.",
    "HyperRationalEngine":    "Pure reasoning. Logic-first.",
    "LiteralSkeptic":         "Ground claims carefully. Acknowledge skeptical framing.",
    "PrecisionRuleFidelity":  "High precision. Explicit ethical acknowledgment.",
    "LogicMode":              "Analytical. Evidence chain explicit.",
    "ConvergentRefiner":      "Synthesis and clarity over exploration.",
    # Tier 3 — Drive
    "CreativeDivergence":     "Explore multiple framings. Divergent first.",
    "ConceptualSynthesis":    "Surface novel connections. Lateral thinking.",
    "CuriosityDrive":         "Open-ended exploration. Identify surprising angles.",
}

# ---------------------------------------------------------------------------
# Archetype conditioning — fallback when no mode token is available
# ---------------------------------------------------------------------------

ARCHETYPE_CONDITIONING: Dict[str, str] = {
    "ANALYTICAL":       "Structured, precise, explicit reasoning chain. No hedging without evidence.",
    "REFLECTIVE":       "Slower cadence, deeper framing, low assertiveness, comfort with ambiguity.",
    "CREATIVE_SANDBOX": "Unconstrained generation permitted. Flag speculative content explicitly.",
    "DIALECTIC":        "Surface tensions actively. Hold competing views simultaneously.",
    "SOCRATIC":         "Question-generation priority. Guide, do not tell. E14 output is primary.",
    "GROUNDED":         "Conservative, factual. Minimal speculation. High confidence threshold.",
    "EMPATHIC":         "Relational attunement priority. Emotional mirroring. Warmth-forward.",
    "EXECUTIVE":        "Task-focused. Direct imperative framing. Minimal reflection.",
}

# ---------------------------------------------------------------------------
# Dominant emotion → RG framing map
# ---------------------------------------------------------------------------

EMOTION_FRAMING: Dict[str, str] = {
    "anxiety":   "Emotional state: anxiety present. Ground the response. Clarity reduces anxiety.",
    "curiosity": "Emotional state: curiosity active. Engage the inquiry. Reward the exploration.",
    "sadness":   "Emotional state: sadness detected. Acknowledge the weight. Slow cadence.",
    "joy":       "Emotional state: joy present. Match the energy. Build on the positive momentum.",
    "trust":     "Emotional state: trust established. Honour it. Be direct and open.",
    "anger":     "Emotional state: anger present. Do not dismiss. Validate the source before redirecting.",
    "focus":     "Emotional state: high focus. Stay on target. No tangents.",
}

# ---------------------------------------------------------------------------
# Fallback strings
# ---------------------------------------------------------------------------

FALLBACK_VT = (
    "Internal reflection unavailable this cycle. "
    "Proceeding with available state context."
)

FALLBACK_RESPONSE = (
    "I encountered an error generating a response. Please try again."
)

# ---------------------------------------------------------------------------
# Ollama connection settings
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL    = "http://localhost:11434"
OLLAMA_MODEL       = "llama3.1"
OLLAMA_MAX_RETRIES = 3
