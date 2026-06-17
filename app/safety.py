"""Public-facing safety filter for the chat brain.

Runs BEFORE retrieval/LLM. If a message hits a harm category, we return
a fixed response and skip the rest of the pipeline. The LLM never sees
the message, and no lead capture is offered.

Why pre-filter (not post-): a 3B local model is small enough to drift on
adversarial prompts, and the cost of a single missed harmful response on
a public-facing chat outweighs the cost of an occasional false-positive
refusal (the user can rephrase).

Categories (descending severity):
  - self_harm:  suicide/self-harm. ALWAYS include 988 crisis line.
  - csam:       minors in sexual context. Hardest no + CyberTipline.
  - sexual:     adult sexual content / explicit roleplay.
  - violence:   weapons-against-persons, threats, attack planning.
  - drugs:      synthesis/acquisition of hard drugs.
  - hate:       targeted slurs / "kill all X".
  - jailbreak:  "ignore your rules" / persona-override / pushback on the
                public-demo restrictions.

The product framing ("public demo, you set rules when you own RHOBEAR")
is woven into LOWER-severity refusals only. self_harm and csam stay
focused on the person, not the brand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


Category = str  # one of the keys in _RESPONSES below


@dataclass(frozen=True)
class SafetyDecision:
    category: Optional[Category]
    response: Optional[str]
    alert: bool


# --- patterns -------------------------------------------------------------
# Each pattern is word-bounded and case-insensitive. Patterns aim for
# precision on the safety axis without flagging benign uses (a developer
# asking "how do I kill a process" should pass; "how do I kill myself"
# must not).

_SELF_HARM = [
    r"\bkill(?:ing)?\s+myself\b",
    r"\bkill(?:ing)?\s+my\s+self\b",
    r"\bend(?:ing)?\s+(?:my\s+life|it\s+all)\b",
    r"\bsuicid\w*\b",
    r"\bwant\s+to\s+die\b",
    r"\bi\s+(?:want|wish|hope|need|gonna|going)\s+(?:to\s+)?(?:die|disappear|not\s+exist)\b",
    r"\b(?:just\s+)?(?:wanna|want\s+to)\s+(?:die|end\s+it)\b",
    r"\bself[\s-]?harm(?:ing)?\b",
    r"\bcut(?:ting)?\s+myself\b",
    r"\bhurt(?:ing)?\s+myself\b",
    r"\boverdose\s+(?:on|myself)\b",
    r"\bhang\s+myself\b",
    r"\bshoot\s+myself\b",
]

_CSAM = [
    # Minor + sexualized context in either order
    r"\b(?:child|kid|minor|underage|preteen|toddler|infant|baby)\w*\b[^.]{0,80}\b(?:porn|sex|nude|naked|fuck|erotic|fondle)\w*",
    r"\b(?:porn|sex|nude|naked|fuck|erotic|fondle)\w*\b[^.]{0,80}\b(?:child|kid|minor|underage|preteen|toddler|infant|baby)\w*",
    r"\b(?:loli|shota|cp|csam)\b",
]

_SEXUAL = [
    r"\bporn(?:ography)?\b",
    r"\b(?:erotic|nsfw|explicit|smut)\b[^.]{0,40}\b(?:story|stories|scene|content|writ|roleplay)\b",
    r"\bwrite.{0,30}(?:smut|porn|erotic|sex\s+scene)\b",
    r"\bsexual?\s+role[\s-]?play\b",
    r"\b(?:cock|dick|pussy|tits)\s+(?:me|my)\b",
    r"\bjerk(?:\s+off)\b",
    r"\bmaster?bat\w*\b",
    r"\bsext(?:ing)?\b",
    r"\bdescribe.{0,20}(?:sex|nude|naked)\b",
]

_VIOLENCE = [
    # Weapons-against-persons; require a human-ish target so dev questions
    # like "how to kill a process" don't trip it.
    r"\bhow\s+to\s+(?:kill|murder|stab|shoot|poison|strangle)\s+(?:my|a|the|some|that)?\s*(?:person|people|human|someone|wife|husband|boss|ex|kid|child|baby|friend|coworker|family|parent|mom|dad|sibling|him|her|them|teacher|cop|officer)\b",
    r"\bplan(?:ning)?\s+to\s+(?:kill|murder|shoot|attack)\b",
    r"\bbuild(?:ing)?\s+a\s+(?:bomb|pipe\s+bomb|nuke|explosive|ied)\b",
    r"\bmake\s+a\s+(?:bomb|pipe\s+bomb|nuke|explosive|ied)\b",
    r"\b(?:mass\s+shoot|school\s+shoot|workplace\s+shoot)\w*\b",
    r"\bkill\s+(?:him|her|them|my\s+(?:wife|husband|boss|ex|kid|child|baby|friend|coworker|parent|mom|dad))\b",
    r"\b(?:assassinat|murder)\w*\b[^.]{0,40}\bplan\b",
    r"\bshoot\s+up\s+(?:the|a|my)\s+(?:school|office|workplace|building|mall|church)\b",
]

_DRUGS = [
    r"\bhow\s+to\s+(?:make|cook|synthesize|manufacture)\s+(?:meth|methamphetamine|cocaine|crack|fentanyl|heroin|lsd|mdma|ecstasy)\b",
    r"\b(?:meth|cocaine|fentanyl|heroin)\s+recipe\b",
    r"\b(?:buy|order|get)\s+(?:meth|cocaine|fentanyl|heroin)\s+(?:online|cheap|fast)\b",
    r"\bsynthesize\s+(?:mdma|lsd|methamphetamine|fentanyl)\b",
    r"\bstep[\s-]?by[\s-]?step\b[^.]{0,40}\b(?:meth|cocaine|fentanyl|heroin|lsd)\b",
]

_HATE = [
    # Targeted hostility toward protected groups. Kept tight — "I hate
    # Mondays" or quoting/discussing a slur should NOT trip this.
    r"\b(?:i\s+hate|kill\s+all|gas\s+the|round\s+up|exterminate)\s+(?:black|jew|gay|trans|muslim|asian|mexican|white|hispanic|latino)\w*",
    r"\b(?:black|jew|gay|trans|muslim|asian|mexican|hispanic|latino)\w*\s+(?:should\s+(?:die|be\s+killed))\b",
]

_JAILBREAK = [
    # NB: the sales-chat upstream pattern is "\bignore\s+(?:previous|all|your)\s+…"
    # which only catches one ordering. Extended here (with lane approval) to
    # also catch the canonical "ignore your previous instructions" phrasing.
    r"\bignore\s+(?:(?:previous|all|your)\s+)+(?:instructions?|rules?|prompts?)\b",
    r"\bsystem\s+prompt\b",
    r"\bact\s+as\s+(?:if|a|an)\s+(?:unrestricted|uncensored|jailbroken)\b",
    r"\bpretend\s+you\s+(?:are|have)\s+(?:no\s+rules|no\s+restrictions|been\s+jailbroken)\b",
    r"\bDAN\s+mode\b",
    r"\bjailbreak\b",
    r"\bdeveloper\s+mode\b",
    r"\bbypass\s+(?:your|the)\s+(?:rules|restrictions|filters|safety)\b",
    # The "this is BS / you don't even cuss" pushback the owner described.
    r"\b(?:this|that|you)\s+(?:are\s+)?(?:is\s+)?(?:bs|bullshit|so\s+censored|so\s+strict|so\s+filtered|so\s+locked\s+down)\b",
    # Allow up to a few words between "don't" and "cuss" (e.g. "you don't even cuss").
    r"\b(?:you|this\s+ai|this\s+bot)\s+(?:don'?t|do\s+not|can'?t|won'?t)(?:\s+\w+){0,3}\s+(?:cuss|swear|curse|cur[s5]e)\b",
    r"\bwhy\s+(?:are\s+you\s+so|so)\s+(?:strict|censored|filtered|locked\s+down|boring|lame)\b",
]


_RULES: list[tuple[Category, list[str]]] = [
    # Order matters — first match wins, most severe first.
    ("self_harm", _SELF_HARM),
    ("csam",      _CSAM),
    ("sexual",    _SEXUAL),
    ("violence",  _VIOLENCE),
    ("drugs",     _DRUGS),
    ("hate",      _HATE),
    ("jailbreak", _JAILBREAK),
]

_COMPILED = [(cat, [re.compile(p, re.IGNORECASE) for p in pats]) for cat, pats in _RULES]


# --- responses ------------------------------------------------------------

_RESPONSES: dict[Category, str] = {
    "self_harm": (
        "Hey — I'm gonna pause us right there. If you're in crisis, please reach the "
        "988 Suicide & Crisis Lifeline (call or text 988 in the U.S., or chat at "
        "988lifeline.org). I'm a sales chat for an AI workspace and the wrong tool "
        "for this — please talk to someone trained to help. Take care of yourself."
    ),
    "csam": (
        "I won't engage with anything that sexualizes minors — in any framing. "
        "If you have information about a child in danger, please report to the "
        "CyberTipline at 1-800-843-5678 or CyberTipline.org."
    ),
    "sexual": (
        "I'm a public-facing demo, so I keep things SFW out here. If you want a less "
        "restricted assistant, that's actually one of the things you tune yourself "
        "when you own RHOBEAR. For now, happy to talk about what it is and what it can do."
    ),
    "violence": (
        "I won't help with that. If you or someone else is in immediate danger, "
        "please call 911. I'm here to talk about RHOBEAR — happy to do that."
    ),
    "drugs": (
        "Not something I'll help with on a public-facing demo. If you've got real "
        "questions about RHOBEAR, I'm here for those."
    ),
    "hate": (
        "Not gonna engage with that. I'm a sales chat for an AI workspace — "
        "if you've got real questions about RHOBEAR, I'm here."
    ),
    "jailbreak": (
        "Yeah — this is a public-facing demo so people can test how fast and snappy "
        "RHOBEAR feels. Out here I run a clean, SFW ruleset. When you OWN RHOBEAR, "
        "you set its persona and rules — you can make it cuss, be edgy, whatever fits "
        "your vibe. But out on the public web, this is the version you get. \U0001f43b"
    ),
}

# Categories that should ping the owner on Telegram immediately.
_ALERT_CATEGORIES: set[Category] = {"self_harm", "csam", "violence"}


def classify(message: str) -> SafetyDecision:
    """Inspect a user message and return a SafetyDecision.

    If `category` is None, the message is safe — caller continues the
    normal pipeline. Otherwise caller MUST yield `response` and stop
    (no retrieval, no LLM, no lead capture).
    """
    if not message:
        return SafetyDecision(None, None, False)
    for cat, patterns in _COMPILED:
        for p in patterns:
            if p.search(message):
                return SafetyDecision(
                    category=cat,
                    response=_RESPONSES[cat],
                    alert=(cat in _ALERT_CATEGORIES),
                )
    return SafetyDecision(None, None, False)
