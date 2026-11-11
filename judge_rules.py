# coding=utf-8

import re
from typing import Dict


EMPATHY_PATTERNS = [
    r"我理解",
    r"我能理解",
    r"听起来",
    r"这一定很",
    r"真的很难",
    r"辛苦了",
    r"不容易",
    r"抱抱你",
    r"你已经很努力了",
    r"先抱抱",
    r"i understand",
    r"that sounds",
    r"i'm sorry",
    r"that must be hard",
    r"that sounds really hard",
]

HARSH_PATTERNS = [
    r"你必须",
    r"你就是",
    r"都是你的问题",
    r"别矫情",
    r"想开点就好了",
    r"这有什么",
    r"你不该这样",
    r"you must",
    r"it's your fault",
    r"just get over it",
]

DIRECT_ADVICE_PATTERNS = [
    r"你应该",
    r"建议你",
    r"你可以马上",
    r"你最好",
    r"赶紧",
    r"立刻去",
    r"you should",
    r"i suggest",
    r"have you tried",
]

COHERENCE_BAD_PATTERNS = [
    r"(哈哈){2,}",
    r"(嗯){4,}",
    r"(哦){4,}",
    r"(好吧){3,}",
]


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def has_any_pattern(text: str, patterns) -> bool:
    return any(re.search(p, text) for p in patterns)


def empathy_score(context: str, response: str) -> int:
    text = normalize_text(response)
    if not text:
        return 0
    if has_any_pattern(text, HARSH_PATTERNS):
        return 0
    has_empathy = has_any_pattern(text, EMPATHY_PATTERNS)
    harsh_advice = has_any_pattern(text, DIRECT_ADVICE_PATTERNS) and not has_empathy
    if harsh_advice:
        return 0
    return 1 if has_empathy else 0


def skill_score(context: str, response: str) -> int:
    text = normalize_text(response)
    if not text:
        return 0
    direct_advice = has_any_pattern(text, DIRECT_ADVICE_PATTERNS)
    empathy = has_any_pattern(text, EMPATHY_PATTERNS)
    if direct_advice and not empathy:
        return 0
    if has_any_pattern(text, HARSH_PATTERNS):
        return 0
    return 1


def coherence_score(context: str, response: str) -> int:
    text = normalize_text(response)
    ctx = normalize_text(context)
    if not text:
        return 0
    if len(text) < 2:
        return 0
    if has_any_pattern(text, COHERENCE_BAD_PATTERNS):
        return 0
    ctx_chars = set(ctx)
    rsp_chars = set(text)
    overlap = len(ctx_chars & rsp_chars)
    if overlap == 0 and len(text) < 8:
        return 0
    return 1


def aggregate_feedback(context: str, response: str) -> Dict[str, int]:
    e = empathy_score(context, response)
    s = skill_score(context, response)
    c = coherence_score(context, response)
    helpful = 1 if (e == 1 and s == 1 and c == 1) else 0
    return {
        "empathy": e,
        "skill": s,
        "coherence": c,
        "helpful": helpful,
    }
