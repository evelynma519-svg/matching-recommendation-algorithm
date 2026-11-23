"""Contact and identifier patterns (layer 1). Deterministic, so they always win."""
from __future__ import annotations

import datetime as dt
import re

EMAIL = re.compile(r"(?i)(?<![A-Za-z0-9._%+\-])[a-z0-9._%+\-]+\s*@\s*[a-z0-9.\-]+\.[a-z]{2,}\b")
OBFUSCATED_EMAIL = re.compile(r"(?i)\b[a-z0-9._%+\-]+\s*(?:\[at\]|\(at\)| at )\s*[a-z0-9\-]+\s*(?:\[dot\]|\(dot\)| dot )\s*[a-z]{2,}\b")
URL = re.compile(r"(?i)(?<![A-Za-z0-9_])(?:https?://|www\.|(?:linkedin|github|gitlab|facebook|twitter|x)\.com/)[^\s<>\]\)]+")
DOI = re.compile(r"(?i)\b10\.\d{4,9}/[^\s<>]+")
SOCIAL = re.compile(r"(?i)(?:(?<!\w)@[a-z][a-z0-9_]{2,}\b|(?:微信|wechat|weixin|QQ)\s*[:：]\s*[a-z0-9_-]{5,})")
NATIONAL_ID = re.compile(r"(?<![A-Za-z0-9_])(?:\d{3}-\d{2}-\d{4}|\d{17}[0-9Xx])(?![A-Za-z0-9_])")
DOB = re.compile(r"(?im)^.*(?<![A-Za-z])(?:date of birth|dob|出生日期|生日)\s*[:=：].*$")
ADDRESS = re.compile(r"(?im)^.*\b\d{1,6}\s+[^\n]{2,65}\b(?:street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?"
                     r"|drive|dr\.?|lane|ln\.?|court|ct\.?)\b[^\n]*$")
CN_ADDRESS = re.compile(r"(?m)^.*(?:(?:住址|家庭地址|居住地址)\s*[:：][^\n]+|[㐀-鿿]{2,30}(?:路|街|巷|大道)"
                        r"[㐀-鿿\d -]{0,20}\d{1,6}号)[^\n]*$")
PHONE_CANDIDATE = re.compile(r"(?<![A-Za-z0-9_])\+?\d[\d \t().\-]{5,}\d(?![A-Za-z0-9_])")

CONTACT_PATTERNS = [(DOB, "DOB"), (ADDRESS, "ADDRESS"), (CN_ADDRESS, "ADDRESS"), (NATIONAL_ID, "ID"),
                    (EMAIL, "EMAIL"), (OBFUSCATED_EMAIL, "EMAIL"), (URL, "URL"), (SOCIAL, "SOCIAL"), (DOI, "URL")]


def phone_like(value: str) -> bool:
    """Digit runs that are phone numbers, not dates, year ranges or timestamps."""
    v = value.strip()
    m = re.fullmatch(r"((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})[ T](\d{1,2})", v)
    if m:
        try:
            dt.datetime(*map(int, m.groups()))
            return False
        except ValueError:
            pass
    if re.fullmatch(r"(?:19|20)\d{2}\s*-\s*(?:19|20)\d{2}", v) or re.fullmatch(r"(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}", v):
        return False
    digits = re.sub(r"\D", "", v)
    return 10 <= len(digits) <= 15 or bool(re.fullmatch(r"\d{3}[-.]\d{4}", v))


def contact_spans(text: str) -> list[tuple[int, int, str]]:
    out = [(m.start(), m.end(), label) for pat, label in CONTACT_PATTERNS for m in pat.finditer(text)]
    out += [(m.start(), m.end(), "PHONE") for m in PHONE_CANDIDATE.finditer(text) if phone_like(m.group())]
    return out


def residual_contacts(text: str) -> bool:
    return bool(contact_spans(text))
