"""Text normalization that keeps a 1:1 view of the source for span bookkeeping."""
from __future__ import annotations

import html
import re
import unicodedata
from html.parser import HTMLParser

HTML_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th", "table",
             "tbody", "thead", "span", "a", "b", "strong", "em", "i", "u", "html", "body", "head", "section",
             "font", "hr", "img", "sub", "sup", "code", "pre", "style", "script"}
_BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}
_TAG = re.compile(r"</?(?:" + "|".join(sorted(HTML_TAGS)) + r")\b[^>]*>", re.I)


class _Plain(HTMLParser):
    """HTML to text. Known layout tags become line breaks; unknown tags are kept as literal text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        elif tag not in HTML_TAGS:
            self.parts.append(self.get_starttag_text() or "")
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        elif tag not in HTML_TAGS:
            self.parts.append(f"</{tag}>")
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def plain(value) -> str:
    """NFKC, HTML stripped, whitespace collapsed. Run once, before any span is computed."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).replace("\x00", "")
    if _TAG.search(text):
        p = _Plain()
        p.feed(text)
        text = "".join(p.parts)
    text = html.unescape(text)
    text = re.sub(r"[\t  ]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip().casefold()


_GLUED = re.compile(r"(?<=[a-z]{2})\.(?=[A-Z][a-z])")


def split_glued_sentences(text: str, keep) -> tuple[str, int]:
    """PDF extraction often glues sentences ("built APIs.Led a team"), which makes a capitalized
    word look like the start of a name. Insert a space after the period, unless the whole token is
    a known technical term (`keep(token)` is True), e.g. "Node.Js" style casing."""
    count = 0

    def fix(m):
        nonlocal count
        a, b = m.start(), m.end()
        while a > 0 and re.match(r"[A-Za-z0-9_+#.]", text[a - 1]):
            a -= 1
        while b < len(text) and re.match(r"[A-Za-z0-9_+#.]", text[b]):
            b += 1
        if keep(norm(text[a:b])):
            return m.group()
        count += 1
        return ". "

    return _GLUED.sub(fix, text), count


def window(text: str, a: int, b: int, width: int) -> str:
    """Text within `width` characters of [a, b), widened to whole words at both ends. A window cut
    mid-word would turn "the" into "he" and fake a person cue."""
    lo, hi = max(0, a - width), min(len(text), b + width)
    while lo > 0 and text[lo - 1].isalnum():
        lo -= 1
    while hi < len(text) and text[hi].isalnum():
        hi += 1
    return text[lo:hi]
