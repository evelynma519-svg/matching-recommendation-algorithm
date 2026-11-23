"""Person-name detectors. The ladder decides; a detector only proposes candidate spans.

`PersonDetector` is the seam. Shipped implementations:

- `SpacyDetector`  local spaCy pipelines (English and/or Chinese); recommended for real text.
- `HeuristicDetector`  capitalized word runs; dependency-free, for demos and tests. It over-proposes
  on purpose: the ladder's allow layers are what keep technical words readable.

Everything runs locally. No text is sent to any service.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .lexicons import ZH_COMMON_SURNAMES
from .spans import HAN


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    ambiguous: bool = False         # detector is unsure; a mask becomes [AMBIG] instead of [PERSON]


class PersonDetector(Protocol):
    name: str

    def detect(self, text: str) -> list[Candidate]: ...


class DetectorUnavailable(RuntimeError):
    pass


_CAP = r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?"
_RUN = re.compile(rf"(?<![\w.]){_CAP}(?:[ \t]+(?:[A-Z]\.[ \t]*)?{_CAP}){{0,2}}(?![\w])")
_SENTENCE_START = re.compile(r"(?:^|[\n.!?:•●▪\-|]\s*)$")
_SURNAMES = "".join(sorted(ZH_COMMON_SURNAMES))
_ZH_NAME_CUE = re.compile(rf"(?:与|和|同|跟|向|由|导师|经理|同事|推荐人|联系人)[:：]?\s*([{_SURNAMES}][\u4e00-\u9fff]{{1,2}}?)"
                          rf"(?=合作|共同|一起|负责|指导|协作|完成|的|，|。|、|\s|$)")


class HeuristicDetector:
    """Runs of capitalized words. Multi-word runs are proposed as names; a single capitalized word is
    proposed as ambiguous unless it starts a sentence or line. For Chinese, a common surname plus one or
    two characters is proposed when it sits between relational cue words (与…合作, 导师…)."""

    name = "heuristic"

    def detect(self, text: str) -> list[Candidate]:
        out = []
        for m in _RUN.finditer(text):
            words = m.group().split()
            if len(words) >= 2:
                out.append(Candidate(m.start(), m.end()))
            elif not _SENTENCE_START.search(text[max(0, m.start() - 3):m.start()]):
                out.append(Candidate(m.start(), m.end(), ambiguous=True))
        # Chinese names have no capitals; propose surname + 1-2 characters between relational cue words.
        out += [Candidate(m.start(1), m.end(1)) for m in _ZH_NAME_CUE.finditer(text)]
        return out


class SpacyDetector:
    """spaCy NER, local models only. Picks the model per text by script; mixed text runs both."""

    name = "spacy"

    def __init__(self, en_model: str = "en_core_web_sm", zh_model: str | None = "zh_core_web_sm"):
        try:
            import spacy
        except ImportError as e:
            raise DetectorUnavailable("pip install spacy and a model, e.g. python -m spacy download en_core_web_sm") from e
        self.en = spacy.load(en_model, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
        self.zh = None
        if zh_model:
            try:
                self.zh = spacy.load(zh_model, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
            except OSError:
                self.zh = None
        self.name = f"spacy:{en_model}" + (f"+{zh_model}" if self.zh else "")

    def detect(self, text: str) -> list[Candidate]:
        models = [self.en]
        if self.zh is not None and HAN.search(text):
            models.append(self.zh)
        out = []
        for nlp in models:
            for e in nlp(text).ents:
                if e.label_ == "PERSON":
                    single = len(e.text.split()) == 1 and not HAN.search(e.text)
                    out.append(Candidate(e.start_char, e.end_char, ambiguous=single))
        return out
