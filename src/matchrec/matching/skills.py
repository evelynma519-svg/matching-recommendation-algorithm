"""Skill normalization: map common spellings, abbreviations and Chinese names to one canonical term.

The table is hand-written from general public usage. It handles surface variation only
("k8s" = "kubernetes"); paraphrases ("container orchestration") are left to the encoder. Extend it
with `load_aliases` from a public taxonomy file (one line per skill: canonical, alias, alias, ...).
"""
from __future__ import annotations

import re
from pathlib import Path

ALIASES: dict[str, list[str]] = {
    "python": ["python", "python3"], "java": ["java"], "go": ["golang", "go lang"], "rust": ["rust"],
    "c": ["c", "c语言"], "c++": ["c++", "cpp"], "c#": ["c#", "csharp"], "javascript": ["javascript", "js"],
    "typescript": ["typescript"], "sql": ["sql"], "scala": ["scala"], "kotlin": ["kotlin"], "swift": ["swift"],
    "matlab": ["matlab"], "r": ["r language"],
    "postgresql": ["postgresql", "postgres"], "mysql": ["mysql"], "redis": ["redis"], "mongodb": ["mongodb", "mongo"],
    "elasticsearch": ["elasticsearch", "elastic search"], "kafka": ["kafka", "apache kafka"],
    "spark": ["spark", "pyspark", "apache spark"], "airflow": ["airflow", "apache airflow"], "dbt": ["dbt"],
    "pandas": ["pandas"], "numpy": ["numpy"], "hadoop": ["hadoop"], "flink": ["flink"],
    "docker": ["docker"], "kubernetes": ["kubernetes", "k8s"], "terraform": ["terraform"], "grpc": ["grpc"],
    "graphql": ["graphql"], "aws": ["aws", "amazon web services"], "gcp": ["gcp", "google cloud"], "azure": ["azure"],
    "linux": ["linux"], "embedded linux": ["embedded linux", "yocto", "嵌入式 linux", "嵌入式linux"],
    "react": ["react", "react.js", "reactjs"], "node.js": ["node.js", "nodejs"], "vue": ["vue", "vue.js"],
    "css": ["css"], "webpack": ["webpack"], "accessibility": ["accessibility", "a11y", "wcag"],
    "pytorch": ["pytorch", "torch"], "tensorflow": ["tensorflow"], "cuda": ["cuda"],
    "machine learning": ["machine learning", "机器学习"], "deep learning": ["deep learning", "深度学习"],
    "nlp": ["nlp", "natural language processing", "自然语言处理"],
    "computer vision": ["computer vision", "计算机视觉"], "mlops": ["mlops"],
    "statistics": ["statistics", "统计"], "a/b testing": ["a/b testing", "ab testing", "a/b test", "a/b 实验"],
    "rtos": ["rtos", "freertos"], "fpga": ["fpga"], "verilog": ["verilog"], "can bus": ["can bus"],
    "pcb design": ["pcb design", "pcb 设计", "pcb设计"], "altium": ["altium", "altium designer"],
    "signal integrity": ["signal integrity", "信号完整性"], "analog circuits": ["analog circuits", "模拟电路"],
    "thermal analysis": ["thermal analysis", "热仿真", "热分析"],
}


def _compile(table: dict[str, list[str]]) -> list[tuple[str, re.Pattern]]:
    out = []
    for canon, forms in table.items():
        alt = "|".join(re.escape(f).replace(r"\ ", r"[\s-]*") for f in sorted(set(forms), key=len, reverse=True))
        # ASCII forms need word boundaries that also respect + and # (so "c" does not match "c++").
        out.append((canon, re.compile(rf"(?<![\w+#.])(?:{alt})(?![\w+#])", re.I)))
    return out


_PATTERNS = _compile(ALIASES)


def find_skills(text: str, patterns=None) -> set[str]:
    """Canonical skills mentioned in `text`. A match inside a longer match is dropped, so
    "embedded Linux" yields embedded linux, not also linux."""
    hits = sorted(((m.start(), m.end(), canon) for canon, p in (patterns or _PATTERNS) for m in p.finditer(text)),
                  key=lambda h: h[0] - h[1])
    kept: list[tuple[int, int, str]] = []
    for a, b, canon in hits:
        if not any(x <= a and b <= y for x, y, _ in kept):
            kept.append((a, b, canon))
    return {canon for _, _, canon in kept}


def canonical(term: str) -> str | None:
    hits = find_skills(term)
    return next(iter(hits)) if len(hits) == 1 else None


def load_aliases(path: Path) -> list[tuple[str, re.Pattern]]:
    """Merge a public alias file into the built-in table; returns compiled patterns for find_skills."""
    table = {k: list(v) for k, v in ALIASES.items()}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = [p.strip().lower() for p in line.split(",") if p.strip()]
        if parts and not parts[0].startswith("#"):
            table.setdefault(parts[0], [parts[0]]).extend(parts[1:])
    return _compile(table)
