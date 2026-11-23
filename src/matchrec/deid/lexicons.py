"""Public lexicons used by the allow layers.

Every list here is hand-written from general, publicly known vocabulary: programming languages,
frameworks, tools, degree and title abbreviations, month names, and common resume headings. No term
was derived from any private dataset. To widen coverage, load an additional public vocabulary
(for example a skills taxonomy or a list of technology tags) with `load_terms`; keep the source and
license of any such file next to it.
"""
from __future__ import annotations

import re
from pathlib import Path

from .text import norm

TECH = frozenset("""
python java javascript typescript go golang rust ruby php perl scala kotlin swift objective-c c c++ c# f# r julia
matlab octave fortran cobol haskell elixir erlang clojure lua dart groovy bash shell powershell sql plsql t-sql
html css sass less graphql json yaml xml protobuf grpc rest soap websocket
react angular vue svelte next.js node.js express django flask fastapi spring rails laravel asp.net .net
jquery redux webpack vite babel tailwind bootstrap
mysql postgresql postgres sqlite oracle mongodb redis cassandra dynamodb elasticsearch opensearch neo4j
clickhouse snowflake bigquery redshift databricks hive presto trino
spark hadoop kafka flink airflow dbt beam pandas numpy scipy polars dask
pytorch tensorflow keras jax scikit-learn sklearn xgboost lightgbm catboost huggingface transformers onnx
opencv nltk spacy cuda tensorrt triton mlflow kubeflow
docker kubernetes k8s helm terraform ansible jenkins gitlab github bitbucket git svn circleci
aws azure gcp lambda ec2 s3 cloudformation prometheus grafana datadog splunk nginx apache linux unix ubuntu
centos debian windows macos ios android
jira confluence figma tableau looker powerbi excel
selenium cypress playwright pytest junit jest mocha
rtos freertos zephyr embedded verilog vhdl systemverilog fpga asic uvm spice cadence altium kicad orcad
labview simulink autocad solidworks catia ansys comsol abaqus creo
ros ros2 gazebo slam lidar
""".split())

# Technical terms that are also common given names. Kept as technical, unless they sit next to a
# name the ladder has already masked ("Julia Venrid").
DUAL_USE_NAMES = frozenset("julia ruby ada crystal pascal swift jade grace".split())

# The most common Chinese surnames (public demographic knowledge), used only by the heuristic
# detector's cue rule for Chinese names.
ZH_COMMON_SURNAMES = frozenset("王李张刘陈杨黄赵吴周徐孙马朱胡郭何林罗高郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤")

# Tokens with punctuation that tokenizers split; matched as indivisible units.
SYMBOL_TOKENS = ("C/C++", "Objective-C", "ASP.NET", "React.js", "Node.js", "Next.js", "Vue.js", "C++", "C#", "F#", ".NET")

# Generic resume and job-ad vocabulary (titles, seniority, functions). Capitalized in headers, so a
# naive name detector proposes them; this list lets the lexicon layer keep them.
RESUME_VOCAB = frozenset("""senior junior lead principal staff chief head intern associate assistant
engineer engineering developer scientist researcher analyst architect manager director consultant specialist
designer administrator technician officer founder co-founder president owner contractor
software hardware backend frontend full-stack fullstack platform infrastructure data machine learning deep
research product program project quality test security network systems embedded firmware mobile cloud
devops sre reliability operations solutions applications application web
team department group division university college institute school laboratory lab technology science sciences
bachelor master doctor degree diploma certificate present current""".split())

QUALIFICATION = frozenset("""licensed licensing qualifications configuration integration simulation validation implementation
architecture experience responsible responsibilities requirements skills programming""".split())

MONTHS = frozenset("""january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec""".split())

DEGREES = frozenset("phd mba meng mphil mph msw mfa med edd pharmd dba dnp dvm dds beng bfa bba bsn bsw llb llm bsc msc".split())
SHORT_DEGREES = frozenset("aa as ba ab bs sb ma am ms sm md jd".split())
TITLE_ABBR = frozenset("sr jr mgr dir vp svp evp assoc asst".split())
JD_PHRASES = ("nice to have", "must have", "good to have", "preferred qualifications", "minimum qualifications")

# Context windows: words near a token that suggest it is technical, or that it is a person.
TECH_CONTEXT = re.compile(r"(?i)\b(?:skills?|programming|framework|library|using|proficient|knowledge|coding|developer|"
                          r"development|engineering|engineer|software|language|tools?|stack)\b|(?:技能|技术|熟悉|熟练|精通|框架|编程|开发|语言)")
PERSON_CONTEXT = re.compile(r"(?i)\b(?:named|contact|mr|mrs|ms|dr|interviewer|recruiter|referee|called|met|spoke|"
                            r"he|she|his|her)\b|(?:姓名|联系人|面试官|顾问|名叫|先生|女士|推荐人)")

SECTION_HEADINGS = {
    "summary": "summary", "profile": "summary", "professional summary": "summary", "objective": "summary",
    "experience": "experience", "work experience": "experience", "professional experience": "experience",
    "employment history": "experience", "工作经历": "experience", "工作经验": "experience",
    "education": "education", "教育背景": "education", "教育经历": "education",
    "skills": "skills", "technical skills": "skills", "programming languages": "skills", "专业技能": "skills", "技能": "skills",
    "projects": "projects", "项目经历": "projects", "certifications": "certifications", "awards": "awards",
    "honors": "awards", "publications": "publications", "selected publications": "publications", "papers": "publications",
    "conference papers": "publications", "journal articles": "publications", "preprints": "publications",
    "talks": "publications", "论文": "publications", "发表论文": "publications", "论文发表": "publications",
    "publications and patents": "publications", "patents": "patents", "patent": "patents", "专利": "patents",
    "references": "references", "professional references": "references", "referees": "references", "推荐人": "references",
}


def load_terms(path: Path) -> frozenset[str]:
    """One term per line; blank lines and lines starting with # are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return frozenset(norm(l) for l in lines if l.strip() and not l.lstrip().startswith("#"))


def dictionary_words(path: Path | None = None) -> frozenset[str]:
    """Lower-case English words, used only to stop partial-token masking of ordinary words.
    Defaults to the system word list when present (common on Unix)."""
    p = Path(path) if path else Path("/usr/share/dict/words")
    if not p.exists():
        return frozenset()
    return frozenset(w for w in p.read_text(encoding="utf-8", errors="ignore").splitlines()
                     if w and w.islower() and w.isalpha())
