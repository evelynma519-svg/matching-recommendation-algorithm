"""Vocabulary for the synthetic generator. All of it is generic public knowledge or random syllables."""
from __future__ import annotations

# Person and organization names use disjoint syllables, so no person's name is part of an org name.
SYL = ["ka", "lo", "ven", "ri", "do", "mar", "te", "sil", "an", "zo", "bel", "qui", "ra", "nu", "fen", "or"]
ORG_SYL = ["tek", "nova", "quan", "astr", "vex", "lumi", "cor", "dyn", "pix", "gry", "hel", "syn"]
ORG_SUFFIX = ["Labs", "Systems", "Robotics", "Analytics", "Devices", "Cloud", "Bio", "Energy"]
CITIES = ["Metro A", "Metro B", "Metro C", "Remote"]

# Common Chinese surname and given-name characters (public, generic). Names are random combinations.
ZH_SURNAMES = list("王李张刘陈杨黄赵吴周徐孙马朱胡郭何林罗高")
ZH_GIVEN = list("伟芳娜敏静磊洋勇艳杰涛明超秀霞平刚桂英华慧建军")

# Given names that are also technical terms; they stress the de-identification allow layers.
DUAL_USE_GIVEN = ["Julia", "Ruby", "Ada", "Crystal"]

# Canonical skill -> surface forms. The first form is what a job ad usually writes; resumes vary.
SKILLS = {
    "backend": {"python": ["Python"], "go": ["Go", "Golang"], "java": ["Java"], "postgresql": ["PostgreSQL", "Postgres"],
                "redis": ["Redis", "in-memory caching"], "kafka": ["Kafka", "Apache Kafka", "event streaming"],
                "grpc": ["gRPC", "RPC services"], "docker": ["Docker", "containerization"],
                "kubernetes": ["Kubernetes", "k8s", "container orchestration"], "c#": ["C#"]},
    "data": {"python": ["Python"], "sql": ["SQL"], "spark": ["Spark", "PySpark", "Apache Spark"],
             "airflow": ["Airflow", "workflow orchestration"], "dbt": ["dbt", "analytics engineering"],
             "pandas": ["pandas"], "statistics": ["statistics", "statistical modeling", "statistical inference"],
             "a/b testing": ["A/B testing", "online experimentation", "controlled experiments"]},
    "ml": {"python": ["Python"], "pytorch": ["PyTorch", "Torch"], "deep learning": ["deep learning", "neural networks"],
           "nlp": ["NLP", "natural language processing", "text mining"],
           "computer vision": ["computer vision", "image recognition"], "mlops": ["MLOps", "model deployment"],
           "cuda": ["CUDA", "GPU programming"]},
    "frontend": {"typescript": ["TypeScript"], "react": ["React", "React.js"], "css": ["CSS", "responsive layout"],
                 "graphql": ["GraphQL"], "webpack": ["Webpack", "front-end build tooling"],
                 "accessibility": ["accessibility", "WCAG compliance"], "node.js": ["Node.js"]},
    "embedded": {"c": ["C"], "c++": ["C++"], "rtos": ["RTOS", "FreeRTOS", "real-time operating systems"],
                 "embedded linux": ["embedded Linux", "Yocto"], "can bus": ["CAN bus", "automotive networking"],
                 "fpga": ["FPGA", "Verilog"], "matlab": ["MATLAB", "Simulink"]},
    "hardware": {"pcb design": ["PCB design", "board layout"], "signal integrity": ["signal integrity", "SI analysis"],
                 "analog circuits": ["analog circuits", "analog design"], "altium": ["Altium", "Altium Designer"],
                 "thermal analysis": ["thermal analysis", "thermal simulation"]},
}
FAMILIES = {f: list(s) for f, s in SKILLS.items()}
SURFACE = {k: v for fam in SKILLS.values() for k, v in fam.items()}

ZH_SKILL = {"deep learning": "深度学习", "nlp": "自然语言处理", "computer vision": "计算机视觉", "statistics": "统计建模",
            "a/b testing": "A/B 实验", "pcb design": "PCB 设计", "signal integrity": "信号完整性", "analog circuits": "模拟电路",
            "thermal analysis": "热仿真", "embedded linux": "嵌入式 Linux", "accessibility": "无障碍设计", "mlops": "模型部署",
            "redis": "缓存", "docker": "容器化"}

TITLES = {"backend": "Backend Engineer", "data": "Data Engineer", "ml": "Machine Learning Engineer",
          "frontend": "Frontend Engineer", "embedded": "Embedded Software Engineer", "hardware": "Hardware Engineer"}
ZH_TITLES = {"backend": "后端工程师", "data": "数据工程师", "ml": "机器学习工程师", "frontend": "前端工程师",
             "embedded": "嵌入式软件工程师", "hardware": "硬件工程师"}
LEVELS = ["", "Senior ", "Staff "]
ZH_LEVELS = ["", "高级", "资深"]

RESPONSIBILITIES = {
    "backend": "Design and run the services behind the product; own reliability, latency and data consistency.",
    "data": "Build the pipelines and models the business uses to measure and decide.",
    "ml": "Train, evaluate and ship models into production systems.",
    "frontend": "Build fast, accessible interfaces used by customers every day.",
    "embedded": "Write firmware for devices with tight timing, memory and power budgets.",
    "hardware": "Design boards from schematic to bring-up and validation.",
}
VERBS = ["Built and operated", "Designed", "Maintained", "Scaled", "Led the rewrite of"]
ZH_VERBS = ["负责", "主导", "参与", "设计并实现"]
OBJECTS = ["the core services", "the reporting stack", "the device platform", "the customer-facing app", "the data platform"]
