"""Benchmark task definitions for KazMMLU, TUMLU, KazQAD, FLORES-200."""

# Task names in lm-evaluation-harness format
# Install: pip install lm-eval[kaz] or register custom tasks

BENCHMARK_TASKS = {
    "kazmmlu": {
        "task": "kazmmlu",
        "num_fewshot": 5,
        "metric": "acc",
        "description": "KazMMLU: 23K Kazakh knowledge multiple-choice questions",
    },
    "tumlu_mini": {
        "task": "tumlu_mini",
        "num_fewshot": 5,
        "metric": "acc",
        "description": "TUMLU-mini: Turkic language understanding benchmark",
    },
    "kazqad": {
        "task": "kazqad",
        "num_fewshot": 0,
        "metric": "f1",
        "description": "KazQAD: Kazakh open-domain QA",
    },
}

# FLORES-200 evaluated separately via sacrebleu
FLORES_TASKS = [
    {"src_lang": "kaz_Cyrl", "tgt_lang": "eng_Latn", "name": "flores200_kaz_eng"},
    {"src_lang": "kaz_Cyrl", "tgt_lang": "rus_Cyrl", "name": "flores200_kaz_rus"},
]
