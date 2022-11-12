from enum import Enum


class OnnxModelType(str, Enum):
    BERT = "bert"
    BART = "bart"
    GPT2 = "gpt2"


class OnnxOptimizationLevel(int, Enum):
    NONE = 0
    BASIC = 1
    FULL = 99
