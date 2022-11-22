from enum import Enum


class OnnxModelType(str, Enum):
    BERT = "bert"
    GPT2 = "gpt2"


class OnnxOptimizationLevel(int, Enum):
    NONE = 0
    BASIC = 1
    FULL = 99


class ModelFormat(str, Enum):
    TRANSFORMERS = "transformers"
    ONNX = "onnx"
    DEEPSPEED = "deepspeed"
