from enum import Enum


class OnnxModelType(str, Enum):
    """
    The type of transformer model architecture
    (needed for automatic ONNX optimization).
    """

    BERT = "bert"
    GPT2 = "gpt2"


class OnnxOptimizationLevel(int, Enum):
    """
    ONNX graph optimization level
    (see https://onnxruntime.ai/docs/performance/graph-optimizations.html).
    """

    NONE = 0
    BASIC = 1
    FULL = 99


class ModelFormat(str, Enum):
    """
    The type of packaged model (determines the inference method).
    """

    TRANSFORMERS = "transformers"
    ONNX = "onnx"
    DEEPSPEED = "deepspeed"


class Feature(str, Enum):
    """
    Type of the inference task the model was trained for.
    """

    DEFAULT = "default"
    MASKED_LM = "masked-lm"
    CAUSAL_LM = "causal-lm"
    SEQ2SEQ_LM = "seq2seq-lm"
    SEQUENCE_CLASSIFICATION = "sequence-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    MULTIPLE_CHOICE = "multiple-choice"
    OBJECT_DETECTION = "object-detection"
    QUESTION_ANSWERING = "question-answering"
    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_SEGMENTATION = "image-segmentation"
    MASKED_IM = "masked-im"
    SEMANTIC_SEGMENTATION = "semantic-segmentation"
    VISION2SEQ_LM = "vision2seq-lm"
    SPEECH2SEQ_LM = "speech2seq-lm"
