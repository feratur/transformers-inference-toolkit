from typing import TYPE_CHECKING, Tuple, Type

from transformers import AutoConfig, AutoTokenizer
from transformers.onnx import FeaturesManager

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase


def load_config(path: "Path") -> "PretrainedConfig":
    return AutoConfig.from_pretrained(path.joinpath("model").as_posix())


def load_tokenizer(path: "Path") -> "PreTrainedTokenizerBase":
    return AutoTokenizer.from_pretrained(path.joinpath("tokenizer").as_posix())


def load_pretrained(
    path: "Path",
    feature: str,
) -> Tuple["PreTrainedTokenizerBase", "PreTrainedModel"]:
    model_class: Type["PreTrainedModel"]
    model_class = FeaturesManager.get_model_class_for_feature(  # type: ignore
        feature=feature,
        framework="pt",
    )
    tokenizer = load_tokenizer(path)
    model_output = model_class.from_pretrained(
        path.joinpath("model").as_posix(),
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    if isinstance(model_output, tuple):
        model = model_output[0]
    else:
        model = model_output
    return tokenizer, model
