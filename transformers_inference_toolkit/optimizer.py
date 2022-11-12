import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type

from onnx_enums import OnnxModelType, OnnxOptimizationLevel
from transformers import AutoTokenizer
from transformers.onnx import FeaturesManager, export

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers.onnx.config import OnnxConfig


class Optimizer:
    @staticmethod
    def convert_to_onnx(
        input_path: str,
        feature: str = "default",
        model_type: OnnxModelType = OnnxModelType.BERT,
        for_gpu: bool = True,
        fp16: bool = True,
        optimization_level: OnnxOptimizationLevel = OnnxOptimizationLevel.FULL,
        custom_onnx_config: Optional["OnnxConfig"] = None,
    ):
        model_class: Type[
            "PreTrainedModel"
        ] = FeaturesManager.get_model_class_for_feature(feature)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(input_path, "tokenizer"))
        model_output = model_class.from_pretrained(
            os.path.join(input_path, "model"),
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        if isinstance(model_output, tuple):
            model = model_output[0]
        else:
            model = model_output
        if custom_onnx_config:
            onnx_config = custom_onnx_config
        else:
            _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
                model=model,
                feature=feature,
            )
            onnx_config: "OnnxConfig" = model_onnx_config(model.config)
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir_path = Path(tempdir)
            out_file = tempdir_path.joinpath("model.onnx")
            export(
                preprocessor=tokenizer,  # type: ignore
                model=(
                    model.half()
                    if optimization_level == OnnxOptimizationLevel.NONE and fp16
                    else model
                ),
                config=onnx_config,
                opset=onnx_config.default_onnx_opset,
                output=out_file,
                device=("cuda" if for_gpu else "cpu"),
            )
            if optimization_level != OnnxOptimizationLevel.NONE:
                pass
