from typing import TYPE_CHECKING, Optional

from onnxruntime import InferenceSession
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_model
from transformers.onnx import FeaturesManager, export

from .enums import OnnxModelType, OnnxOptimizationLevel

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
    from transformers.onnx.config import OnnxConfig

ONNX_MODEL_FILE = "model.onnx"
CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"


def export_to_onnx(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    output_path: "Path",
    feature: str = "default",
    for_gpu: bool = True,
    custom_onnx_config: Optional["OnnxConfig"] = None,
) -> "Path":
    if custom_onnx_config:
        onnx_config = custom_onnx_config
    else:
        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            model=model,
            feature=feature,
        )
        onnx_config: "OnnxConfig" = model_onnx_config(model.config)
    model_path = output_path.joinpath(ONNX_MODEL_FILE)
    export(
        preprocessor=tokenizer,  # type: ignore
        model=model,
        config=onnx_config,
        opset=onnx_config.default_onnx_opset,
        output=model_path,
        device=("cuda" if for_gpu else "cpu"),
    )
    return model_path


def optimize_onnx(
    model_path: "Path",
    output_path: "Path",
    model_config: "PretrainedConfig",
    model_type: OnnxModelType = OnnxModelType.BERT,
    for_gpu: bool = True,
    fp16: bool = True,
    optimization_level: OnnxOptimizationLevel = OnnxOptimizationLevel.FULL,
) -> "Path":
    model_config_dict = model_config.to_dict()
    optimizer = optimize_model(
        input=model_path.as_posix(),
        model_type=model_type.value,  # type: ignore
        optimization_options=FusionOptions(model_type.value),
        opt_level=optimization_level.value,  # type: ignore
        use_gpu=for_gpu,
        only_onnxruntime=False,
        num_heads=int(
            model_config_dict.get("num_attention_heads")
            or model_config_dict.get("n_head")
            or model_config_dict.get("n_heads")
            or model_config_dict.get("num_heads")
            or 0
        ),
        hidden_size=int(
            model_config_dict.get("hidden_size")
            or model_config_dict.get("n_embd")
            or model_config_dict.get("dim")
            or 0
        ),
    )
    if fp16:
        optimizer.convert_float_to_float16(keep_io_types=True)
    opt_model_path = output_path.joinpath(ONNX_MODEL_FILE)
    optimizer.save_model_to_file(opt_model_path.as_posix())
    return opt_model_path


def get_cuda_device_idx(session: InferenceSession) -> Optional[int]:
    cuda_provider = session.get_provider_options().get(CUDA_PROVIDER)
    if not cuda_provider:
        return None
    return int(cuda_provider["device_id"])


def get_onnx_session(model_path: "Path", cuda: bool = True) -> InferenceSession:
    provider = CUDA_PROVIDER if cuda else CPU_PROVIDER
    return InferenceSession(model_path.as_posix(), providers=[provider])
