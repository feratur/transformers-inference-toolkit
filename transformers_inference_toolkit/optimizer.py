import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type

import torch

from .deepspeed_utils import get_world_size, init_deepspeed_inference
from .enums import Feature, ModelFormat, OnnxModelType, OnnxOptimizationLevel
from .onnx_utils import export_to_onnx, get_onnx_session, optimize_onnx
from .transformers_utils import load_pretrained, save_pretrained

if TYPE_CHECKING:
    from transformers.onnx.config import OnnxConfig


def pack_deepspeed(
    input_path: str,
    output_path: str,
    feature: Feature = Feature.DEFAULT,
    dtype: Optional[torch.dtype] = None,
    replace_with_kernel_inject: bool = True,
    mp_size: Optional[int] = None,
    enable_cuda_graph: bool = False,
):
    """
    Save the model and tokenizer for DeepSpeed Inference.

    :param input_path: Path to the folder containing "model" and
        "tokenizer" directories.
    :param output_path: Path where to save the model and tokenizer package.
    :param feature: The type of specific inference task.
    :param dtype: Desired model data type (leave at None to use model's
        original dtype).
    :param replace_with_kernel_inject: Set to True to inject inference kernels
        for supported models.
    :param mp_size: Desired model parallel size, value 1 meaning no model
        parallelism (leave at None to use all available GPUs).
    :param enable_cuda_graph: The flag for capturing the CUDA-Graph of the
        inference ops.
    :return: None.
    """
    feature_str = str(feature.value)
    tokenizer, model = load_pretrained(Path(input_path), feature_str)
    ds_inference_config = dict(
        dtype=str(model.dtype if dtype is None else dtype),
        replace_with_kernel_inject=replace_with_kernel_inject,
        mp_size=(get_world_size() if mp_size is None else mp_size),
        enable_cuda_graph=enable_cuda_graph,
        replace_method="auto",
    )
    out_path = Path(output_path)
    metadata = dict(
        format=ModelFormat.DEEPSPEED.value,
        feature=feature_str,
        deepspeed_inference_config=ds_inference_config,
    )
    save_pretrained(out_path, metadata, tokenizer, model)
    init_deepspeed_inference(model, **ds_inference_config)


def pack_transformers(
    input_path: str,
    output_path: str,
    feature: Feature = Feature.DEFAULT,
    force_fp16: bool = False,
):
    """
    Save the model and tokenizer without applying any optimizations.

    :param input_path: Path to the folder containing "model" and
        "tokenizer" directories.
    :param output_path: Path where to save the model and tokenizer package.
    :param feature: The type of specific inference task.
    :param force_fp16: True to save the model in half-precision.
    :return: None.
    """
    feature_str = str(feature.value)
    tokenizer, model = load_pretrained(Path(input_path), feature_str)
    if force_fp16:
        model = model.half()
    out_path = Path(output_path)
    metadata = dict(
        format=ModelFormat.TRANSFORMERS.value,
        feature=feature_str,
        force_fp16=force_fp16,
    )
    save_pretrained(out_path, metadata, tokenizer, model)


def pack_onnx(
    input_path: str,
    output_path: str,
    feature: Feature = Feature.DEFAULT,
    model_type: OnnxModelType = OnnxModelType.BERT,
    for_gpu: bool = True,
    fp16: bool = True,
    optimization_level: OnnxOptimizationLevel = OnnxOptimizationLevel.FULL,
    custom_onnx_config_cls: Optional[Type["OnnxConfig"]] = None,
):
    """
    Export to ONNX format (optionally with ONNX optimization).

    :param input_path: Path to the folder containing "model" and
        "tokenizer" directories.
    :param output_path: Path where to save the model and tokenizer package.
    :param feature: The type of specific inference task.
    :param model_type: Basic type of Transformer model architecture
        (only needed for optimization).
    :param for_gpu: True if the inference will be performed on GPU.
    :param fp16: True to use half-precision.
    :param optimization_level: ONNX optimization level
        (NONE to skip optimization, FULL uses hardware-dependent optimizations).
    :param custom_onnx_config_cls: If the ONNX conversion is not
        supported for some particular model by the Transformers
        library a custom OnnxConfig class can be provided.
    :return: None.
    """
    feature_str = str(feature.value)
    pretrained_input_path = Path(input_path)
    tokenizer, model = load_pretrained(pretrained_input_path, feature_str)
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        if fp16 and optimization_level == OnnxOptimizationLevel.NONE:
            model = model.half()
        onnx_model_path = export_to_onnx(
            model=model,
            tokenizer=tokenizer,
            output_path=tempdir_path,
            feature=feature_str,
            for_gpu=for_gpu,
            custom_onnx_config=(
                custom_onnx_config_cls
                and custom_onnx_config_cls(model.config, feature_str)
            ),
        )
        if optimization_level != OnnxOptimizationLevel.NONE:
            opt_dir_path = tempdir_path.joinpath("optimized")
            opt_dir_path.mkdir(parents=True, exist_ok=True)
            onnx_model_path = optimize_onnx(
                model_path=onnx_model_path,
                output_path=opt_dir_path,
                model_config=model.config,
                model_type=model_type,
                for_gpu=for_gpu,
                fp16=(fp16 and model.dtype != torch.float16),
                optimization_level=optimization_level,
            )
        out_path = Path(output_path)
        model_dir_path = out_path.joinpath("model")
        model_dir_path.mkdir(parents=True, exist_ok=True)
        model_file_path = model_dir_path.joinpath(onnx_model_path.name)
        onnx_model_path.rename(model_file_path)
        config_file = pretrained_input_path.joinpath("model/config.json")
        if config_file.exists():
            shutil.copyfile(
                src=config_file.as_posix(),
                dst=model_dir_path.joinpath("config.json").as_posix(),
            )
    onnx_session = get_onnx_session(model_file_path, cuda=for_gpu)
    metadata = dict(
        format=ModelFormat.ONNX.value,
        model_inputs=[inp.name for inp in onnx_session.get_inputs()],
        model_outputs=[out.name for out in onnx_session.get_outputs()],
        feature=feature_str,
        model_type=model_type.value,
        for_gpu=for_gpu,
        fp16=fp16,
        optimization_level=optimization_level.value,
        custom_onnx_config_used=bool(custom_onnx_config_cls),
    )
    save_pretrained(out_path, metadata, tokenizer)
