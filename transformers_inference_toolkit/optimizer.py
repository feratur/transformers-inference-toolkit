import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from .deepspeed_utils import get_world_size, init_deepspeed_inference
from .enums import ModelFormat, OnnxModelType, OnnxOptimizationLevel
from .onnx_utils import export_to_onnx, get_onnx_session, optimize_onnx
from .transformers_utils import load_pretrained, save_pretrained

if TYPE_CHECKING:
    from transformers.onnx.config import OnnxConfig


def pack_deepspeed(
    input_path: str,
    output_path: str,
    feature: str = "default",
    dtype: Optional[torch.dtype] = None,
    replace_with_kernel_inject: bool = True,
    tensor_parallel: Optional[int] = None,
    enable_cuda_graph: bool = False,
    replace_method: str = "auto",
):
    tokenizer, model = load_pretrained(Path(input_path), feature)
    ds_inference_config = dict(
        dtype=(model.dtype if dtype is None else dtype),
        replace_with_kernel_inject=replace_with_kernel_inject,
        tensor_parallel=(
            get_world_size() if tensor_parallel is None else tensor_parallel
        ),
        enable_cuda_graph=enable_cuda_graph,
        replace_method=replace_method,
    )
    init_deepspeed_inference(model, **ds_inference_config)
    out_path = Path(output_path)
    metadata = dict(
        format=ModelFormat.DEEPSPEED.value,
        feature=feature,
        deepspeed_inference_config=ds_inference_config,
    )
    save_pretrained(out_path, metadata, tokenizer, model)


def pack_transformers(
    input_path: str,
    output_path: str,
    feature: str = "default",
    force_fp16: bool = True,
):
    tokenizer, model = load_pretrained(Path(input_path), feature)
    if force_fp16:
        model = model.half()
    out_path = Path(output_path)
    metadata = dict(
        format=ModelFormat.TRANSFORMERS.value,
        feature=feature,
        force_fp16=force_fp16,
    )
    save_pretrained(out_path, metadata, tokenizer, model)


def pack_onnx(
    input_path: str,
    output_path: str,
    feature: str = "default",
    model_type: OnnxModelType = OnnxModelType.BERT,
    for_gpu: bool = True,
    fp16: bool = True,
    optimization_level: OnnxOptimizationLevel = OnnxOptimizationLevel.FULL,
    custom_onnx_config: Optional["OnnxConfig"] = None,
):
    pretrained_input_path = Path(input_path)
    tokenizer, model = load_pretrained(pretrained_input_path, feature)
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        if fp16 and optimization_level == OnnxOptimizationLevel.NONE:
            model = model.half()
        onnx_model_path = export_to_onnx(
            model=model,
            tokenizer=tokenizer,
            output_path=tempdir_path,
            feature=feature,
            for_gpu=for_gpu,
            custom_onnx_config=custom_onnx_config,
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
        model_file_path = model_dir_path.joinpath("model.onnx")
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
        feature=feature,
        model_type=model_type.value,
        for_gpu=for_gpu,
        fp16=fp16,
        optimization_level=optimization_level.value,
        custom_onnx_config_used=bool(custom_onnx_config),
    )
    save_pretrained(out_path, metadata, tokenizer)
