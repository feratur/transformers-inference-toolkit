import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from .deepspeed_utils import init_deepspeed_inference
from .enums import ModelFormat
from .onnx_utils import ONNX_MODEL_FILE, get_cuda_device_idx, get_onnx_session
from .transformers_utils import (
    METADATA_FILE,
    load_config,
    load_pretrained,
    load_tokenizer,
)

if TYPE_CHECKING:
    from transformers import BatchEncoding


class Predictor:
    def __init__(self, path: str, cuda: Optional[bool] = None):
        path_obj = Path(path)
        self.path = path_obj.as_posix()
        with path_obj.joinpath(METADATA_FILE).open(mode="r") as meta_file:
            self.metadata = json.load(meta_file)
        self.config = load_config(path_obj)
        self.model_format = ModelFormat(self.metadata["format"])
        cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        if self.model_format == ModelFormat.ONNX:
            self.tokenizer = load_tokenizer(path_obj)
            self.model = get_onnx_session(
                model_path=path_obj.joinpath(f"model/{ONNX_MODEL_FILE}"),
                cuda=(self.metadata["for_gpu"] if cuda is None else cuda_available),
            )
            device_idx = get_cuda_device_idx(self.model)
            self.device = (
                torch.device("cpu")
                if device_idx is None
                else torch.device(f"cuda:{device_idx}")
            )
        else:
            self.tokenizer, self.model = load_pretrained(
                path=path_obj,
                feature=self.metadata["feature"],
            )
            if self.model_format == ModelFormat.TRANSFORMERS:
                if cuda is None and cuda_available or cuda:
                    self.model = self.model.cuda()
                self.device = self.model.device
            elif self.model_format == ModelFormat.DEEPSPEED:
                self.model = init_deepspeed_inference(
                    model=self.model,
                    **self.metadata["deepspeed_inference_config"],
                )
                self.device = self.model.module.device
            else:
                raise ValueError("Unknown model format")

    def tokenize(
        self,
        *args,
        return_tensors: Optional[str] = None,
        padding: bool = True,
        truncation: bool = True,
        **kwargs,
    ) -> "BatchEncoding":
        return self.tokenizer(
            *args,
            return_tensors=(
                return_tensors
                if return_tensors
                else ("np" if self.model_format == ModelFormat.ONNX else "pt")
            ),
            padding=padding,
            truncation=truncation,
            **kwargs,
        )

    def predict(self, model_input: "BatchEncoding", **kwargs) -> Any:
        if self.model_format == ModelFormat.ONNX:
            outputs = self.model.run(
                None,
                input_feed={
                    key: model_input[key] for key in self.metadata["model_inputs"]
                },
                **kwargs,
            )
            return {
                out_name: torch.from_numpy(out_value)
                for out_name, out_value in zip(self.metadata["model_outputs"], outputs)
            }
        with torch.no_grad():
            return self.model(**model_input.to(self.device), **kwargs)

    def generate(
        self,
        prompt: str,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        **kwargs,
    ) -> List[str]:
        model_input = self.tokenize(prompt, padding=False).to(self.device)
        with torch.no_grad():
            model_output = self.model.generate(**model_input, **kwargs)
        return self.tokenizer.batch_decode(
            sequences=model_output,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def __call__(self, *args, **kwargs) -> Any:
        return self.predict(self.tokenize(*args, **kwargs))
