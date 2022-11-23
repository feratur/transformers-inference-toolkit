import json
from pathlib import Path
from typing import Optional

import torch

from .deepspeed_utils import init_deepspeed_inference
from .enums import ModelFormat
from .onnx_utils import get_onnx_session
from .transformers_utils import load_config, load_pretrained, load_tokenizer


class Predictor:
    def __init__(self, path: str, cuda: Optional[bool] = None):
        path_obj = Path(path)
        self.path = path_obj.as_posix()
        with path_obj.joinpath("metadata.json").open(mode="r") as meta_file:
            self.metadata = json.load(meta_file)
        self.config = load_config(path_obj)
        self.model_format = ModelFormat(self.metadata["format"])
        if cuda is None:
            self.cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
        else:
            self.cuda = cuda
        if self.model_format == ModelFormat.ONNX:
            self.tokenizer = load_tokenizer(path_obj)
            self.model = get_onnx_session(
                model_path=path_obj.joinpath("model/model.onnx"),
                cuda=self.cuda,
            )
        else:
            self.tokenizer, self.model = load_pretrained(
                path=path_obj,
                feature=self.metadata["feature"],
            )
            if self.model_format == ModelFormat.TRANSFORMERS:
                if self.cuda:
                    self.model = self.model.cuda()
            elif self.model_format == ModelFormat.DEEPSPEED:
                self.model = init_deepspeed_inference(
                    model=self.model,
                    **self.metadata["deepspeed_inference_config"],
                )
            else:
                raise ValueError("Unknown model format")
