import os
from typing import TYPE_CHECKING, Optional

import deepspeed
import torch

if TYPE_CHECKING:
    from deepspeed import InferenceEngine
    from transformers import PreTrainedModel

TORCH_DTYPE_MAPPING = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.int8": torch.int8,
}


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", "1"))


def init_deepspeed_inference(
    model: "PreTrainedModel",
    dtype: str,
    mp_size: Optional[int] = None,
    **kwargs,
) -> "InferenceEngine":
    model = model.cuda(get_local_rank())
    return deepspeed.init_inference(
        model=model,
        dtype=TORCH_DTYPE_MAPPING[dtype],
        mp_size=(get_world_size() if mp_size is None else mp_size),
        **kwargs,
    )
