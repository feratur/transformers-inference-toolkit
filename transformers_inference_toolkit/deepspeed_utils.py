import os
from typing import TYPE_CHECKING

import deepspeed

if TYPE_CHECKING:
    from deepspeed import InferenceEngine
    from transformers import PreTrainedModel


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", "1"))


def init_deepspeed_inference(
    model: "PreTrainedModel",
    **kwargs,
) -> "InferenceEngine":
    model = model.cuda(get_local_rank())
    return deepspeed.init_inference(model, **kwargs)
