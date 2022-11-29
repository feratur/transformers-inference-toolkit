# Transformers Inference Toolkit
🤗 [Transformers](https://github.com/huggingface/transformers) library provides great API for manipulating pre-trained NLP (as well as CV and Audio-related) models. However, preparing 🤗 Transformers models for use in production usually requires additional effort. The purpose of `transformers-inference-toolkit` is to get rid of boilerplate code and to simplify automatic optimization and inference process of Huggingface Transformers models.

## Installation
Using `pip`:
```bash
pip install transformers-inference-toolkit
```

## Optimization
The original 🤗 Transformers library includes `transformers.onnx` package, which can be used to convert PyTorch or TensorFlow models into [ONNX](https://onnx.ai/) format. This Toolkit extends this functionality by giving the user an opportunity to automatically [optimize ONNX model graph](https://onnxruntime.ai/docs/performance/graph-optimizations.html) - this is similar to what 🤗 [Optimum](https://github.com/huggingface/optimum) library does, but 🤗 Optimum currently has limited support for locally stored pre-trained models as well as for models of less popular architectures (for example, [MPNet](https://github.com/microsoft/MPNet)).

Aside from ONNX conversion the Toolkit also supports resaving PyTorch models with half-precision and setting up [DeepSpeed Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/).

### Prerequisite
The Toolkit expects your pretrained model (in PyTorch format) and tokenizer to be saved (using `save_pretrained()` method) inside a common parent directory in `model` and `tokenizer` folders respectively. This is how a file structure of `toxic-bert` model should look like:
```bash
toxic-bert
├── model
│   ├── config.json
│   └── pytorch_model.bin
└── tokenizer
    ├── merges.txt
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.json
```

### How to use
Most of the popular Transformer model architectures (like BERT and its variations) can be converted with a single command:
```python
from transformers_inference_toolkit import (
    Feature,
    OnnxModelType,
    OnnxOptimizationLevel,
    optimizer,
)

optimizer.pack_onnx(
    input_path="toxic-bert",
    output_path="toxic-bert-optimized",
    feature=Feature.SEQUENCE_CLASSIFICATION,
    for_gpu=True,
    fp16=True,
    optimization_level=OnnxOptimizationLevel.FULL,
)
```
If your model architecture is not supported out-of-the-box (described [here](https://huggingface.co/docs/transformers/serialization)) you can try writing a custom OnnxConfig class:
```python
from collections import OrderedDict
from transformers.onnx import OnnxConfig

class MPNetOnnxConfig(OnnxConfig):
    @property
    def default_onnx_opset(self):
        return 14

    @property
    def inputs(self):
        dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )

optimizer.pack_onnx(
    input_path="all-mpnet-base-v2",
    output_path="all-mpnet-base-v2-optimized",
    feature=Feature.DEFAULT,
    custom_onnx_config_cls=MPNetOnnxConfig,
)
```
ONNX is not the only option, it is also possible to resave the model for future inference simply using PyTorch (`optimizer.pack_transformers()` method, `force_fp16` argument to save in half-precision) or [DeepSpeed Inference](https://www.deepspeed.ai/tutorials/inference-tutorial/) (`optimizer.pack_deepspeed()` method):
```python
optimizer.pack_deepspeed(
    input_path="gpt-neo",
    output_path="gpt-neo-optimized",
    feature=Feature.CAUSAL_LM,
    replace_with_kernel_inject=True,
    mp_size=1,
)
```
After calling `optimizer` methods the model and tokenizer would be saved at `output_path`. The output directory will also contain `metadata.json` file that is necessary for the `Predictor` object (described below) to correctly load the model:
```bash
toxic-bert-optimized
├── metadata.json
├── model
│   ├── config.json
│   └── model.onnx
└── tokenizer
    ├── special_tokens_map.json
    ├── tokenizer.json
    └── tokenizer_config.json
```
## Prediction
After model and tokenizer are packaged using one of the `optimizer` methods, it is possible to initialize a `Predictor` object:
```python
>>> from transformers_inference_toolkit import Predictor
>>> 
>>> predictor = Predictor("toxic-bert-optimized", cuda=True)
>>> print(predictor("I hate this!"))
{'logits': array([[ 0.02940369, -7.0195312 , -4.7890625 , -6.0664062 , -5.625     ,
        -6.09375   ]], dtype=float32)}
```
The `Predictor` object can be simply called with tokenizer arguments (similar to 🤗 Transformers `pipeline`s, `return_tensors` argument can be omitted, `padding` and `truncation` are `True` by default). For text generation tasks `Predictor.generate()` method (with [generation arguments](https://huggingface.co/docs/transformers/main_classes/text_generation)) can be used:
```python
>>> predictor = Predictor("gpt-neo-optimized", cuda=True)
>>> predictor.generate(
...     "Tommy: Hi Mark!",
...     do_sample=True,
...     top_p=0.9,
...     num_return_sequences=3,
...     max_new_tokens=5,
... )
['Tommy: Hi Mark!\nMadelyn: Hello', 'Tommy: Hi Mark! It’s so', 'Tommy: Hi Mark! How are you?\n']
```
