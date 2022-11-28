# Transformers Inference Toolkit
🤗 [Transformers](https://github.com/huggingface/transformers) library provides great API for manipulating pre-trained NLP (as well as CV and Audio-related) models. However, preparing 🤗 Transformers models for use in production usually requires additional effort. The purpose of `transformers-inference-toolkit` is to simplify automatic optimization and inference process of Huggingface Transformers-based models.

## Installation
Using `pip`:
```bash
pip install transformers-inference-toolkit
```

## Optimization
The original 🤗 Transformers library includes `transformers.onnx` package, which can be used to convert PyTorch or TensorFlow models into [ONNX](https://onnx.ai/) format. This Toolkit extends this functionality by giving the user an opportunity to automatically [optimize ONNX model graph](https://onnxruntime.ai/docs/performance/graph-optimizations.html) - this is similar to what 🤗 [Optimum](https://github.com/huggingface/optimum) library does, but 🤗 Optimum currently has limited support for locally stored pre-trained models as well as for models of less popular architectures (for example, [MPNet](https://github.com/microsoft/MPNet)).

### Prerequisite
The Toolkit expects your pretrained model to be organized in the following way:
```bash
gpt2
├── model
│   ├── config.json
│   └── pytorch_model.bin
└── tokenizer
    ├── merges.txt
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.json
```

# Work-in-progress
