# (FuriosaAI) How to use optimum.litmus
## Prerequisites
- furiosa-libcompiler >= 0.9.0(See for detailed instructions, https://www.notion.so/furiosa/K8s-Pod-SDK-27680e93c9e9484e9b6f49ad11989c82?pvs=4)

## Installation
```
$ python3 -m venv env
$ . env/bin/activate
$ pip3 install --upgrade pip setuptools wheel
$ pip3 install -e .
```
## Usage
### GPT-Neo
https://huggingface.co/docs/transformers/model_doc/gpt_neo

```
$ python3 -m optimum.litmus.nlp.gpt-neo --help
usage: FuriosaAI litmus GPT Neo using HF Optimum API. [-h] [--model-size {125m,1.3b,2.7b}] [--batch-size BATCH_SIZE] [--input-len INPUT_LEN] [--gen-step GEN_STEP]
                                                      [--task {text-generation-with-past}]
                                                      output_dir

positional arguments:
  output_dir            path to directory to save outputs

optional arguments:
  -h, --help            show this help message and exit
  --model-size {125m,1.3b,2.7b}, -s {125m,1.3b,2.7b}
                        available model sizes
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size for model inputs
  --input-len INPUT_LEN
                        Length of input prommpt
  --gen-step GEN_STEP   Generation step to simplify onnx graph
  --task {text-generation-with-past}
                        Task to export model for
```

### GPT2
https://huggingface.co/docs/transformers/model_doc/gpt2

```
$ python3 -m optimum.litmus.nlp.gpt2 --help
usage: FuriosaAI litmus GPT2 using HF Optimum API. [-h] [--model-size {s,m,l,xl}] [--batch-size BATCH_SIZE] [--input-len INPUT_LEN] [--gen-step GEN_STEP] [--task {text-generation-with-past}]
                                                   output_dir

positional arguments:
  output_dir            path to directory to save outputs

optional arguments:
  -h, --help            show this help message and exit
  --model-size {s,m,l,xl}, -s {s,m,l,xl}
                        available model sizes
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size for model inputs
  --input-len INPUT_LEN
                        Length of input prommpt
  --gen-step GEN_STEP   Generation step to simplify onnx graph
  --task {text-generation-with-past}
                        Task to export model for
```

### OPT
https://huggingface.co/docs/transformers/model_doc/opt
```
usage: FuriosaAI litmus OPT using HF Optimum API. [-h] [--model-size {125m,350m,1.3b,2.7b,6.7b,30b,66b}] [--batch-size BATCH_SIZE] [--input-len INPUT_LEN] [--gen-step GEN_STEP]
                                                  [--task {text-generation-with-past}]
                                                  output_dir

positional arguments:
  output_dir            path to directory to save outputs

options:
  -h, --help            show this help message and exit
  --model-size {125m,350m,1.3b,2.7b,6.7b,30b,66b}, -s {125m,350m,1.3b,2.7b,6.7b,30b,66b}
                        available model sizes
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size for model inputs
  --input-len INPUT_LEN
                        Length of input prommpt
  --gen-step GEN_STEP   Generation step to simplify onnx graph
  --task {text-generation-with-past}
                        Task to export model for
```

### LLaMA
https://huggingface.co/docs/transformers/model_doc/llama
```
$ python3 -m optimum.litmus.nlp.llama --help
usage: FuriosaAI litmus LLaMA using HF Optimum API. [-h] [--model-size {7b,13b,30b,65b}] [--batch-size BATCH_SIZE] [--input-len INPUT_LEN] [--gen-step GEN_STEP]
                                                    [--task {text-generation-with-past}]
                                                    output_dir

positional arguments:
  output_dir            path to directory to save outputs

options:
  -h, --help            show this help message and exit
  --model-size {7b,13b,30b,65b}, -s {7b,13b,30b,65b}
                        available model sizes
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size for model inputs
  --input-len INPUT_LEN
                        Length of input prommpt
  --gen-step GEN_STEP   Generation step to simplify onnx graph
  --task {text-generation-with-past}
                        Task to export model for
```

### toy model
```
(optimum) root@linux-warboy-jasonzcnl2:~/workspace/optimum# python3 -m optimum.litmus.nlp.toy_model --help
usage: FuriosaAI litmus exporting toy model(w/o pretrained weights) using HF Optimum API. [-h] [--config-path CONFIG_PATH] [--batch-size BATCH_SIZE]
                                                                                          [--input-len INPUT_LEN] [--gen-step GEN_STEP]
                                                                                          [--task {text-generation-with-past}]
                                                                                          output_dir

positional arguments:
  output_dir            path to directory to save outputs

options:
  -h, --help            show this help message and exit
  --config-path CONFIG_PATH, -c CONFIG_PATH
                        path to model config saved in json format
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size for model inputs
  --input-len INPUT_LEN
                        Length of input prommpt
  --gen-step GEN_STEP   Generation step to simplify onnx graph
  --task {text-generation-with-past}
                        Task to export model for
```

<details>
<summary>example</summary>

```
  $ python3 -m optimum.litmus.nlp.toy_model toy/gpt2 -c configs/gpt2-toy.json -b 1 --input-len 128 --gen-step 0
  Proceeding model exporting and optimization based given model config:
  {
    "activation_function": "gelu_new",
    "architectures": [
      "GPT2LMHeadModel"
    ],
    "attn_pdrop": 0.1,
    "bos_token_id": 1023,
    "embd_pdrop": 0.1,
    "eos_token_id": 1023,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 3,
    "n_positions": 1024,
    "resid_pdrop": 0.1,
    "summary_activation": null,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": true,
    "summary_type": "cls_index",
    "summary_use_proj": true,
    "task_specific_params": {
      "text-generation": {
        "do_sample": true,
        "max_length": 50
      }
    },
    "vocab_size": 1024,
    "_reference": "https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config"
  }
  Exporting ONNX Model...
  use_past = False is different than use_present_in_outputs = True, the value of use_present_in_outputs value will be used for the outputs.
  Using framework PyTorch: 2.0.1+cu117
  Overriding 2 configuration item(s)
          - use_cache -> True
          - pad_token_id -> 0
  /root/miniconda3/envs/optimum/lib/python3.10/site-packages/transformers/models/gpt2/modeling_gpt2.py:810: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
    if batch_size <= 0:
  ============= Diagnostic Run torch.onnx.export version 2.0.1+cu117 =============
  verbose: False, log level: Level.ERROR
  ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

  Using framework PyTorch: 2.0.1+cu117
  Overriding 2 configuration item(s)
          - use_cache -> True
          - pad_token_id -> 0
  Asked a sequence length of 16, but a sequence length of 1 will be used with use_past == True for `input_ids`.
  ============= Diagnostic Run torch.onnx.export version 2.0.1+cu117 =============
  verbose: False, log level: Level.ERROR
  ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================

  Asked a sequence length of 16, but a sequence length of 1 will be used with use_past == True for `input_ids`.
  Post-processing the exported models...
  Validating ONNX model toy/gpt2/decoder_model_merged.onnx...
          -[✓] ONNX model output names match reference model (present.0.key, present.0.value, present.2.value, present.1.key, present.1.value, present.2.key, logits)
          - Validating ONNX Model output "logits":
                  -[✓] (2, 16, 1024) matches (2, 16, 1024)
                  -[✓] all values close (atol: 1e-05)
          - Validating ONNX Model output "present.0.key":
                  -[✓] (2, 4, 16, 32) matches (2, 4, 16, 32)
                  -[✓] all values close (atol: 1e-05)
          - Validating ONNX Model output "present.0.value":
                  -[✓] (2, 4, 16, 32) matches (2, 4, 16, 32)
                  -[✓] all values close (atol: 1e-05)
          - Validating ONNX Model output "present.1.key":
                  -[✓] (2, 4, 16, 32) matches (2, 4, 16, 32)
                  -[✓] all values close (atol: 1e-05)
          - Validating ONNX Model output "present.1.value":
                  -[✓] (2, 4, 16, 32) matches (2, 4, 16, 32)
                  -[✓] all values close (atol: 1e-05)
          - Validating ONNX Model output "present.2.key":
                  -[✓] (2, 4, 16, 32) matches (2, 4, 16, 32)
                  -[✓] all values close (atol: 1e-05)
          - Validating ONNX Model output "present.2.value":
                  -[✓] (2, 4, 16, 32) matches (2, 4, 16, 32)
                  -[✓] all values close (atol: 1e-05)
  Validating ONNX model toy/gpt2/decoder_model_merged.onnx...
  Asked a sequence length of 16, but a sequence length of 1 will be used with use_past == True for `input_ids`.
          -[✓] ONNX model output names match reference model (present.0.key, present.0.value, present.2.value, present.1.key, present.1.value, present.2.key, logits)
          - Validating ONNX Model output "logits":
                  -[✓] (2, 1, 1024) matches (2, 1, 1024)
                  -[✓] all values close (atol: 1e-05)
          - Validating ONNX Model output "present.0.key":
                  -[✓] (2, 4, 17, 32) matches (2, 4, 17, 32)
                  -[✓] all values close (atol: 1e-05)
          - Validating ONNX Model output "present.0.value":
                  -[✓] (2, 4, 17, 32) matches (2, 4, 17, 32)
                  -[✓] all values close (atol: 1e-05)
          - Validating ONNX Model output "present.1.key":
                  -[✓] (2, 4, 17, 32) matches (2, 4, 17, 32)
                  -[✓] all values close (atol: 1e-05)
          - Validating ONNX Model output "present.1.value":
                  -[✓] (2, 4, 17, 32) matches (2, 4, 17, 32)
                  -[✓] all values close (atol: 1e-05)
          - Validating ONNX Model output "present.2.key":
                  -[✓] (2, 4, 17, 32) matches (2, 4, 17, 32)
                  -[✓] all values close (atol: 1e-05)
          - Validating ONNX Model output "present.2.value":
                  -[✓] (2, 4, 17, 32) matches (2, 4, 17, 32)
                  -[✓] all values close (atol: 1e-05)
  The ONNX export succeeded and the exported model was saved at: toy/gpt2
  Simplifying ONNX Model...
  Checking 1/5...
  Checking 2/5...
  Checking 3/5...
  Checking 4/5...
  Checking 5/5...
  ┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
  ┃                 ┃ Original Model ┃ Simplified Model ┃
  ┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
  │ Add             │ 33             │ 30               │
  │ Cast            │ 11             │ 1                │
  │ Concat          │ 40             │ 0                │
  │ Constant        │ 343            │ 42               │
  │ ConstantOfShape │ 3              │ 0                │
  │ Div             │ 10             │ 10               │
  │ Gather          │ 53             │ 1                │
  │ Gemm            │ 12             │ 12               │
  │ Identity        │ 22             │ 0                │
  │ MatMul          │ 7              │ 7                │
  │ Mul             │ 20             │ 20               │
  │ Pow             │ 13             │ 10               │
  │ Range           │ 1              │ 0                │
  │ ReduceMean      │ 14             │ 14               │
  │ Reshape         │ 40             │ 39               │
  │ Shape           │ 73             │ 0                │
  │ Slice           │ 28             │ 0                │
  │ Softmax         │ 3              │ 3                │
  │ Split           │ 3              │ 3                │
  │ Sqrt            │ 7              │ 7                │
  │ Squeeze         │ 22             │ 0                │
  │ Sub             │ 11             │ 8                │
  │ Tanh            │ 3              │ 3                │
  │ Transpose       │ 15             │ 15               │
  │ Unsqueeze       │ 78             │ 2                │
  │ Where           │ 3              │ 3                │
  │ Model Size      │ 4.9MiB         │ 3.4MiB           │
  └─────────────────┴────────────────┴──────────────────┘
  [1/1] 🔍   Compiling from onnx to dfg
  Done in 0.01256042s
  ✨  Finished in 0.01283372s
```
</details>

### Stable Diffusion
https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_2

```
$ python3 -m optimum.litmus.multimodal.stable-diffusion -h
usage: FuriosaAI litmus Stable Diffusion using HF Optimum API. [-h] --version {1.5,2.1} [--batch-size BATCH_SIZE] [--latent_shape latent_height latent_width] [--input-len INPUT_LEN] output_dir

positional arguments:
  output_dir            path to directory to save outputs

options:
  -h, --help            show this help message and exit
  --version {1.5,2.1}, -v {1.5,2.1}
                        Available model versions
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size for latent and prompt inputs
  --latent_shape latent_height latent_width
                        Shape of latent input. Note it is 1/8 of output image sizes
  --input-len INPUT_LEN
                        Length of input prompt
```

[![ONNX Runtime](https://github.com/huggingface/optimum/actions/workflows/test_onnxruntime.yml/badge.svg)](https://github.com/huggingface/optimum/actions/workflows/test_onnxruntime.yml)

# Hugging Face Optimum

🤗 Optimum is an extension of 🤗 Transformers and Diffusers, providing a set of optimization tools enabling maximum efficiency to train and run models on targeted hardware, while keeping things easy to use.

## Installation

🤗 Optimum can be installed using `pip` as follows:

```bash
python -m pip install optimum
```

If you'd like to use the accelerator-specific features of 🤗 Optimum, you can install the required dependencies according to the table below:

| Accelerator                                                                                                            | Installation                                      |
|:-----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------|
| [ONNX Runtime](https://onnxruntime.ai/docs/)                                                                           | `python -m pip install optimum[onnxruntime]`      |
| [Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html)       | `python -m pip install optimum[neural-compressor]`|
| [OpenVINO](https://docs.openvino.ai/latest/index.html)                                                                 | `python -m pip install optimum[openvino,nncf]`    |
| [Habana Gaudi Processor (HPU)](https://habana.ai/training/)                                                            | `python -m pip install optimum[habana]`           |

To install from source:

```bash
python -m pip install git+https://github.com/huggingface/optimum.git
```

For the accelerator-specific features, append `#egg=optimum[accelerator_type]` to the above command:

```bash
python -m pip install git+https://github.com/huggingface/optimum.git#egg=optimum[onnxruntime]
```

## Accelerated Inference

🤗 Optimum provides multiple tools to export and run optimized models on various ecosystems: 

- [ONNX](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model) / [ONNX Runtime](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/models)
- TensorFlow Lite
- [OpenVINO](https://huggingface.co/docs/optimum/intel/inference)
- Habana first-gen Gaudi / Gaudi2, more details [here](https://huggingface.co/docs/optimum/main/en/habana/usage_guides/accelerate_inference)

The [export](https://huggingface.co/docs/optimum/exporters/overview) and optimizations can be done both programmatically and with a command line.

### Features summary

| Features                           | [ONNX Runtime](https://huggingface.co/docs/optimum/main/en/onnxruntime/overview)| [Neural Compressor](https://huggingface.co/docs/optimum/main/en/intel/optimization_inc)| [OpenVINO](https://huggingface.co/docs/optimum/main/en/intel/inference)| [TensorFlow Lite](https://huggingface.co/docs/optimum/main/en/exporters/tflite/overview)|
|:----------------------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Graph optimization                 | :heavy_check_mark: | N/A                | :heavy_check_mark: | N/A                |
| Post-training dynamic quantization | :heavy_check_mark: | :heavy_check_mark: | N/A                | :heavy_check_mark: |
| Post-training static quantization  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Quantization Aware Training (QAT)  | N/A                | :heavy_check_mark: | :heavy_check_mark: | N/A                |
| FP16 (half precision)              | :heavy_check_mark: | N/A                | :heavy_check_mark: | :heavy_check_mark: |
| Pruning                            | N/A                | :heavy_check_mark: | :heavy_check_mark: | N/A                |
| Knowledge Distillation             | N/A                | :heavy_check_mark: | :heavy_check_mark: | N/A                |


### OpenVINO

This requires to install the OpenVINO extra by doing `pip install optimum[openvino,nncf]`

To load a model and run inference with OpenVINO Runtime, you can just replace your `AutoModelForXxx` class with the corresponding `OVModelForXxx` class. To load a PyTorch checkpoint and convert it to the OpenVINO format on-the-fly, you can set `export=True` when loading your model.

```diff
- from transformers import AutoModelForSequenceClassification
+ from optimum.intel import OVModelForSequenceClassification
  from transformers import AutoTokenizer, pipeline

  model_id = "distilbert-base-uncased-finetuned-sst-2-english"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
- model = AutoModelForSequenceClassification.from_pretrained(model_id)
+ model = OVModelForSequenceClassification.from_pretrained(model_id, export=True)
  model.save_pretrained("./distilbert")

  classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
  results = classifier("He's a dreadful magician.")
```

You can find more examples in the [documentation](https://huggingface.co/docs/optimum/intel/inference) and in the [examples](https://github.com/huggingface/optimum-intel/tree/main/examples/openvino).

### Neural Compressor

This requires to install the Neural Compressor extra by doing `pip install optimum[neural-compressor]`

Dynamic quantization can be applied on your model:

```bash
optimum-cli inc quantize --model distilbert-base-cased-distilled-squad --output ./quantized_distilbert
```

To load a model quantized with Intel Neural Compressor, hosted locally or on the 🤗 hub, you can do as follows :
```python
from optimum.intel import INCModelForSequenceClassification

model_id = "Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-dynamic"
model = INCModelForSequenceClassification.from_pretrained(model_id)
```

You can find more examples in the [documentation](https://huggingface.co/docs/optimum/intel/optimization_inc) and in the [examples](https://github.com/huggingface/optimum-intel/tree/main/examples/neural_compressor).

### ONNX + ONNX Runtime

This requires to install the ONNX Runtime extra by doing `pip install optimum[exporters,onnxruntime]`

It is possible to export 🤗 Transformers models to the [ONNX](https://onnx.ai/) format and perform graph optimization as well as quantization easily:

```plain
optimum-cli export onnx -m deepset/roberta-base-squad2 --optimize O2 roberta_base_qa_onnx
```

The model can then be quantized using `onnxruntime`:

```bash
optimum-cli onnxruntime quantize \
  --avx512 \
  --onnx_model roberta_base_qa_onnx \
  -o quantized_roberta_base_qa_onnx
```

These commands will export `deepset/roberta-base-squad2` and perform [O2 graph optimization](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization#optimization-configuration) on the exported model, and finally quantize it with the [avx512 configuration](https://huggingface.co/docs/optimum/main/en/onnxruntime/package_reference/configuration#optimum.onnxruntime.AutoQuantizationConfig.avx512).

For more information on the ONNX export, please check the [documentation](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model).

#### Run the exported model using ONNX Runtime

Once the model is exported to the ONNX format, we provide Python classes enabling you to run the exported ONNX model in a seemless manner using [ONNX Runtime](https://onnxruntime.ai/) in the backend:

```diff
- from transformers import AutoModelForQuestionAnswering
+ from optimum.onnxruntime import ORTModelForQuestionAnswering
  from transformers import AutoTokenizer, pipeline

  model_id = "deepset/roberta-base-squad2"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
- model = AutoModelForQuestionAnswering.from_pretrained(model_id)
+ model = ORTModelForQuestionAnswering.from_pretrained("roberta_base_qa_onnx")
  qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
  question = "What's Optimum?"
  context = "Optimum is an awesome library everyone should use!"
  results = qa_pipe(question=question, context=context)
```

More details on how to run ONNX models with `ORTModelForXXX` classes [here](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/models).

### TensorFlow Lite

This requires to install the Exporters extra by doing `pip install optimum[exporters-tf]`

Just as for ONNX, it is possible to export models to [TensorFlow Lite](https://www.tensorflow.org/lite) and quantize them:

```plain
optimum-cli export tflite \
  -m deepset/roberta-base-squad2 \
  --sequence_length 384  \
  --quantize int8-dynamic roberta_tflite_model
```

## Accelerated training

🤗 Optimum provides wrappers around the original 🤗 Transformers [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) to enable training on powerful hardware easily.
We support many providers:

- Habana's Gaudi processors
- ONNX Runtime (optimized for GPUs)

### Habana

This requires to install the Habana extra by doing `pip install optimum[habana]`

```diff
- from transformers import Trainer, TrainingArguments
+ from optimum.habana import GaudiTrainer, GaudiTrainingArguments

  # Download a pretrained model from the Hub
  model = AutoModelForXxx.from_pretrained("bert-base-uncased")

  # Define the training arguments
- training_args = TrainingArguments(
+ training_args = GaudiTrainingArguments(
      output_dir="path/to/save/folder/",
+     use_habana=True,
+     use_lazy_mode=True,
+     gaudi_config_name="Habana/bert-base-uncased",
      ...
  )

  # Initialize the trainer
- trainer = Trainer(
+ trainer = GaudiTrainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      ...
  )

  # Use Habana Gaudi processor for training!
  trainer.train()
```

You can find more examples in the [documentation](https://huggingface.co/docs/optimum/habana/quickstart) and in the [examples](https://github.com/huggingface/optimum-habana/tree/main/examples).

### ONNX Runtime

```diff
- from transformers import Trainer, TrainingArguments
+ from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments

  # Download a pretrained model from the Hub
  model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

  # Define the training arguments
- training_args = TrainingArguments(
+ training_args = ORTTrainingArguments(
      output_dir="path/to/save/folder/",
      optim="adamw_ort_fused",
      ...
  )

  # Create a ONNX Runtime Trainer
- trainer = Trainer(
+ trainer = ORTTrainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
+     feature="sequence-classification", # The model type to export to ONNX
      ...
  )

  # Use ONNX Runtime for training!
  trainer.train()
```

You can find more examples in the [documentation](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/trainer) and in the [examples](https://github.com/huggingface/optimum/tree/main/examples/onnxruntime/training).
