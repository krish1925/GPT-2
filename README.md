# GPT-2 Implemented in Pytorch

## Description

In this project, I built baseline bigram language model and try to improve the performance of that model with a transformer-based decoder only language model. As this had to be feasible to run on a macbook m1, I trained these models on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset and tried to improve the generation abilities of the language model.

To compare the performance of a reasonable trained model vs the baseline I used model weights for a 20M parameter model trained on the same dataset, thus not requiring to train the entire model from scratch.

## Installation

To install the required packages, run the following command:

```bash
pip install numpy torch tiktoken wandb einops
```

where:

1. `numpy` is a library for numerical computing. **Required**.
2. `torch` is the PyTorch library. **Required**.
3. `tiktoken` is a library for tokenizing text data. **Required**.
4. `wandb` is the Weights and Biases library for tracking experiments, it is optional but recommended.
5. `einops` is a library for tensor manipulation, it is fantastic and I highly recommend it but is not required for this project.

No other other external libraries (like `transformers` or `torchtext`) were used for this project.

Minor imports (like torch.var or torch.std) are used, but not big ones like (nn.TransformerEncoderLayer).

## Dataset

Dataset is downloaded and split into train and validation, which can be seen in the `data` folder.

## Pretrained Models

Pretrained models are located provided in the zipped file, if not you can download them from the link below, **IGNORE BELOW IF PROVIDED**.

Download all pretrained models from [this link](https://drive.google.com/file/d/1g09qUM9WibdfQVgkj6IAj8K2S3SGwc91/view?usp=sharing)

After downloading the pretrained model, add the folder to the root directory. All the pretrained models should be in the `pretrained_models` folder in the MiniGPT folder.
